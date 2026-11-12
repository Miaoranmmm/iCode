# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Manually extract durable memories from ended host snapshots in one process."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from chrys.app.memory_hosts.context import EMPTY_CATALOG
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, extraction_tool_schemas, prepare_profile
from chrys.app.memory_hosts.extraction_freeze import FreezeInputs, freeze_regression_set, validate_destination
from chrys.app.memory_hosts.extraction_replay import (
    FrozenSource,
    ReplayManifest,
    load_manifest,
    text_digest,
    verify_frozen_scan,
)
from chrys.app.memory_hosts.extraction_review import ReviewLists, ReviewSnapshot, load_review_snapshot
from chrys.app.memory_hosts.extraction_tasks import OperationNotAdmitted, OperationOwner
from chrys.foundation.config.settings import Settings
from chrys.foundation.platform import get_platform
from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.foundation.util.once_close import finish_close
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.extraction import (
    DEFAULT_MATERIAL_PROFILE,
    MAX_SESSIONS,
    ExtractionBatch,
    ExtractionError,
    ExtractionExecutor,
    MaterialProfile,
    ProjectRunLock,
    ScanDiagnostic,
    ScanResult,
    SessionKey,
    SourceIndex,
    SourceSession,
    WatermarkSnapshot,
    WatermarkStore,
    build_batches,
    candidate_status,
    classify_result,
    conflicting_sessions,
    material_budget,
    project_sources,
    render_prompt,
    tuned_material_profile,
)
from chrys.service.memory.operations import MemoryReader
from chrys.service.memory.project import bind_project
from chrys.service.memory.render import render_catalog
from chrys.service.memory.storage.file_change_log import FileChangeLog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from chrys.service.memory.tuning import V0, MemoryTuning, load_active, load_candidate, load_version
from chrys.service.memory.verification import ConversationView
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.resolver import resolve_selectable_profile
from chrys.service.profiles.models.schema import ModelProfile

_WATERMARK_PHASES = frozenset({"attempt", "success", "incomplete_limits"})


class HistoryReader(Protocol):
    @property
    def paths(self) -> Mapping[SessionKey, Path]: ...

    async def scan(self, project: MemoryScope) -> ScanResult: ...

    async def read_files(self, project: MemoryScope, paths: tuple[Path, ...]) -> ScanResult: ...

    async def revalidate(self, source: SourceSession) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ExtractionOptions:
    project_dir: Path
    memory_dir: Path
    claude_source: Path | None = None
    codex_source: Path | None = None
    sessions: frozenset[str] = frozenset()
    dry_run: bool = False
    replay: Path | None = None
    max_turns: int | None = None
    max_output_tokens: int | None = None
    batch_timeout_seconds: float | None = None
    freeze_to: Path | None = None
    baseline_dir: Path | None = None
    tuning: str | None = None


@dataclass(slots=True)
class RunSignals:
    """Candidate coverage and structural annotations, without source content."""

    candidate_sessions: int = 0
    annotated_sessions: int = 0
    annotations: dict[str, int] = field(default_factory=lambda: {"user_interrupt": 0})
    signals_unavailable_sessions: int = 0


def _source_signals(sources: Sequence[SourceSession], unavailable: frozenset[SessionKey] = frozenset()) -> RunSignals:
    signals = RunSignals(
        candidate_sessions=len(sources),
        signals_unavailable_sessions=len({source.key for source in sources} & unavailable),
    )
    for source in sources:
        if source.annotations:
            signals.annotated_sessions += 1
        for annotation in source.annotations:
            signals.annotations[annotation.kind] = signals.annotations.get(annotation.kind, 0) + 1
    return signals


@dataclass(slots=True)
class RunReport:
    """Only metadata is emitted; model answers, inputs, and exceptions stay private."""

    run_id: str = field(default_factory=lambda: "extract-" + uuid4().hex)
    exit_code: int = 0
    status: str = "completed"
    profile: dict[str, str | int | float] = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    signals: RunSignals = field(default_factory=RunSignals)
    baseline_counts: dict[str, int] = field(default_factory=dict)
    batches: list[dict[str, object]] = field(default_factory=list)
    conflict_count: int = 0
    not_started_batches: int = 0
    cancel_requested: bool = False
    elapsed_seconds: float = 0
    tuning_version: str = "v0"
    material_changed: bool = False
    request_accounting_note: str = "Adapter-internal thinking-binding resends and SDK retries (default 2) are not counted by the guard or usage totals."


def _check_cancelled(owner: OperationOwner) -> None:
    if owner.cancel_requested:
        raise OperationNotAdmitted()


def _settle_exception(report: RunReport, summary: dict[str, object] | None, phase: str, owner: OperationOwner) -> None:
    # Cancellation of reads closes admission even if their drained task raises.
    # Admitted writes retain their actual durable outcome instead.
    if owner.cancel_requested and phase not in _WATERMARK_PHASES:
        saved = bool(summary and summary.get("saved"))
        report.cancel_requested = True
        report.status, report.exit_code = ("saved_then_cancelled" if saved else "cancelled_before_" + phase), 130
    if summary is not None:
        summary["status"] = report.status
        summary["failed_phase"] = phase


def _references(project_dir: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, path.read_text(encoding="utf-8"))
        for name in ("AGENTS.md", "CLAUDE.md", "CLAUDE.local.md")
        if (path := project_dir / name).is_file()
    )


def _readers(options: ExtractionOptions, replay: ReplayManifest | None) -> dict[str, HistoryReader]:
    from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader
    from chrys.app.memory_hosts.codex_history import CodexHistoryReader

    readers: dict[str, HistoryReader] = {}
    hosts = {source.key.host for source in replay.sources} if replay else set()
    frozen_sessions = frozenset(source.key.session for source in replay.sources) if replay else frozenset()
    if replay and options.sessions and options.sessions != frozen_sessions:
        raise ExtractionError("invalid_replay", "Replay requires the frozen session membership.")
    fallback = options.replay.parent if options.replay is not None else None
    if options.claude_source is not None or "claude-code" in hosts:
        source_dir = options.claude_source or fallback
        if source_dir is None:
            raise ExtractionError("source_required", "A Claude source directory is required.")
        readers["claude-code"] = ClaudeHistoryReader(
            source_dir,
            sessions=frozen_sessions if replay else options.sessions,
            replay=replay is not None,
        )
    if options.codex_source is not None or "codex" in hosts:
        source_dir = options.codex_source or fallback
        if source_dir is None:
            raise ExtractionError("source_required", "A Codex source directory is required.")
        readers["codex"] = CodexHistoryReader(
            source_dir,
            sessions=frozen_sessions if replay else options.sessions,
            replay=replay is not None,
        )
    return readers


async def _scan(
    project: MemoryScope,
    readers: Mapping[str, HistoryReader],
    replay: ReplayManifest | None,
    owner: OperationOwner,
    report: RunReport,
) -> ScanResult:
    sources: list[SourceSession] = []
    diagnostics: list[ScanDiagnostic] = []
    fatal = False
    for host, reader in readers.items():
        if replay is None:
            scanned = await owner.run(reader.scan(project))
        else:
            paths = tuple(source.path for source in replay.sources if source.key.host == host)
            scanned = await owner.run(reader.read_files(project, paths))
        _check_cancelled(owner)
        report.diagnostics.extend(asdict(diagnostic) for diagnostic in scanned.diagnostics)
        diagnostics.extend(scanned.diagnostics)
        sources.extend(scanned.sessions)
        fatal = fatal or scanned.fatal
    if fatal:
        raise ExtractionError("source_scan_failed", "A source could not be scanned safely.")
    collisions = conflicting_sessions(sources)
    if collisions:
        report.diagnostics.extend(
            {"kind": "session_id_collision", "session": session} for session in sorted(collisions)
        )
        sources = [source for source in sources if source.key.session not in collisions]
    result = ScanResult(tuple(sources), tuple(diagnostics))
    return verify_frozen_scan(replay, result) if replay is not None else result


def _model_input(
    batch: ExtractionBatch,
    catalog: str,
    references: tuple[tuple[str, str], ...],
    material_profile: MaterialProfile,
    review: ReviewLists,
) -> str:
    """The text one extraction call sends: instructions, rendered prompt and tool schemas."""
    return (
        material_profile.instructions
        + render_prompt(
            batch,
            catalog,
            references,
            profile=material_profile,
            related_entries=review.related_entries,
            withdrawn_entries=review.withdrawn_entries,
        )
        + json.dumps(extraction_tool_schemas(material_profile), ensure_ascii=False)
    )


def _budget(
    profile: ModelProfile,
    project: MemoryScope,
    catalog: str,
    references: tuple[tuple[str, str], ...],
    material_profile: MaterialProfile = DEFAULT_MATERIAL_PROFILE,
    review: ReviewLists | None = None,
    tuning: MemoryTuning = V0,
) -> int:
    empty = ExtractionBatch(project, (), "", SourceIndex(ConversationView((), None)))
    fixed = _model_input(empty, catalog, references, material_profile, review or ReviewLists())
    try:
        return material_budget(
            profile.max_context_tokens, profile.max_output_tokens, fixed, max_material_tokens=tuning.material_budget
        )
    except ExtractionError as error:
        details = (
            ", ".join(
                f"{name}: {MixedLanguageTokenizer().count_tokens(text)} estimated tokens" for name, text in references
            )
            or "no reference files"
        )
        raise ExtractionError(error.kind, f"Fixed input exceeds the window ({details}).") from error


def _select(
    sources: tuple[SourceSession, ...], snapshot: WatermarkSnapshot, project_id: str, report: RunReport
) -> tuple[SourceSession, ...]:
    eligible: list[SourceSession] = []
    for source in sources:
        status = candidate_status(source, snapshot.record(project_id, source.key))
        if status == "candidate":
            eligible.append(source)
        else:
            report.diagnostics.append({"kind": status, "host": source.key.host, "session": source.key.session})
    if any(item["kind"] == "source_rewritten" for item in report.diagnostics):
        raise ExtractionError("source_rewritten", "Source content changed while its modification time moved backwards.")
    return tuple(eligible)


async def run_extraction(
    options: ExtractionOptions,
    profile: ModelProfile,
    *,
    readers: Mapping[str, HistoryReader] | None = None,
    executor: ExtractionExecutor | None = None,
    material_profile: MaterialProfile = DEFAULT_MATERIAL_PROFILE,
    tuning: MemoryTuning | None = None,
) -> RunReport:
    """Preflight, revalidate, record an attempt, run, and commit only proven outcomes."""
    report = RunReport()
    owner = OperationOwner()
    lock: ProjectRunLock | None = None
    phase = "preflight"
    current_summary: dict[str, object] | None = None
    freeze_inputs: FreezeInputs | None = None
    started = time.monotonic()
    try:
        if (options.freeze_to is not None and (not options.dry_run or options.replay is not None)) or (
            options.baseline_dir is not None and options.freeze_to is None
        ):
            raise ExtractionError(
                "invalid_freeze", "Freeze requires dry-run, excludes replay, and owns baseline input."
            )
        phase = "replay_preflight"
        manifest = await owner.run(asyncio.to_thread(load_manifest, options.replay)) if options.replay else None
        _check_cancelled(owner)
        if options.tuning is not None and manifest is None:
            raise ExtractionError("invalid_tuning", "A candidate tuning requires frozen replay inputs.")
        phase = "tuning"
        config_dir = get_platform().config_dir
        try:
            if options.tuning is not None:
                tuning = await owner.run(asyncio.to_thread(load_candidate, config_dir, options.tuning))
            elif manifest is not None:
                tuning = await owner.run(asyncio.to_thread(load_version, config_dir, manifest.tuning_version))
            elif tuning is None:
                tuning = await owner.run(asyncio.to_thread(load_active, config_dir))
        except ValueError as error:
            raise ExtractionError("invalid_tuning", "The requested memory tuning is unavailable or invalid.") from error
        _check_cancelled(owner)
        report.tuning_version = tuning.version
        if options.tuning is not None and not options.tuning.removeprefix("v").isdecimal():
            report.tuning_version = "file:" + text_digest(json.dumps(asdict(tuning), sort_keys=True))
        report.material_changed = options.tuning is not None
        strict = manifest is not None and options.tuning is None
        options = replace(
            options,
            max_turns=tuning.max_turns if options.max_turns is None else options.max_turns,
            max_output_tokens=tuning.max_output_tokens
            if options.max_output_tokens is None
            else options.max_output_tokens,
            batch_timeout_seconds=tuning.batch_timeout_seconds
            if options.batch_timeout_seconds is None
            else options.batch_timeout_seconds,
        )
        max_turns = options.max_turns if options.max_turns is not None else tuning.max_turns
        max_output = options.max_output_tokens if options.max_output_tokens is not None else tuning.max_output_tokens
        timeout_seconds = (
            options.batch_timeout_seconds if options.batch_timeout_seconds is not None else tuning.batch_timeout_seconds
        )
        phase = "preflight"
        prepared_profile = prepare_profile(profile, max_output)
        effective_profile = prepared_profile.profile
        report.profile = {
            "id": effective_profile.id,
            "name": effective_profile.name,
            "provider": effective_profile.provider,
            "api_style": effective_profile.api_style,
            "wire": prepared_profile.wire,
            "model_id": effective_profile.model_id,
            "max_turns": max_turns,
            "max_output_tokens": effective_profile.max_output_tokens,
            "batch_timeout_seconds": timeout_seconds,
        }
        if max_turns <= 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ExtractionError("invalid_limits", "Turns and timeout must be positive.")
        phase = "project"
        project = await owner.run(asyncio.to_thread(bind_project, str(options.project_dir)))
        _check_cancelled(owner)
        project_id = project.project_id
        if project_id is None:
            raise ExtractionError("project_unavailable", "Extraction needs a bound project.")
        protected_roots = tuple(
            path
            for path in (
                get_platform().config_dir / "memory",
                options.memory_dir,
                options.baseline_dir,
                options.claude_source,
                options.codex_source,
                options.project_dir,
                Path(project.root) if project.root is not None else None,
            )
            if path is not None
        )
        if options.freeze_to is not None:
            phase = "freeze_preflight"
            await owner.run(
                asyncio.to_thread(validate_destination, options.freeze_to, options.memory_dir, protected_roots)
            )
            _check_cancelled(owner)
        phase = "replay_preflight"
        if manifest is not None:
            material_profile = manifest.material_profile
            formal_root = get_platform().config_dir / "memory"
            isolated = await owner.run(asyncio.to_thread(options.memory_dir.resolve))
            _check_cancelled(owner)
            formal = await owner.run(asyncio.to_thread(formal_root.resolve))
            _check_cancelled(owner)
            if isolated == formal or manifest.project_id != project_id:
                raise ExtractionError(
                    "invalid_replay", "Replay needs the frozen project and a separate memory directory."
                )
            await owner.run(asyncio.to_thread(manifest.verify_inputs, options.memory_dir))
            _check_cancelled(owner)
            phase = "references"
            references = await owner.run(asyncio.to_thread(manifest.read_references))
        else:
            phase = "references"
            references = await owner.run(asyncio.to_thread(_references, options.project_dir))
        _check_cancelled(owner)
        material_profile = tuned_material_profile(material_profile, tuning)
        report.profile.update(
            material_profile=material_profile.name,
            render_version=material_profile.render_version,
            prompt_version=material_profile.prompt_version,
            tuning_version=report.tuning_version,
        )
        active_readers = readers if readers is not None else _readers(options, manifest)
        if not active_readers:
            raise ExtractionError("source_required", "Specify a source directory or a frozen replay manifest.")
        phase = "scan"
        scanned = await _scan(project, active_readers, manifest, owner, report)
        sources = scanned.sessions
        watermarks = WatermarkStore(options.memory_dir)
        phase = "watermark_read"
        snapshot = WatermarkSnapshot() if manifest else await owner.run(watermarks.load())
        _check_cancelled(owner)
        selected = project_sources(
            sources if manifest else _select(sources, snapshot, project_id, report), material_profile
        )
        report.signals = _source_signals(
            selected,
            frozenset(
                note.session
                for note in scanned.diagnostics
                if note.kind == "signals_unavailable" and note.session is not None
            ),
        )
        store = FileMemoryStore(options.memory_dir)

        async def refresh_budget(sources: tuple[SourceSession, ...]) -> tuple[int, str, ReviewSnapshot]:
            """Read the current catalog and review lists, then size the material that still fits."""
            catalog = (
                render_catalog(await owner.run(MemoryReader(store, project).catalog()), tuning=tuning) or EMPTY_CATALOG
            )
            _check_cancelled(owner)
            review_snapshot = (
                await owner.run(load_review_snapshot(store, project))
                if material_profile.tool_schema == "engineering"
                else ReviewSnapshot()
            )
            _check_cancelled(owner)
            lists = review_snapshot.lists(sources, budget_bound=True, related_limit=tuning.related_entry_cap)
            available = _budget(effective_profile, project, catalog, references, material_profile, lists, tuning)
            return available, catalog, review_snapshot

        phase = "catalog"
        available, catalog, review_snapshot = await refresh_budget(selected)
        budget = manifest.material_budget if strict and manifest is not None else available
        report.profile.update(
            material_budget=budget, oversize_selection=tuning.oversize_selection, batch_cap=tuning.batch_cap
        )
        if budget > available or (
            strict and manifest is not None and manifest.max_output_tokens != effective_profile.max_output_tokens
        ):
            raise ExtractionError("input_budget_exceeded", "The profile cannot reproduce the frozen replay budget.")
        caps: dict[SessionKey, int] = {}
        for source in selected:
            record = snapshot.record(project_id, source.key)
            same_attempted_content = (
                record.attempted_updated is not None and record.attempted_updated.sha256 == source.revision.sha256
            )
            caps[source.key] = record.batch_session_cap if same_attempted_content else MAX_SESSIONS

        def plan_batches(
            sources: tuple[SourceSession, ...], budget: int, max_batches: int
        ) -> tuple[ExtractionBatch, ...]:
            return build_batches(
                project,
                sources,
                budget,
                caps=caps,
                profile=material_profile,
                max_batches=max_batches,
                oversize_selection=tuning.oversize_selection,
            )

        plan = plan_batches(selected, budget, tuning.batch_cap)
        if strict and manifest is not None:
            manifest.verify_batches(plan)
        if options.dry_run:
            report.status = "replay_dry_run" if manifest else "dry_run"
            report.batches = [_batch_summary(batch, report.run_id, "planned") for batch in plan]
            if options.freeze_to is not None:
                captured: dict[SessionKey, FrozenSource] = {}
                for batch in plan:
                    for source in batch.sources:
                        path = active_readers[source.key.host].paths.get(source.key)
                        if path is None:
                            raise ExtractionError("source_unavailable", "A planned source path is unavailable.")
                        captured[source.key] = FrozenSource(
                            source.key, path, source.revision.sha256, source.revision.updated_ns
                        )
                freeze_inputs = FreezeInputs(
                    options.freeze_to,
                    options.memory_dir,
                    options.baseline_dir,
                    protected_roots,
                    project_id,
                    budget,
                    effective_profile.max_output_tokens,
                    tuple(captured.values()),
                    references,
                    plan,
                    material_profile=material_profile,
                    tuning_version=tuning.version,
                )
            return report
        if not plan:
            report.status = "no_candidates"
            return report
        if manifest is None:
            lock = ProjectRunLock(options.memory_dir, project)
            phase = "run_lock"
            acquired = await owner.run(lock.try_acquire())
            _check_cancelled(owner)
            if not acquired:
                report.status = "already_running"
                return report
        runner = executor or ICodeExecutor(
            prepared_profile,
            store,
            FileChangeLog(options.memory_dir),
            report.run_id,
            references=references,
            material_profile=material_profile,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            tuning=tuning,
        )
        remaining = selected
        for number in range(tuning.batch_cap):
            current_summary = None
            if not remaining:
                break
            planned = plan_batches(remaining, budget, tuning.batch_cap - number)
            if not planned:
                break
            batch = planned[0]
            summary = _batch_summary(batch, report.run_id, "planned")
            current_summary = summary
            report.batches.append(summary)
            report.not_started_batches = len(planned)
            # A previous batch may have changed the catalog; its complete text is
            # budgeted afresh before admitting the next batch's attempt.
            phase = "catalog"
            available, catalog, review_snapshot = await refresh_budget(remaining)
            if strict and budget > available:
                raise ExtractionError("input_budget_exceeded", "The frozen batch no longer fits the profile window.")
            if not strict:
                budget = available
            planned = plan_batches(remaining, budget, tuning.batch_cap - number)
            if not planned:
                break
            batch = planned[0]
            summary.update(_batch_summary(batch, report.run_id, "planned"))
            report.not_started_batches = len(planned)
            invalid: set[SessionKey] = set()
            phase = "revalidation"
            for source in batch.sources:
                reason = await owner.run(active_readers[source.key.host].revalidate(source))
                if reason:
                    invalid.add(source.key)
                    report.diagnostics.append({"kind": reason, "host": source.key.host, "session": source.key.session})
            if invalid:
                if manifest:
                    raise ExtractionError("invalid_replay", "A frozen source changed during replay.")
                remaining = tuple(source for source in remaining if source.key not in invalid)
                # Rebuild without charging the unaffected members an attempt.
                if not remaining:
                    _check_cancelled(owner)
                    summary["status"] = "source_unavailable"
                    report.not_started_batches = 0
                    break
                batch = plan_batches(remaining, budget, 1)[0]
                # Revalidation of the rebuilt batch is necessary before admission.
                validated: list[SourceSession] = []
                for source in batch.sources:
                    reason = await owner.run(active_readers[source.key.host].revalidate(source))
                    if reason:
                        report.diagnostics.append(
                            {"kind": reason, "host": source.key.host, "session": source.key.session}
                        )
                        remaining = tuple(item for item in remaining if item.key != source.key)
                    else:
                        validated.append(source)
                if not validated:
                    _check_cancelled(owner)
                    summary["status"] = "source_unavailable"
                    continue
                batch = plan_batches(tuple(validated), budget, 1)[0]
            summary.update(_batch_summary(batch, report.run_id, "planned"))
            review = review_snapshot.lists(batch.sources, related_limit=tuning.related_entry_cap)
            estimated_input = MixedLanguageTokenizer().count_tokens(
                _model_input(batch, catalog, references, material_profile, review)
            )
            summary["estimated_input_tokens"] = estimated_input
            following_batches = len(
                plan_batches(
                    tuple(source for source in remaining if source.key not in {item.key for item in batch.sources}),
                    budget,
                    tuning.batch_cap - number - 1,
                )
            )
            report.not_started_batches = following_batches + 1
            ticket = None
            if manifest is None:
                phase = "attempt"
                _check_cancelled(owner)
                summary["commit_phase"], summary["commit_outcome"] = phase, None
                prepared = await owner.run(watermarks.prepare(project, batch.sources))
                summary["cancel_requested"] = owner.cancel_requested
                summary["commit_outcome"] = prepared.outcome
                if prepared.outcome != "committed" or prepared.ticket is None:
                    report.status, report.exit_code = (
                        ("commit_unknown", 3) if prepared.outcome == "commit_unknown" else ("attempt_write_failed", 1)
                    )
                    summary["status"] = report.status
                    break
                ticket = prepared.ticket
                summary["status"] = "started"
                report.not_started_batches = following_batches
                if owner.cancel_requested:
                    report.status, report.exit_code = "attempt_recorded_then_cancelled", 130
                    summary["status"] = report.status
                    break
            phase = "model"
            if owner.cancel_requested:
                raise OperationNotAdmitted()
            summary["status"] = "started"
            report.not_started_batches = following_batches
            result = await runner.run(batch)
            summary.update(asdict(result))
            report.conflict_count += result.conflict_count
            summary["initial_input_estimate_delta"] = (
                None if result.first_input_tokens is None else result.first_input_tokens - estimated_input
            )
            classification = classify_result(result)
            report.cancel_requested = report.cancel_requested or result.cancel_requested
            if classification == "failure":
                report.status, report.exit_code = "failed", 1
                summary["status"] = report.status
                break
            if classification == "cancelled":
                report.cancel_requested = True
                report.status, report.exit_code = (
                    ("saved_then_cancelled" if result.saved else "cancelled_before_commit"),
                    130,
                )
                summary["status"] = report.status
                break
            changed = False
            phase = "revalidation"
            for source in batch.sources:
                reason = await owner.run(active_readers[source.key.host].revalidate(source))
                if reason:
                    changed = True
                    report.diagnostics.append({"kind": reason, "host": source.key.host, "session": source.key.session})
            if changed:
                _check_cancelled(owner)
                classification = "failure"
            if classification not in {"success", "incomplete_limits"}:
                report.status, report.exit_code = "failed", 1
                summary["status"] = report.status
                break
            phase = "success" if classification == "success" else "incomplete_limits"
            if owner.cancel_requested:
                raise OperationNotAdmitted()
            if ticket is not None:
                summary["commit_phase"], summary["commit_outcome"] = phase, None
                committed = await owner.run(
                    watermarks.succeed(ticket)
                    if classification == "success"
                    else watermarks.limited(ticket, result.stop_reason or result.termination)
                )
                summary["cancel_requested"] = bool(summary["cancel_requested"]) or owner.cancel_requested
                summary["commit_outcome"] = committed.outcome
                summary["batch_split"] = [asdict(split) for split in committed.splits]
                if committed.outcome != "committed":
                    report.status, report.exit_code = (
                        ("commit_unknown", 3)
                        if committed.outcome == "commit_unknown"
                        else ("watermark_write_failed", 1)
                    )
                    summary["status"] = report.status
                    break
            prefix = "replay_" if manifest else ""
            if classification == "incomplete_limits":
                report.status, report.exit_code = prefix + "incomplete_limits", 2
                if owner.cancel_requested:
                    report.status = "incomplete_limits_committed_then_cancelled"
                summary["status"] = report.status
                break
            summary["status"] = prefix + "completed"
            if owner.cancel_requested:
                report.status, report.exit_code = "committed_then_cancelled", 0
                summary["status"] = report.status
                break
            remaining = tuple(
                source for source in remaining if source.key not in {member.key for member in batch.sources}
            )
            report.status = prefix + "completed"
            current_summary = None
        return report
    except asyncio.CancelledError:
        report.cancel_requested = True
        saved = bool(current_summary and current_summary.get("saved"))
        report.status, report.exit_code = ("saved_then_cancelled" if saved else "cancelled_before_" + phase), 130
        _settle_exception(report, current_summary, phase, owner)
        if current_summary is not None:
            current_summary["cancel_requested"] = True
        return report
    except (ExtractionError, MemoryOperationError) as error:
        report.status, report.exit_code = error.kind, 1
        report.diagnostics.append(
            {
                "kind": error.kind,
                "message": error.message if isinstance(error, ExtractionError) else "Memory operation failed.",
            }
        )
        _settle_exception(report, current_summary, phase, owner)
        return report
    except Exception as error:
        report.status, report.exit_code = "extraction_failed", 1
        if isinstance(error, TimeoutError) and current_summary is not None and phase in _WATERMARK_PHASES:
            # A lock timeout occurs before the watermark transaction writes.
            current_summary["commit_outcome"] = "not_committed"
        _settle_exception(report, current_summary, phase, owner)
        return report
    finally:

        async def cleanup() -> None:
            await owner.drain()
            if lock is not None:
                await lock.release()

        task = asyncio.create_task(cleanup())
        try:
            try:
                await finish_close(task)
            except asyncio.CancelledError:
                report.cancel_requested = True
                task.result()
        except Exception:
            report.diagnostics.append({"kind": "cleanup_warning"})
        report.cancel_requested = report.cancel_requested or owner.cancel_requested
        if current_summary is not None:
            current_summary["cancel_requested"] = bool(current_summary["cancel_requested"]) or owner.cancel_requested
        report.elapsed_seconds = round(time.monotonic() - started, 6)
        if freeze_inputs is not None and report.exit_code == 0:
            phase = "freeze"
            try:
                _check_cancelled(owner)
                if report.cancel_requested:
                    raise OperationNotAdmitted()

                async def verify_replay(path: Path) -> None:
                    replay_options = replace(
                        options,
                        claude_source=None,
                        codex_source=None,
                        sessions=frozenset(),
                        replay=path,
                        freeze_to=None,
                        baseline_dir=None,
                        dry_run=True,
                    )
                    checked = await run_extraction(replay_options, profile)
                    if checked.exit_code != 0:
                        raise ExtractionError(
                            "freeze_verification_failed", "Frozen inputs did not reproduce the planned batches."
                        )

                def report_snapshot(counts: dict[str, int]) -> dict[str, object]:
                    _check_cancelled(owner)
                    report.baseline_counts = dict(counts)
                    report.elapsed_seconds = round(time.monotonic() - started, 6)
                    return asdict(report)

                publication = await owner.run(freeze_regression_set(freeze_inputs, verify_replay, report_snapshot))
                if owner.cancel_requested:
                    removal = asyncio.create_task(asyncio.to_thread(publication.discard))
                    try:
                        await finish_close(removal)
                    except asyncio.CancelledError:
                        removal.result()
                    raise OperationNotAdmitted()
            except asyncio.CancelledError:
                report.cancel_requested = True
                report.status, report.exit_code = "cancelled_before_freeze", 130
            except ExtractionError as error:
                report.status, report.exit_code = error.kind, 1
                report.diagnostics.append({"kind": error.kind, "message": error.message})
            except Exception:
                report.status, report.exit_code = "freeze_write_failed", 1
                report.diagnostics.append({"kind": "freeze_write_failed"})


def _batch_summary(batch: ExtractionBatch, run_id: str, status: str) -> dict[str, object]:
    signals = _source_signals(batch.sources)
    return {
        "run_id": run_id,
        "status": status,
        "cancel_requested": False,
        "sessions": [asdict(source.key) for source in batch.sources],
        "annotated_sessions": signals.annotated_sessions,
        "annotations": signals.annotations,
        "displayed_messages": len(batch.source_index.view.messages),
        "displayed_fragments": sum(len(message.fragments) for message in batch.source_index.view.messages),
        "estimated_material_tokens": batch.estimated_tokens,
        "material_sha256": text_digest(batch.material),
        "omitted_messages": batch.omitted_messages,
        "omitted_characters": batch.omitted_characters,
        "material_counts": [asdict(count) for count in batch.material_counts],
        "omitted_error_results": batch.omitted_error_results,
        "omitted_signal_neighbors": batch.omitted_signal_neighbors,
        "omitted_assistant_messages": batch.omitted_assistant_messages,
        "conflict_count": 0,
        "withdrawn_count": 0,
        "merged_count": 0,
        "restored_count": 0,
        "consecutive_error_threshold": 10,
    }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Bootstrap once; model construction is delayed until a real eligible batch."""
    parser = argparse.ArgumentParser(description="Extract durable memories from ended source sessions")
    parser.add_argument("-C", type=Path, required=True)
    parser.add_argument("--claude-source", type=Path)
    parser.add_argument("--codex-source", type=Path)
    parser.add_argument("--model-profile")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--memory-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--freeze-to", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--tuning", help="Candidate tuning number or JSON file for frozen replay")
    parser.add_argument("--max-turns", type=_positive)
    parser.add_argument("--max-output-tokens", type=_positive)
    parser.add_argument("--batch-timeout-seconds", type=_seconds)
    args = parser.parse_args(argv)
    if args.freeze_to is not None and not args.dry_run:
        parser.error("--freeze-to requires --dry-run")
    if args.freeze_to is not None and args.replay is not None:
        parser.error("--freeze-to cannot be used with --replay")
    if args.baseline_dir is not None and args.freeze_to is None:
        parser.error("--baseline-dir requires --freeze-to")
    if args.replay is not None and args.memory_dir is None:
        parser.error("--replay requires an explicit isolated --memory-dir")
    if args.tuning is not None and args.replay is None:
        parser.error("--tuning requires --replay")
    try:
        runtime = bootstrap_runtime(project_root=args.C, apply_patches=False, setup_telemetry=False)
        settings: Settings = runtime.loaded.settings
        if settings.raw_http_capture:
            raise ExtractionError("raw_http_capture_enabled", "Disable raw model HTTP capture before extraction.")
        registry = ModelProfileRegistry()
        registry.load_all()
        selector = args.model_profile if args.model_profile is not None else settings.model_profile
        profile = resolve_selectable_profile(registry, selector) if selector else None
        if profile is None:
            raise ExtractionError("model_profile_unavailable", "Choose a complete profile by id or a unique name.")
        options = ExtractionOptions(
            args.C,
            args.memory_dir or get_platform().config_dir / "memory",
            args.claude_source,
            args.codex_source,
            frozenset(args.session),
            args.dry_run,
            args.replay,
            args.max_turns,
            args.max_output_tokens,
            args.batch_timeout_seconds,
            args.freeze_to,
            args.baseline_dir,
            args.tuning,
        )
        report = asyncio.run(run_extraction(options, profile))
        sys.stdout.write(json.dumps(asdict(report), ensure_ascii=True) + "\n")
        return report.exit_code
    except (ExtractionError, MemoryOperationError) as error:
        sys.stderr.write(f"{error.kind}: Extraction preflight failed.\n")
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception:
        sys.stderr.write("Extraction could not start.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
