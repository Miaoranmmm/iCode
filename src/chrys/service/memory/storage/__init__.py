# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Memory storage: the entry schema, the ``MemoryStore`` interface and its file backend.

A store works on a given ``MemoryScope``; deciding which project a directory
belongs to is ``chrys.service.memory.project``'s job, not the store's.
"""
