# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Crash-safety tests for the CuTe DSL ``.o`` disk cache.

These exercise the file-I/O contract only (fake artifacts, no GPU): a compiled
kernel must never be published to its canonical ``{sha}.o`` path in a partial
state. If an interrupted export leaves a truncated ``.o`` at that path, any
process that later loads and executes it dies with SIGILL ("Fatal Python
error: Illegal instruction") — an unrecoverable crash, not an exception the
caller can handle.
"""

import pytest

# Importing the cache pulls in cutlass; skip cleanly where it is unavailable
# (e.g. a CPU-only dev box) rather than erroring at collection.
cute_cache = pytest.importorskip("bionemo_ir.dsl_kernels.cute_cache")
DiskCache = cute_cache.DiskCache


class _FakeArtifact:
    """Stand-in for a ``cute.compile`` result.

    ``export_to_c`` writes ``payload`` to the given path. When ``fail_after`` is
    set it writes only that many bytes and then raises — simulating a process
    killed (OOM / SIGKILL) partway through the export.
    """

    def __init__(self, payload: bytes, fail_after: int | None = None):
        self.payload = payload
        self.fail_after = fail_after
        self.export_target: str | None = None

    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        self.export_target = object_file_path
        if self.fail_after is not None:
            with open(object_file_path, "wb") as f:
                f.write(self.payload[: self.fail_after])
            raise RuntimeError("simulated OOM mid-export")
        with open(object_file_path, "wb") as f:
            f.write(self.payload)


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Point the disk cache at an isolated temp dir for the test."""
    monkeypatch.setattr(DiskCache, "_CACHE_DIR", str(tmp_path))
    return tmp_path


def _o_files(root):
    """Canonical published artifacts (excludes ``*.o.tmp`` staging files)."""
    return [p for p in root.rglob("*.o") if not p.name.endswith(".o.tmp")]


def _tmp_files(root):
    return list(root.rglob("*.o.tmp"))


def test_crashed_export_publishes_no_partial_object(cache_dir):
    """A writer killed mid-export must not leave a canonical ``.o`` behind."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "crash")

    # export_to_c writes 3 of 16 bytes, then raises. save_to_cache swallows the
    # error (logged) — the point is what it leaves on disk.
    cache.save_to_cache(key, _FakeArtifact(b"OBJECT_CONTENTS!", fail_after=3))

    assert _o_files(cache_dir) == [], "a torn export left a canonical .o that peers would load and execute"
    assert _tmp_files(cache_dir) == [], "staging temp file was not cleaned up"


def test_successful_export_publishes_complete_object(cache_dir, monkeypatch):
    """A clean export publishes exactly the full artifact, and load returns it."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "ok")
    payload = b"COMPLETE_OBJECT_FILE"

    cache.save_to_cache(key, _FakeArtifact(payload))

    published = _o_files(cache_dir)
    assert len(published) == 1
    assert published[0].read_bytes() == payload
    assert _tmp_files(cache_dir) == []

    sentinel = object()
    monkeypatch.setattr(
        cute_cache.cute.runtime, "load_module", lambda path, enable_tvm_ffi: {cute_cache.EXPORT_FUNC_NAME: sentinel}
    )
    assert cache.load_from_cache(key) is sentinel


def test_load_purges_corrupt_object(cache_dir, monkeypatch):
    """A ``.o`` that fails to load is deleted so peers recompile, not re-crash."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "corrupt")

    # Publish a valid artifact, then clobber the file with garbage to mimic a
    # legacy corrupt object predating the atomic-write path.
    cache.save_to_cache(key, _FakeArtifact(b"VALID"))
    (o_path,) = _o_files(cache_dir)
    o_path.write_bytes(b"\x00\xff not a real object file")

    def _boom(path, enable_tvm_ffi):
        raise RuntimeError("corrupt object file")

    monkeypatch.setattr(cute_cache.cute.runtime, "load_module", _boom)

    assert cache.load_from_cache(key) is None
    assert not o_path.exists(), "corrupt .o was not purged on load failure"


def test_source_cache_separates_host_isa(cache_dir, monkeypatch):
    """A host wrapper compiled for one CPU ISA must miss on another ISA."""
    monkeypatch.setattr(cute_cache, "_device_sm_count", lambda: 132)
    host_isa = ["x86_64:avx2"]
    monkeypatch.setattr(cute_cache, "_host_cpu_isa_signature", lambda: host_isa[0])

    load_calls = []
    sentinel = object()
    monkeypatch.setattr(
        cute_cache.cute.runtime,
        "load_module",
        lambda path, enable_tvm_ffi: load_calls.append(path) or {cute_cache.EXPORT_FUNC_NAME: sentinel},
    )

    cache = cute_cache.CuteKernelCache()
    key = ("pairwise_attention", "float16", 128)
    try:
        cute_cache._compute_source_fingerprint.cache_clear()
        avx2_fingerprint = cute_cache._compute_source_fingerprint()
        cache.save_to_cache(key, _FakeArtifact(b"AVX2_OBJECT"))

        host_isa[0] = "x86_64:avx2,avx512f,avx512vl"
        cute_cache._compute_source_fingerprint.cache_clear()
        avx512_fingerprint = cute_cache._compute_source_fingerprint()
        assert cache.load_from_cache(key) is None
        assert load_calls == []

        host_isa[0] = "x86_64:avx2"
        cute_cache._compute_source_fingerprint.cache_clear()
        assert cache.load_from_cache(key) is sentinel
        assert len(load_calls) == 1
    finally:
        cute_cache._compute_source_fingerprint.cache_clear()

    assert avx2_fingerprint != avx512_fingerprint


def test_cpuinfo_profiles_are_canonical_and_affinity_scoped():
    cpuinfo = """\
processor : 0
vendor_id : GenuineIntel
flags : fma avx2 sse4_2
cpu family : 6

processor : 1
cpu family : 6
flags : avx512vl sse4_2 avx2 fma
vendor_id : GenuineIntel
"""
    reordered = """\
flags : sse4_2 avx2 fma
processor : 0
cpu family : 6
vendor_id : GenuineIntel

vendor_id : GenuineIntel
flags : fma avx2 sse4_2 avx512vl
cpu family : 6
processor : 1
"""

    assert cute_cache._cpuinfo_isa_profiles(cpuinfo, {0, 1}) == cute_cache._cpuinfo_isa_profiles(reordered, {0, 1})
    cpu0 = cute_cache._cpuinfo_isa_profiles(cpuinfo, {0})
    cpu1 = cute_cache._cpuinfo_isa_profiles(cpuinfo, {1})
    assert len(cpu0) == len(cpu1) == 1
    assert cpu0 != cpu1


def test_host_isa_fallback_is_not_shared(monkeypatch):
    monkeypatch.setattr(cute_cache.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cute_cache.os, "sched_getaffinity", lambda pid: {0, 1})
    process_id = [123]
    monkeypatch.setattr(cute_cache.os, "getpid", lambda: process_id[0])
    monkeypatch.setattr(cute_cache, "_HOST_CPU_FALLBACK_NONCE", "process-random-nonce")

    monkeypatch.setattr(cute_cache, "_read_cpuinfo", lambda: None)
    unavailable = cute_cache._host_cpu_isa_signature()
    assert unavailable == "machine=x86_64|profiles=unavailable|nonshareable=123-process-random-nonce"
    process_id[0] = 456
    assert cute_cache._host_cpu_isa_signature() != unavailable

    heterogeneous = """\
processor : 0
flags : avx2

processor : 1
flags : avx2 avx512f
"""
    monkeypatch.setattr(cute_cache, "_read_cpuinfo", lambda: heterogeneous)
    mixed = cute_cache._host_cpu_isa_signature()
    assert "nonshareable=456-process-random-nonce" in mixed

    monkeypatch.setattr(cute_cache.os, "sched_getaffinity", lambda pid: {0})
    uniform = cute_cache._host_cpu_isa_signature()
    assert "nonshareable=" not in uniform
