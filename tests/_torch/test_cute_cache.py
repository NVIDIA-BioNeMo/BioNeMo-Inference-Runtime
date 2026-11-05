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

import fcntl
import importlib.util
import os
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from bionemo_ir.dsl_kernels import cute_cache

DiskCache = cute_cache.DiskCache
# Fingerprinting and loading a cached object import the CuTeDSL compiler.
requires_compiler = pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTeDSL is not installed")


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


@requires_compiler
def test_crashed_export_publishes_no_partial_object(cache_dir):
    """A writer killed mid-export must not leave a canonical ``.o`` behind."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "crash")

    # export_to_c writes 3 of 16 bytes, then raises. save_to_cache swallows the
    # error (logged) — the point is what it leaves on disk.
    cache.save_to_cache(key, _FakeArtifact(b"OBJECT_CONTENTS!", fail_after=3))

    assert _o_files(cache_dir) == [], "a torn export left a canonical .o that peers would load and execute"
    assert _tmp_files(cache_dir) == [], "staging temp file was not cleaned up"


@requires_compiler
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
        cute_cache._cute().runtime, "load_module", lambda path, enable_tvm_ffi: {cute_cache.EXPORT_FUNC_NAME: sentinel}
    )
    assert cache.load_from_cache(key) is sentinel

    cache_paths = [cache_dir, published[0].parent]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in cache_paths)


def test_secure_cache_dir_tightens_existing_directory(tmp_path):
    """A world-writable cache directory is narrowed to owner-only access."""
    path = tmp_path / "cache"
    path.mkdir()
    os.chmod(path, 0o777)

    assert cute_cache._secure_cache_dir(path) == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_secure_cache_dir_rejects_symlinked_directory(tmp_path):
    """A symlink standing in for the cache directory is refused, not chmod'ed."""
    target = tmp_path / "target"
    target.mkdir()
    os.chmod(target, 0o777)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(PermissionError):
        cute_cache._secure_cache_dir(link)

    assert stat.S_IMODE(target.stat().st_mode) == 0o777


@requires_compiler
def test_load_rejects_unowned_object(cache_dir, monkeypatch):
    """An object this user does not own is never handed to ``load_module``."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "unowned")
    cache.save_to_cache(key, _FakeArtifact(b"OBJECT"))
    monkeypatch.setattr(cute_cache, "_is_owned_object", lambda path: False)
    monkeypatch.setattr(cute_cache._cute().runtime, "load_module", pytest.fail)

    assert cache.load_from_cache(key) is None


@requires_compiler
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

    monkeypatch.setattr(cute_cache._cute().runtime, "load_module", _boom)

    assert cache.load_from_cache(key) is None
    assert not o_path.exists(), "corrupt .o was not purged on load failure"


@requires_compiler
def test_source_cache_separates_host_isa(cache_dir, monkeypatch):
    """A host wrapper compiled for one CPU ISA must miss on another ISA."""
    monkeypatch.setattr(cute_cache, "_device_sm_count", lambda: 132)
    host_isa = ["x86_64:avx2"]
    monkeypatch.setattr(cute_cache, "_host_cpu_isa_signature", lambda: host_isa[0])

    load_calls = []
    sentinel = object()
    monkeypatch.setattr(
        cute_cache._cute().runtime,
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


@requires_compiler
def test_miss_claims_the_key_until_save(cache_dir):
    """A miss locks the key for its compiler; the save releases it."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "claim")
    sha = cute_cache._key_to_hash(key)

    assert cache.load_from_cache(key) is None
    assert sha in cute_cache._compile_claims
    cache.save_to_cache(key, _FakeArtifact(b"OBJECT"))
    assert sha not in cute_cache._compile_claims


@requires_compiler
def test_miss_waits_for_a_peer_compile(cache_dir, monkeypatch):
    """A miss while a peer compiles the key loads the peer's object instead."""
    cache = cute_cache.CuteKernelCache()
    key = ("adaln", "float16", 128, "peer")
    sha = cute_cache._key_to_hash(key)
    sentinel = object()
    monkeypatch.setattr(
        cute_cache._cute().runtime, "load_module", lambda path, enable_tvm_ffi: {cute_cache.EXPORT_FUNC_NAME: sentinel}
    )
    cache_path = cute_cache._secure_cache_dir(cache_dir / cute_cache._compute_source_fingerprint())
    peer = os.open(cache_path / f"{sha}.compile", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(peer, fcntl.LOCK_EX)

    def peer_compiles():
        time.sleep(0.5)
        cache.save_to_cache(key, _FakeArtifact(b"PEER_OBJECT"))
        fcntl.flock(peer, fcntl.LOCK_UN)
        os.close(peer)

    thread = threading.Thread(target=peer_compiles)
    thread.start()
    assert cache.load_from_cache(key) is sentinel
    thread.join()
    assert sha not in cute_cache._compile_claims


def _dsl_op_locations() -> tuple[list[str], object]:
    """Build locations the way a kernel does: the DSL's op wrapper tags each call with its caller's site."""
    from cutlass._mlir import ir
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    @dsl_op.dsl_user_op
    def record(*, loc: ir.Location | None = None) -> str:
        """Return the location attached by the DSL wrapper."""
        return str(loc)

    with ir.Context(), ir.Location.unknown():
        # Two sites, the first one hit three times.
        locations = [record() for _ in range(3)]
        locations.append(record())
    return locations, dsl_op.inspect


@requires_compiler
def test_memoized_op_locations_match_the_dsl() -> None:
    """The memo hands the DSL the locations it computes itself, one lookup per site."""
    import inspect

    plain, plain_inspect = _dsl_op_locations()
    with cute_cache.memoized_op_locations():
        memoized, memo = _dsl_op_locations()
    assert plain_inspect is inspect
    assert isinstance(memo, cute_cache._SiteMemoizedInspect)
    assert memoized == plain
    assert len(set(plain)) == 2 and "record()" in plain[0]
    assert len(memo._sites) == 2


def _caller_frame_info(memo: cute_cache._SiteMemoizedInspect) -> tuple[object, object]:
    """Look up the caller's frame, suspended at this call, through the memo and through ``inspect``."""
    import inspect
    import sys

    frame = sys._getframe(1)
    return memo.getframeinfo(frame), inspect.getframeinfo(frame)


def test_memoized_frame_info_equals_inspect() -> None:
    """Each memoized answer is ``inspect.getframeinfo``'s, positions and source context included."""
    import inspect

    memo = cute_cache._SiteMemoizedInspect()
    answers = [_caller_frame_info(memo) for _ in range(3)]
    answers.append(_caller_frame_info(memo))
    for got, want in answers:
        assert got == want and got.positions == want.positions
    assert answers[0][0] is answers[2][0] and answers[0][0] != answers[3][0]
    assert len(memo._sites) == 2
    assert memo.currentframe is inspect.currentframe


@requires_compiler
def test_memoized_op_locations_restore_on_exit() -> None:
    """Nested and failing compiles restore the DSL's ``inspect`` only when the outermost one ends."""
    import inspect

    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    with pytest.raises(RuntimeError, match="compile failed"):
        with cute_cache.memoized_op_locations():
            outer = dsl_op.inspect
            with cute_cache.memoized_op_locations():
                assert dsl_op.inspect is outer
            assert dsl_op.inspect is outer
            raise RuntimeError("compile failed")
    assert isinstance(outer, cute_cache._SiteMemoizedInspect)
    assert dsl_op.inspect is inspect


@requires_compiler
def test_memoized_op_locations_keep_each_file(tmp_path: Path) -> None:
    """Equal code objects from two files keep their own file's location.

    Args:
        tmp_path: Directory for the two generated source files.
    """
    from cutlass._mlir import ir
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    @dsl_op.dsl_user_op
    def record(*, loc: ir.Location | None = None) -> str:
        """Return the location attached to each generated file's call site."""
        return str(loc)

    source = "def site():\n    return record()\n"
    sites = []
    for name in ("first.py", "second.py"):
        path = tmp_path / name
        path.write_text(source)
        namespace = {"record": record}
        exec(compile(source, str(path), "exec"), namespace)
        sites.append(namespace["site"])
    assert sites[0].__code__ == sites[1].__code__

    with ir.Context(), ir.Location.unknown():
        plain = [site() for site in sites]
        with cute_cache.memoized_op_locations():
            memoized = [site() for site in sites]
    assert "first.py" in plain[0] and "second.py" in plain[1]
    assert memoized == plain


def _exit_code_in_fork(check: Callable[[], bool]) -> int:
    """Run ``check`` in a forked child; 0 if it returns true, -SIGALRM if it hangs for 5 s."""
    import signal

    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            signal.alarm(5)
            code = 0 if check() else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def _scope_restores() -> bool:
    """Check that a memo scope restores the upstream inspect module."""
    import inspect

    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    with cute_cache.memoized_op_locations():
        entered = isinstance(dsl_op.inspect, cute_cache._SiteMemoizedInspect)
    return entered and dsl_op.inspect is inspect


@requires_compiler
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fork_drops_other_threads_scopes() -> None:
    """A child forked while another thread compiles restores ``inspect`` after its own compile."""
    import inspect

    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    entered, release = threading.Event(), threading.Event()

    def compiling() -> None:
        """Hold a memo scope open until the child fork check completes."""
        with cute_cache.memoized_op_locations():
            entered.set()
            release.wait()

    thread = threading.Thread(target=compiling)
    thread.start()
    entered.wait()
    try:
        code = _exit_code_in_fork(_scope_restores)
    finally:
        release.set()
        thread.join()
    assert code == 0
    assert dsl_op.inspect is inspect


@requires_compiler
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fork_while_another_thread_holds_the_lock() -> None:
    """A child forked while another thread holds the memo lock can still compile."""
    locked, release = threading.Event(), threading.Event()

    def holding() -> None:
        """Hold the memo lock until the child fork check completes."""
        with cute_cache._memo_lock:
            locked.set()
            release.wait()

    thread = threading.Thread(target=holding)
    thread.start()
    locked.wait()
    try:
        code = _exit_code_in_fork(_scope_restores)
    finally:
        release.set()
        thread.join()
    assert code == 0


@requires_compiler
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fork_inside_a_scope_keeps_it() -> None:
    """A child forked inside two memo scopes keeps the memo until it leaves both, then restores."""
    import inspect
    import signal

    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    pid = -1
    checks: list[bool] = []
    try:
        with cute_cache.memoized_op_locations():
            with cute_cache.memoized_op_locations():
                memo = dsl_op.inspect
                pid = os.fork()
                if pid == 0:
                    signal.alarm(5)
                    checks.append(dsl_op.inspect is memo)
                    with cute_cache.memoized_op_locations():
                        checks.append(dsl_op.inspect is memo)
                checks.append(dsl_op.inspect is memo)
            checks.append(dsl_op.inspect is memo)
        checks.extend([dsl_op.inspect is inspect, not cute_cache._memo_scopes])
    finally:
        if pid == 0:
            os._exit(0 if len(checks) == 6 and all(checks) else 2)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert dsl_op.inspect is inspect


@requires_compiler
def test_memoized_op_locations_leave_a_foreign_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``inspect`` someone else put on the DSL's op module stays in place.

    Args:
        monkeypatch: Fixture that restores the foreign inspect reference.
    """
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    foreign = object()
    monkeypatch.setattr(dsl_op, "inspect", foreign)
    with cute_cache.memoized_op_locations():
        assert dsl_op.inspect is foreign
    assert dsl_op.inspect is foreign


@requires_compiler
@pytest.mark.parametrize("version", ["4.5.2", "4.5.3"])
def test_memoized_op_locations_validated_versions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    """Validated compiler versions enable memoization without warnings.

    Args:
        monkeypatch: Fixture that restores the compiler version and warning state.
        caplog: Captured log records.
        version: Validated compiler version to exercise.
    """
    import inspect

    import cutlass
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    monkeypatch.setattr(cutlass, "__version__", version)
    monkeypatch.setattr(cute_cache, "_memo_warning_logged", False)
    with cute_cache.memoized_op_locations():
        assert isinstance(dsl_op.inspect, cute_cache._SiteMemoizedInspect)
    assert dsl_op.inspect is inspect
    assert not caplog.records


@requires_compiler
@pytest.mark.parametrize("version", ["4.6.0", "4.8.1", "999.0", "4.5.3.dev0", "unknown"])
def test_unvalidated_version_compiles_unchanged_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    """An unsupported compiler bypasses the private API and preserves compile results and errors.

    Args:
        monkeypatch: Fixture that restores the compiler stub, version and warning state.
        caplog: Captured log records.
        version: Unsupported compiler version to exercise.
    """
    import inspect

    import cutlass
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    monkeypatch.setattr(cutlass, "__version__", version)
    monkeypatch.setattr(cute_cache, "_memo_warning_logged", False)
    calls = []
    result = object()

    def compile_kernel(*args: object, **kwargs: object) -> object:
        """Record forwarded arguments and simulate an upstream compiler failure."""
        assert dsl_op.inspect is inspect
        calls.append((args, kwargs))
        if len(calls) == 2:
            raise ValueError("upstream compiler error at kernel.py:42")
        return result

    monkeypatch.setattr(cute_cache._cute(), "compile", compile_kernel)
    cache = cute_cache.CuteKernelCache()
    assert cache.compile("kernel", "tensor", options="--test", extra=3) is result
    with pytest.raises(ValueError, match="upstream compiler error at kernel.py:42"):
        cache.compile("kernel", "tensor", options="--test", extra=3)
    assert calls == [(("kernel", "tensor"), {"options": "--test", "extra": 3})] * 2
    assert not cute_cache._memo_scopes
    assert len(caplog.records) == 1
    warning = caplog.records[0]
    assert warning.levelname == "WARNING"
    assert f"CuTeDSL {version}" in warning.message
    assert "upstream compiler unchanged" in warning.message
    assert "_OP_LOCATION_MEMO_VERSIONS" in warning.message


@requires_compiler
@pytest.mark.parametrize("change", ["missing_module", "missing_inspect", "foreign_inspect"])
def test_changed_location_helper_falls_back_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, change: str
) -> None:
    """A changed private helper is left untouched, even on a validated version.

    Args:
        monkeypatch: Fixture that restores the compiler version, helper and warning state.
        caplog: Captured log records.
        change: Private helper incompatibility to simulate.
    """
    import sys

    import cutlass
    from cutlass.base_dsl import _mlir_helpers
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    monkeypatch.setattr(cutlass, "__version__", "4.5.3")
    monkeypatch.setattr(cute_cache, "_memo_warning_logged", False)
    if change == "missing_module":
        monkeypatch.delattr(_mlir_helpers, "op")
        monkeypatch.setitem(sys.modules, dsl_op.__name__, None)
    elif change == "missing_inspect":
        monkeypatch.delattr(dsl_op, "inspect")
    else:
        monkeypatch.setattr(dsl_op, "inspect", object())
    original = getattr(dsl_op, "inspect", None)
    for _ in range(2):
        with cute_cache.memoized_op_locations():
            assert getattr(dsl_op, "inspect", None) is original
    assert not cute_cache._memo_scopes
    assert len(caplog.records) == 1
    expected = "private location helper is unavailable" if change == "missing_module" else "inspect reference changed"
    assert expected in caplog.text


@requires_compiler
@pytest.mark.parametrize("version", ["4.5.2", "4.5.3"])
@pytest.mark.parametrize("fails", [False, True])
def test_compile_memoizes_locations_and_restores_inspect(
    monkeypatch: pytest.MonkeyPatch, version: str, fails: bool
) -> None:
    """The compile entry point scopes memoization around the upstream call.

    Args:
        monkeypatch: Fixture that restores the compiler stub and version.
        version: Validated compiler version to exercise.
        fails: Whether the compiler stub raises instead of returning an artifact.
    """
    import inspect

    import cutlass
    from cutlass.base_dsl._mlir_helpers import op as dsl_op

    monkeypatch.setattr(cutlass, "__version__", version)
    result = object()
    calls = []

    def compile_kernel(*args: object, **kwargs: object) -> object:
        """Check the active memo and preserve upstream results or errors."""
        assert isinstance(dsl_op.inspect, cute_cache._SiteMemoizedInspect)
        calls.append((args, kwargs))
        if fails:
            raise ValueError("upstream compile failed")
        return result

    compiler = ModuleType("stub_cute")
    monkeypatch.setattr(compiler, "compile", compile_kernel, raising=False)
    monkeypatch.setattr(cute_cache, "_cute", lambda: compiler)
    cache = cute_cache.CuteKernelCache()
    if fails:
        with pytest.raises(ValueError, match="upstream compile failed"):
            cache.compile("kernel", "tensor", options="--test", extra=3)
    else:
        assert cache.compile("kernel", "tensor", options="--test", extra=3) is result
    assert calls == [(("kernel", "tensor"), {"options": "--test", "extra": 3})]
    assert dsl_op.inspect is inspect
    assert not cute_cache._memo_scopes
