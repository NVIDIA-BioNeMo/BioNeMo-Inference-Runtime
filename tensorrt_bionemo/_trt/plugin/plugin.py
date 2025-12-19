# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import importlib.util
import platform
from pathlib import Path

CUEQUIV_PKG = importlib.util.find_spec("cuequivariance_ops")
CUEQUIV_PKG_LIB = None
if CUEQUIV_PKG is not None:
    CUEQUIV_PKG_LIB = Path(
        CUEQUIV_PKG.origin).parent.absolute() / "lib" / "libcue_ops.so"
    if CUEQUIV_PKG_LIB.exists():
        CUEQUIV_PKG_LIB = ctypes.CDLL(CUEQUIV_PKG_LIB.as_posix())

if CUEQUIV_PKG_LIB is None:
    raise ImportError('CuEQuivariance Ops library is unavailable')

TRT_BNM_PLUGIN_NAMESPACE = 'tensorrt_bionemo'


def plugin_lib_path() -> str:
    project_dir = Path(__file__).parent.parent.parent.absolute()
    dyn_lib = "libnvinfer_plugin_tensorrt_bionemo.so"
    return str(project_dir.joinpath("libs", dyn_lib))


def _load_plugin_lib():
    on_windows = platform.system() == "Windows"
    winmode = 0 if on_windows else None
    handle = ctypes.CDLL(
        plugin_lib_path(),
        mode=ctypes.
        RTLD_LOCAL,  # Load local here to avoid conflicts with TensorRT-LLM
        winmode=winmode)
    try:
        handle.initTrtBnmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtBnmPlugins.restype = ctypes.c_bool
    except AttributeError as err:
        raise ImportError('TensorRT-BioNemo Plugin is unavailable') from err

    try:
        assert handle.initTrtBnmPlugins(
            None, TRT_BNM_PLUGIN_NAMESPACE.encode('utf-8'))
    except OSError as e:
        windows_err = """
        The error above may be caused by an outdated Microsoft Visual C++ Redistributable Version.
        Please install the latest MSVC from the link below and re-launch.

        https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version
        """
        err_msg = dedent(windows_err if on_windows else "Unknown error")
        raise RuntimeError(err_msg) from e
    except Exception as e:
        raise e
