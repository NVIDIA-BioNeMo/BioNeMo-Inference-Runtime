import os
import subprocess
from pathlib import Path

import setuptools.build_meta as stbm

ROOT_DIR = Path(__file__).parent.resolve()
TRT_ROOT_DIR = os.environ.get("TRT_ROOT_DIR", "/usr/local/tensorrt")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Debug")
RECOMPLIE_CPP = int(os.environ.get("RECOMPLIE_CPP", "0"))


def _run_cmake(need_build: bool = True):
    if not need_build:
        return

    output_path = ROOT_DIR / "tensorrt_bionemo" / "libs" / "libnvinfer_plugin_tensorrt_bionemo.so"
    if output_path.exists() and not RECOMPLIE_CPP:
        return
    print(f"==> Building C++ extension for TensorRT Bionemo...")
    cpp_dir = ROOT_DIR / "cpp"
    build_dir = cpp_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
        f"-DTRT_ROOT_DIR={TRT_ROOT_DIR}",
        f"-DFAST_BUILD=ON",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        f"-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
    ]
    subprocess.run([
        "cmake",
        "-S",
        cpp_dir.as_posix(),
        "-B",
        build_dir.as_posix(),
        *cmake_args,
    ],
                   check=True)

    num_cores = os.cpu_count()
    subprocess.run([
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        str(num_cores),
        "--target",
        "nvinfer_plugin_tensorrt_bionemo",
    ],
                   check=True)
    subprocess.run([
        "cmake",
        "--install",
        build_dir.as_posix(),
    ], check=True)


def prepare_metadata_for_build_editable(metadata_directory,
                                        config_settings=None):
    return stbm.prepare_metadata_for_build_editable(metadata_directory,
                                                    config_settings)


def build_editable(wheel_directory,
                   config_settings=None,
                   metadata_directory=None):
    _run_cmake(need_build=True)
    return stbm.build_editable(wheel_directory, config_settings,
                               metadata_directory)
