import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
TRT_ROOT_DIR = os.environ.get("TRT_ROOT_DIR", "/usr/local/tensorrt")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Debug")
RECOMPILE_CPP = int(os.environ.get("RECOMPILE_CPP", "0"))
CUEQ_VERSION = os.environ.get("CUEQ_VERSION", "0.7.0")


def _check_and_install_packaging():
    """Check if packaging is available with version >= 24 and install if needed."""
    try:
        import packaging
        from packaging import version
        current_version = version.parse(packaging.__version__)
        required_version = version.parse("24.0")

        if current_version >= required_version:
            print(
                f"==> packaging {packaging.__version__} is already installed")
            return True
        else:
            print(
                f"==> packaging {packaging.__version__} is too old, upgrading to >= 24..."
            )
    except ImportError:
        print(f"==> packaging not found, installing...")

    try:
        subprocess.run(["pip", "install", "packaging>=24"], check=True)
        print(f"==> Successfully installed packaging >= 24")
        return True
    except subprocess.CalledProcessError as e:
        print(f"==> Failed to install packaging: {e}")
        return False


_check_and_install_packaging()


def _check_and_install_cuequivariance():
    """Check if cuequivariance_ops is available and install if needed."""
    try:
        import cuequivariance_ops  # noqa: F401
        print(f"==> cuequivariance_ops is already installed")
        return True
    except ImportError:
        print(f"==> cuequivariance_ops not found, installing...")
        try:
            subprocess.run([
                "pip", "install",
                f"cuequivariance-ops-torch-cu13=={CUEQ_VERSION}"
            ],
                           check=True)
            print(f"==> Successfully installed cuequivariance_ops")
            return True
        except subprocess.CalledProcessError as e:
            print(f"==> Failed to install cuequivariance_ops: {e}")
            return False


def _run_cmake(need_build: bool = True):
    try:
        import torch  # noqa: F401
    except ImportError:
        print(f"==> torch not found, you need to install torch first...")
        return
    _check_and_install_cuequivariance()
    if not need_build:
        return

    output_libs = [
        ROOT_DIR / "tensorrt_bionemo" / "libs" /
        "libkernels_tensorrt_bionemo.so",
        ROOT_DIR / "tensorrt_bionemo" / "libs" /
        "libnvinfer_plugin_tensorrt_bionemo.so",
        ROOT_DIR / "tensorrt_bionemo" / "libs" /
        "libtorch_crsc_tensorrt_bionemo.so",
    ]
    all_exist = True
    for output_lib in output_libs:
        all_exist = all_exist and output_lib.exists()
    if all_exist and not RECOMPILE_CPP:
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
    subprocess.run(
        ["cmake", "--build",
         str(build_dir), "--parallel",
         str(num_cores)],
        check=True)
    subprocess.run([
        "cmake",
        "--install",
        build_dir.as_posix(),
    ], check=True)


def prepare_metadata_for_build_editable(metadata_directory,
                                        config_settings=None):
    import setuptools.build_meta as stbm
    return stbm.prepare_metadata_for_build_editable(metadata_directory,
                                                    config_settings)


def build_editable(wheel_directory,
                   config_settings=None,
                   metadata_directory=None):
    _run_cmake(need_build=True)
    import setuptools.build_meta as stbm
    return stbm.build_editable(wheel_directory, config_settings,
                               metadata_directory)
