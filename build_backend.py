import os
import subprocess
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
TRT_ROOT_DIR = os.environ.get("TRT_ROOT_DIR", "/usr/local/tensorrt")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Release")
RECOMPILE_CPP = int(os.environ.get("RECOMPILE_CPP", "0"))
CUEQ_VERSION = os.environ.get("CUEQ_VERSION", "0.8.1")


def get_platform_tag():
    """Auto-detect platform tag for wheel filename."""
    # Method 1: Use packaging library (most accurate)
    try:
        from packaging.tags import sys_tags
        tag = next(sys_tags())
        return tag.platform
    except ImportError:
        pass
    
    # Method 2: Use sysconfig (standard library)
    platform = sysconfig.get_platform()
    return platform.replace("-", "_").replace(".", "_")


def get_python_tag():
    """Get Python tag like 'cp311' or 'py3'."""
    impl = sys.implementation.name[:2]  # 'cp' for CPython
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    return f"{impl}{version}"  # e.g., 'cp311'

def get_abi_tag():
    return "none"

def get_cuda_version():
    """Detect CUDA version like 'cu130' for CUDA 13.0."""
    # Try nvcc first
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            major, minor = match.groups()
            return f"cu{major}{minor}"
    except Exception:
        pass
    
    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            version = torch.version.cuda.replace(".", "")
            return f"cu{version}"
    except Exception:
        pass
    
    # Fallback
    return os.environ.get("CUDA_TAG", "cu126")


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
        "libnvinfer_plugin_tensorrt_bionemo.so"
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


def get_requires_for_build_sdist(config_settings=None):
    return []

def get_requires_for_build_wheel(config_settings=None):
    return []

def get_requires_for_build_editable(config_settings=None):
    return []

def prepare_metadata_for_build_editable(metadata_directory,
                                        config_settings=None):
    import setuptools.build_meta as stbm
    return stbm.prepare_metadata_for_build_editable(metadata_directory,
                                                    config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    import setuptools.build_meta as stbm
    return stbm.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_editable(wheel_directory,
                   config_settings=None,
                   metadata_directory=None):
    _run_cmake(need_build=True)
    import setuptools.build_meta as stbm
    return stbm.build_editable(wheel_directory, config_settings,
                               metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    import setuptools.build_meta as stbm
    return stbm.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _run_cmake(need_build=True)
    import setuptools.build_meta as stbm
    wheel_name = stbm.build_wheel(wheel_directory, config_settings, metadata_directory)
    wheel_path = Path(wheel_directory) / wheel_name
    print(f"==> Built wheel: {wheel_path}")
    # Parse original name
    # Format: {name}-{version}-{python}-{abi}-{platform}.whl
    match = re.match(r"^(.+?)-(\d+\.\d+\.\d+\.?[a-z0-9]*)-(.+?)-(.+?)-(.+?)\.whl$", wheel_name)
    if match:
        name = match.group(1)
        base_version = match.group(2)
        
        # Build custom tags
        cuda_tag = get_cuda_version()
        python_tag = get_python_tag()
        abi_tag = get_abi_tag()
        platform_tag = get_platform_tag()
        
        new_name = f"{name}-{base_version}+{cuda_tag}-{python_tag}-{abi_tag}-{platform_tag}.whl"
        
        new_path = Path(wheel_directory) / new_name
        wheel_path.rename(new_path)
        print(f"==> Renamed: {wheel_name} -> {new_name}")
        return new_name
    return wheel_name
