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

import os
from pathlib import Path
from setuptools.dist import Distribution
from setuptools import find_packages, setup

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True

def parse_requirements(filename: os.PathLike):
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != "-", line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith("#") or line.startswith("-r"):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            else:
                deps.append(line)
    return deps, extra_URLs


def get_version():
    version_file = Path(
        __file__).resolve().parent / "tensorrt_bionemo" / "version.py"
    version = None
    with open(version_file) as f:
        for line in f:
            if not line.startswith("__version__"):
                continue
            version = line.split('"')[1]

    if version is None:
        raise RuntimeError(f"Could not set version from {version_file}")

    return version


required_deps, extra_URLs = parse_requirements("requirements.txt")
devel_deps, _ = parse_requirements("requirements-dev.txt")

setup(
    name="tensorrt_bionemo",
    version=get_version(),
    description=
    "TensorRT BioNemo: A TensorRT Toolbox for BioNemo structure prediction models",
    long_description=
    "TensorRT BioNemo: A TensorRT Toolbox for BioNemo structure prediction models",
    author="NVIDIA Corporation",
    # TODO: FIXME
    # url="https://github.com/NVIDIA/TensorRT-BioNemo"
    # download_url="https://github.com/NVIDIA/TensorRT-BioNemo/releases"
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"tensorrt_bionemo": ["libs/*.so"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    license="Apache License 2.0",
    keywords="nvidia tensorrt bioinformatics deeplearning inference",
    install_requires=required_deps,
    extras_require={
        "dev": devel_deps,
    },
    python_requires=">=3.10",
    zip_safe=True,
    entry_points={
        "console_scripts": ["trtbnm-build=tensorrt_bionemo.commands.build:main"]
    },
    distclass=BinaryDistribution,
)
