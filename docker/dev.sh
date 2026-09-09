#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build this worktree's dev image and open a shell in it.
#
# The image comes from `make -C docker dev`, so there is still exactly one
# image definition in the repo; this script drives that build and owns
# everything around it -- the tag, the container, the mounts.
#
# The container is long-lived and named after the worktree directory, so every
# worktree gets its own and a second terminal joins the one already running
# rather than starting a rival. Everything worth keeping lives on the host: the
# checkout is bind-mounted, and so are the caches, which are shared by every
# worktree and survive the container.
#
# Run it from anywhere -- it works on the worktree it lives in, not on $PWD.
# Guide: docs/ref/docker-images.md.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared by every worktree: weights are tens of GB and a warm ccache/kernel
# cache is the difference between a minute and an hour.
HOST_CACHE="${BIOIR_HOST_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/bionemo-ir-dev}"

# The library's own weight cache, resolved exactly as bionemo_ir.CACHE_DIR and
# scripts/fetch_weights.sh resolve it, so a stage run on the host and one run in
# here address the same directory. Deliberately not under HOST_CACHE: it is the
# package's cache, not this script's, and host-side tooling reads it too.
WEIGHT_CACHE="${BIOIR_CACHE:-${HOME}/.cache/bionemo_ir}"

# Where the checkout is mounted, and where every shell starts.
WORKDIR=/bioir

# Host environment worth carrying in: gated-checkpoint credentials and the NGC
# coordinates `fetch_weights.sh --source ngc` needs. Unset ones are skipped.
PASS_ENV=(HF_TOKEN HUGGING_FACE_HUB_TOKEN BIOIR_NGC_ORG BIOIR_NGC_TEAM NGC_API_KEY)

usage() {
  cat <<'EOF'
Usage: docker/dev.sh [options] [command ...]

Builds this worktree's dev image and opens a shell in it, creating the
container on first use. Without a command, runs bash.

Options:
  --rm          Remove this worktree's container and exit.
  --reset       Remove it first, then create it fresh.
  --no-build    Skip `make -C docker dev` and use the image as it is.
  -h, --help    Show this help.

Environment:
  BIOIR_HOST_CACHE  host cache root (default: ~/.cache/bionemo-ir-dev)
  BIOIR_CACHE       weight cache, shared with the host (default:
                    ~/.cache/bionemo_ir). Baked into the container at create
                    time, so changing it needs --reset.
  BIOIR_DEV_IMAGE   image:tag to run instead of the one this script builds
                    (implies --no-build)
  BIOIR_CONTAINER   container name (default: derived from the worktree dir)
  BIOIR_GPUS        --gpus value, or "none" (default: all when the nvidia
                    container runtime is installed, none otherwise)
EOF
}

die() {
  printf 'dev.sh: %s\n' "$1" >&2
  exit 1
}

REMOVE=0
RESET=0
BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rm)
      REMOVE=1
      shift
      ;;
    --reset)
      RESET=1
      shift
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      usage >&2
      die "unknown option: $1"
      ;;
    *)
      break
      ;;
  esac
done

command -v docker >/dev/null || die "docker not found on PATH"
REPO_ROOT="$(git -C "${DOCKER_DIR}" rev-parse --show-toplevel)" || die "not a git checkout: ${DOCKER_DIR}"

# The worktree directory is the one thing that distinguishes concurrent
# checkouts of this repo, and docker names may hold [a-zA-Z0-9_.-] only.
SLUG="$(printf '%s' "$(basename "${REPO_ROOT}")" | tr -c 'a-zA-Z0-9_.-' '-')"
CONTAINER="${BIOIR_CONTAINER:-bioir-${SLUG}}"
# One image per worktree, so a rebuild on one branch does not pull the rug from
# under a container on another. The layers are shared, so the copy is a tag.
REGISTRY_IMAGE=bionemo-ir
IMAGE="${BIOIR_DEV_IMAGE:-${REGISTRY_IMAGE}:dev-${SLUG}}"
if [[ -n ${BIOIR_DEV_IMAGE:-} ]]; then
  BUILD=0
fi

container_exists() {
  docker container inspect "${CONTAINER}" >/dev/null 2>&1
}

remove_container() {
  container_exists || return 0
  printf 'Removing container %s\n' "${CONTAINER}"
  docker rm --force "${CONTAINER}" >/dev/null
}

if [[ ${REMOVE} -eq 1 ]]; then
  remove_container
  exit 0
fi

if [[ ${BUILD} -eq 1 ]]; then
  command -v make >/dev/null || die "make not found on PATH"
  make -C "${DOCKER_DIR}" dev REGISTRY_IMAGE="${REGISTRY_IMAGE}" TAG="dev-${SLUG}"
fi
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image ${IMAGE} not found -- drop --no-build"

if [[ ${RESET} -eq 1 ]]; then
  remove_container
fi

# A rebuilt image under the same tag leaves the running container on the old
# one, silently. The container holds nothing but a `sleep`, so replace it.
if container_exists; then
  running_image="$(docker container inspect --format '{{.Image}}' "${CONTAINER}")"
  wanted_image="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
  if [[ ${running_image} != "${wanted_image}" ]]; then
    printf 'Image changed since %s was created; recreating it.\n' "${CONTAINER}"
    remove_container
  fi
fi

if ! container_exists; then
  # The image's own $HOME: the user is created at build time from the host
  # UID/GID, so do not guess the path. --entrypoint skips the NGC banner.
  container_home="$(docker run --rm --entrypoint sh "${IMAGE}" -c 'printf %s "${HOME}"')"
  [[ -n ${container_home} ]] || die "could not read HOME from ${IMAGE}"

  # Create the host directories first. A bind mount whose source is missing is
  # created by the daemon as root, which the container user cannot then write.
  mkdir -p "${HOST_CACHE}/cache" "${HOST_CACHE}/home-cache" "${WEIGHT_CACHE}"

  # One array, appended to throughout: an empty array expanded under `set -u`
  # is an error on bash 3.2, which is what macOS ships.
  run_args=(
    --detach
    --name "${CONTAINER}"
    --hostname "${CONTAINER}"
    --workdir "${WORKDIR}"
    # Kernel compilation and the xdist test workers both need far more than the
    # 64 MB default shm. These match .devcontainer/docker-compose.yml and what
    # the CI test job passes, so a local run matches CI.
    --ipc host
    --shm-size 10g
    --ulimit memlock=-1
    --ulimit stack=67108864
    --cap-add SYS_PTRACE
    --volume "${REPO_ROOT}:${WORKDIR}"
    # ccache, the pip cache and the compiled-kernel (CuTeDSL/Triton) cache,
    # placed here by the Dockerfile's CCACHE_DIR / PIP_CACHE_DIR /
    # BIOIR_KERNEL_CACHE_DIR, plus the bash history.
    --volume "${HOST_CACHE}/cache:/cache"
    # The container user's ~/.cache: HuggingFace hub checkpoints (~/.cache/hf,
    # a path hardcoded in hubs/hf.py), torch hub, and the prek hook environments.
    --volume "${HOST_CACHE}/home-cache:${container_home}/.cache"
    # Staged checkpoints and raw downloads, at the SAME absolute path it has on
    # the host, with BIOIR_CACHE pointing both sides at it. The path has to
    # match: fetch_weights.sh stages by symlinking the probe dirs at the raw
    # downloads, and an absolute symlink written on one side dangles on the
    # other as soon as the two disagree. So weights fetched on the host are
    # usable in here, and vice versa, with no second download.
    --volume "${WEIGHT_CACHE}:${WEIGHT_CACHE}"
    --env "BIOIR_CACHE=${WEIGHT_CACHE}"
  )

  # A linked worktree's .git is a file pointing at an absolute path inside the
  # main checkout; without that path mounted too, git does not work in here.
  git_common_dir="$(cd "${REPO_ROOT}" && cd "$(git rev-parse --git-common-dir)" && pwd)"
  if [[ ${git_common_dir} != "${REPO_ROOT}/.git" ]]; then
    run_args+=(--volume "${git_common_dir}:${git_common_dir}")
  fi

  case "${BIOIR_GPUS:-auto}" in
    none) ;;
    auto)
      if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        run_args+=(--gpus all)
      else
        printf 'No nvidia container runtime found; starting without GPUs.\n' >&2
      fi
      ;;
    *) run_args+=(--gpus "${BIOIR_GPUS}") ;;
  esac

  for var in "${PASS_ENV[@]}"; do
    if [[ -n ${!var:-} ]]; then
      run_args+=(--env "${var}")
    fi
  done

  printf 'Creating container %s from %s\n' "${CONTAINER}" "${IMAGE}"
  docker run "${run_args[@]}" "${IMAGE}" sleep infinity >/dev/null

  cat <<EOF
Caches on the host, shared by every worktree, under ${HOST_CACHE}
Weights: ${WEIGHT_CACHE}, the same path here and on the host
The checkout is bind-mounted, so the package is not installed yet:
    pip install -e '.[dev]'
EOF
elif [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER}")" != "true" ]]; then
  docker start "${CONTAINER}" >/dev/null
fi

# No TTY when the caller is a pipe or a CI job; asking for one fails there.
tty_args=(--interactive)
if [[ -t 0 && -t 1 ]]; then
  tty_args+=(--tty)
fi

exec docker exec "${tty_args[@]}" --workdir "${WORKDIR}" "${CONTAINER}" "${@:-bash}"
