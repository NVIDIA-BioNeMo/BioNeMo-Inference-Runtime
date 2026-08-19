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

# Host-side setup, run by devcontainer.json's initializeCommand before Compose
# creates the container. Compose has no hook of its own, which is why both jobs
# below live here rather than in docker-compose.yml.
#
# 1. Creates the bind-mount sources. A bind mount whose source does not exist
#    yet is created by the daemon as root, and the container user -- built from
#    the host UID -- then cannot write it.
# 2. Writes the host UID/GID to .devcontainer/.env, which Compose interpolates
#    into the image's USER_UID/USER_GID build args. docker/Makefile derives
#    these with `id -u`/`id -g`, but Compose interpolation cannot run a command
#    and bash does not export UID, so the values have to reach it through a
#    file. Doing it here keeps all three build paths on the same UID.
#
# Keep the defaults in step with docker/dev.sh and scripts/fetch_weights.sh:
# all three resolve the same two roots, so a Compose session, a docker/dev.sh
# shell and a host-side fetch share one cache.
set -euo pipefail

HOST_CACHE="${BIOIR_HOST_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/bionemo-ir-dev}"
WEIGHT_CACHE="${BIOIR_CACHE:-${HOME}/.cache/bionemo_ir}"

mkdir -p "${HOST_CACHE}/cache" "${HOST_CACHE}/home-cache" "${WEIGHT_CACHE}"

# Rewrite rather than append, so repeated runs do not stack duplicate keys, and
# carry over every other line so cache overrides put here by hand survive.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
ENV_TMP="$(mktemp "${ENV_FILE}.XXXXXX")"
trap 'rm -f "${ENV_TMP}"' EXIT

if [[ -f ${ENV_FILE} ]]; then
  grep -v -E '^(USER_UID|USER_GID)=' "${ENV_FILE}" >"${ENV_TMP}" || true
fi
printf 'USER_UID=%s\nUSER_GID=%s\n' "$(id -u)" "$(id -g)" >>"${ENV_TMP}"
mv "${ENV_TMP}" "${ENV_FILE}"
