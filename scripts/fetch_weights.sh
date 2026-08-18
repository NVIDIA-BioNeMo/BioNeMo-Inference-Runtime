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

# Download model checkpoints and metadata, then stage them into the directories
# the library probes, so a separate pytest process (local shell or IDE) resolves
# them with no *_CKPT environment variable.
#
# Two sources, selected by --source:
#
#   public  Upstream publishers — the OpenFold S3 bucket and HuggingFace. Needs
#           no credentials. This is what an external contributor uses.
#   ngc     The NGC org holding the same weights. Used only when a token or an
#           existing `ngc` login is present.
#   auto    (default) ngc when it is usable, else public.
#
# Weights that neither source can supply are skipped with a note, not an error:
# the tests that want them skip themselves when their env var is unset, so a
# partial stage still runs everything it can.
#
# See docs/ref/model-weights.md for which models each source covers and the
# licence terms that come with them.
set -euo pipefail

# Mirrors bionemo_ir.CACHE_DIR (pure bash, no torch import). Keep in sync
# with bionemo_ir/__init__.py.
cache_root() {
  printf '%s\n' "${BIOIR_CACHE:-${HOME}/.cache/bionemo_ir}"
}

# Raw download staging area (the probe cache symlinks into it). On CI it's
# /packages/model_cache (an on-node path, off the container home — not a shared
# cache, so it is re-fetched each job); off-CI it lands under cache_root so a dev
# box stages without a writable /packages. An explicit MODEL_CACHE_DIR always wins.
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${CI:+/packages/model_cache}}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(cache_root)/model_cache}"

# The NGC org/team holding these weights is deployment-specific and has no
# default here: `--source ngc` is opt-in and requires both to be set.
NGC_ORG="${BIOIR_NGC_ORG:-}"
NGC_TEAM="${BIOIR_NGC_TEAM:-}"

# Model families. Each is one NGC model holding every file listed for it below;
# under --source public each file has its own URL in PUBLIC_URLS. A fetch pulls
# only the files the active target needs — naming a family activates all of its
# rows, naming a single weight activates just that one.
FAMILIES=("alphafold2" "boltz" "openfold2" "openfold3" "protenix")

# One row per checkpoint file: "<family>|<filename>|<probe_dir>|<env_var>".
#   family    — which FAMILIES model the file belongs to.
#   filename  — the file's name within that family.
#   probe_dir — hub/cache key (FoldingSupportMatrix value): the probe subdir and
#               the load_local_weights() key.
#   env_var   — the <MODEL>_CKPT env var the test phase exports.
WEIGHTS=(
  "alphafold2|alphafold2_1.pt|alphafold2_1|ALPHAFOLD2_1_CKPT"
  "alphafold2|alphafold2_2.pt|alphafold2_2|ALPHAFOLD2_2_CKPT"
  "alphafold2|alphafold2_3.pt|alphafold2_3|ALPHAFOLD2_3_CKPT"
  "alphafold2|alphafold2_4.pt|alphafold2_4|ALPHAFOLD2_4_CKPT"
  "alphafold2|alphafold2_5.pt|alphafold2_5|ALPHAFOLD2_5_CKPT"
  "alphafold2|params_model_1_multimer_v3.pt|alphafold2_multimer_1|ALPHAFOLD2_MULTIMER_1_CKPT"
  "alphafold2|alphafold2_multimer_2.pt|alphafold2_multimer_2|ALPHAFOLD2_MULTIMER_2_CKPT"
  "alphafold2|alphafold2_multimer_3.pt|alphafold2_multimer_3|ALPHAFOLD2_MULTIMER_3_CKPT"
  "alphafold2|alphafold2_multimer_4.pt|alphafold2_multimer_4|ALPHAFOLD2_MULTIMER_4_CKPT"
  "alphafold2|alphafold2_multimer_5.pt|alphafold2_multimer_5|ALPHAFOLD2_MULTIMER_5_CKPT"
  "boltz|boltz1_conf.ckpt|boltz-1|BOLTZ1_CKPT"
  "boltz|boltz2_conf.ckpt|boltz-2|BOLTZ2_CKPT"
  "boltz|boltz2_aff.ckpt|boltz-2-affinity|BOLTZ2_AFFINITY_CKPT"
  "openfold2|finetuning_2.pt|openfold2_finetuning_2|OPENFOLD2_FINETUNING_2_CKPT"
  "openfold2|finetuning_3.pt|openfold2_finetuning_3|OPENFOLD2_FINETUNING_3_CKPT"
  "openfold2|finetuning_4.pt|openfold2_finetuning_4|OPENFOLD2_FINETUNING_4_CKPT"
  "openfold2|finetuning_5.pt|openfold2_finetuning_5|OPENFOLD2_FINETUNING_5_CKPT"
  "openfold2|finetuning_no_templ_1.pt|openfold2_no_templ_1|OPENFOLD2_NO_TEMPL_1_CKPT"
  "openfold2|finetuning_no_templ_2.pt|openfold2_no_templ_2|OPENFOLD2_NO_TEMPL_2_CKPT"
  "openfold2|finetuning_no_templ_ptm_1.pt|openfold2_no_templ_ptm_1|OPENFOLD2_NO_TEMPL_PTM_1_CKPT"
  "openfold2|finetuning_ptm_1.pt|openfold2_ptm_1|OPENFOLD2_PTM_1_CKPT"
  "openfold2|finetuning_ptm_2.pt|openfold2_ptm_2|OPENFOLD2_PTM_2_CKPT"
  "openfold3|of3-p2-155k.pt|openfold3|OPENFOLD3_CKPT"
  "protenix|protenix-v2.pt|protenix-v2|PROTENIX_V2_CKPT"
)

# Non-checkpoint metadata files that live alongside checkpoints.
# Format: "<family>|<filename>|<env_var>|<kind>"
#   kind = "file"    → export env_var pointing at the raw file.
#   kind = "tar"     → extract the tar into MODEL_CACHE_DIR/<family>_<basename>/
#                      and export env_var pointing at the extracted directory.
METADATA=(
  "boltz|ccd.pkl|BOLTZ_CCD_PATH|file"
  "boltz|mols.tar|BOLTZ_MOL_DIR|tar"
)

# Public, no-credentials download URLs, keyed "<family>/<filename>".
#
# The AlphaFold2 rows are deliberately absent: DeepMind publishes JAX ``.npz``
# parameters, and the loader here wants an OpenFold-format PyTorch state dict.
# Converting is a local step an external user runs once, so these stage only from
# a path the user supplies. The licence is not the obstacle — see
# docs/ref/model-weights.md.
PUBLIC_URLS=(
  "openfold2/finetuning_2.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_2.pt"
  "openfold2/finetuning_3.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_3.pt"
  "openfold2/finetuning_4.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_4.pt"
  "openfold2/finetuning_5.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_5.pt"
  "openfold2/finetuning_no_templ_1.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_no_templ_1.pt"
  "openfold2/finetuning_no_templ_2.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_no_templ_2.pt"
  "openfold2/finetuning_no_templ_ptm_1.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_no_templ_ptm_1.pt"
  "openfold2/finetuning_ptm_1.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_ptm_1.pt"
  "openfold2/finetuning_ptm_2.pt|https://openfold.s3.amazonaws.com/openfold_params/finetuning_ptm_2.pt"
  "boltz/boltz1_conf.ckpt|https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
  "boltz/boltz2_conf.ckpt|https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"
  "boltz/boltz2_aff.ckpt|https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt"
  "boltz/ccd.pkl|https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
  "boltz/mols.tar|https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
  "protenix/protenix-v2.pt|https://huggingface.co/TMF001/protenix-v2-weights/resolve/main/protenix-v2.pt"
  # GATED: anonymous GET returns 401. Accept the terms on the model page and
  # export HF_TOKEN; without one the download degrades to a skip.
  "openfold3/of3-p2-155k.pt|https://huggingface.co/OpenFold/OpenFold3/resolve/main/checkpoints/of3-p2-155k.pt"
)

# Field accessors (keep the call sites readable).
row_family() { printf '%s\n' "${1%%|*}"; }
row_file() {
  local r="${1#*|}"
  printf '%s\n' "${r%%|*}"
}
row_dir() {
  local r="${1#*|*|}"
  printf '%s\n' "${r%%|*}"
}
row_env() { printf '%s\n' "${1##*|}"; }

meta_family() { printf '%s\n' "${1%%|*}"; }
meta_file() {
  local r="${1#*|}"
  printf '%s\n' "${r%%|*}"
}
meta_env() {
  local r="${1#*|*|}"
  printf '%s\n' "${r%%|*}"
}
meta_kind() { printf '%s\n' "${1##*|}"; }

public_url() {
  local key="$1" row
  for row in "${PUBLIC_URLS[@]}"; do
    [[ "${row%%|*}" == "${key}" ]] && {
      printf '%s\n' "${row#*|}"
      return 0
    }
  done
  return 1
}

NGC_RETRY_ATTEMPTS=3
NGC_RETRY_DELAY_SECONDS=5

print_help() {
  cat <<EOF
fetch_weights.sh — download model checkpoints and stage them for the test suite.

Usage:
  fetch_weights.sh                     Stage everything the selected source covers.
  fetch_weights.sh --model MODEL       Stage one family or one weight.
  fetch_weights.sh --source public     Force upstream public downloads (no credentials).
  fetch_weights.sh --source ngc        Force NGC (needs BIOIR_NGC_ORG/_TEAM + credentials).
  fetch_weights.sh --source auto       NGC when usable, else public (default).
  fetch_weights.sh --source none       Stage nothing; only report what is resolvable.
  fetch_weights.sh -h | --help         Show this help.

MODEL is a family name:
  ${FAMILIES[*]}
or a single weight's hub/cache name.

Staged weights resolve through the library's cache probe with no *_CKPT env var.
Anything this script cannot fetch is reported and skipped; the tests that need it
skip themselves.

Environment:
  MODEL_CACHE_DIR   where raw downloads land (default: <cache>/model_cache)
  BIOIR_CACHE       cache root (default: ~/.cache/bionemo_ir)
  BIOIR_CHECKPOINTS checkpoint probe dir override
  BIOIR_METADATA    metadata probe dir override
  HF_TOKEN          HuggingFace token, for gated repos (OpenFold3)
  BIOIR_NGC_ORG     NGC org, required by --source ngc
  BIOIR_NGC_TEAM    NGC team, required by --source ngc
  ALPHAFOLD2_DIR    directory of locally converted AlphaFold2 *.pt files
EOF
}

FETCH_SOURCE="auto"
TARGET_MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || {
        echo "--source needs a value" >&2
        exit 2
      }
      FETCH_SOURCE="$2"
      shift
      ;;
    --source=*) FETCH_SOURCE="${1#--source=}" ;;
    --model)
      [[ $# -ge 2 ]] || {
        echo "--model needs a value" >&2
        exit 2
      }
      TARGET_MODEL="$2"
      shift
      ;;
    --model=*) TARGET_MODEL="${1#--model=}" ;;
    -h | --help)
      print_help
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      print_help >&2
      exit 2
      ;;
  esac
  shift
done

case "${FETCH_SOURCE}" in
  auto | ngc | public | none) ;;
  *)
    echo "--source must be auto, ngc, public or none (got '${FETCH_SOURCE}')" >&2
    exit 2
    ;;
esac

# Resolve the target set: which WEIGHTS rows to stage and which FAMILIES to
# fetch. A target is a family name (all its rows) or a single weight's
# probe/cache name (one row); no target means every family.
declare -a ACTIVE_ROWS=()
declare -a ACTIVE_FAMILIES=()

# Append FAMILY to ACTIVE_FAMILIES unless it is already there. Load-bearing: the
# fetch loop runs one background job per entry and they share that family's
# <family>_version.txt / <family>_dir_path.txt, so a duplicate would race two
# concurrent downloads and two writers over the same files. Naming several weights
# of one family (TEST_FAMILIES=alphafold2_1,alphafold2_multimer_1) hits this.
add_family() {
  local family="$1" f
  for f in ${ACTIVE_FAMILIES[@]+"${ACTIVE_FAMILIES[@]}"}; do
    [[ "${f}" == "${family}" ]] && return 0
  done
  ACTIVE_FAMILIES+=("${family}")
}

add_target() {
  local target="$1" f w
  for f in "${FAMILIES[@]}"; do
    if [[ "${f}" == "${target}" ]]; then
      add_family "${target}"
      for w in "${WEIGHTS[@]}"; do
        [[ "$(row_family "$w")" == "${target}" ]] && ACTIVE_ROWS+=("$w")
      done
      return 0
    fi
  done
  for w in "${WEIGHTS[@]}"; do
    if [[ "$(row_dir "$w")" == "${target}" ]]; then
      ACTIVE_ROWS+=("$w")
      add_family "$(row_family "$w")"
      return 0
    fi
  done
  # Fatal, not a warning: the tests that want these weights pytest.skip when the
  # env var is missing, so a typo would drop coverage without failing anything.
  echo "unknown model '${target}'. Families: ${FAMILIES[*]}" >&2
  exit 2
}

if [[ -n "${TARGET_MODEL}" ]]; then
  add_target "${TARGET_MODEL}"
elif [[ -n "${TEST_FAMILIES:-}" ]]; then
  # A CI shard sets TEST_FAMILIES so it stages only the checkpoints it runs.
  IFS=', ' read -r -a _targets <<<"${TEST_FAMILIES}"
  for target in ${_targets[@]+"${_targets[@]}"}; do
    [[ -n "${target}" ]] && add_target "${target}"
  done
  ((${#ACTIVE_FAMILIES[@]})) || {
    echo "TEST_FAMILIES selected nothing. Families: ${FAMILIES[*]}" >&2
    exit 2
  }
  echo "staging checkpoint families: ${ACTIVE_FAMILIES[*]}"
else
  ACTIVE_ROWS=("${WEIGHTS[@]}")
  ACTIVE_FAMILIES=("${FAMILIES[@]}")
fi

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

NGC=""

ngc_is_usable() {
  # Prefer an already-installed, working CLI; only fetch the bundled Linux build
  # when none is usable. `ngc --version` gates on "usable", not just "present".
  if command -v ngc >/dev/null 2>&1 && ngc --version >/dev/null 2>&1; then
    NGC="ngc"
  elif [[ -n "${NGC_API_KEY:-}" ]]; then
    # Extract into a temp dir, not the CWD: in the dev container the CWD is the
    # bind-mounted repo, so unzipping here would litter the host working tree.
    local ngc_tmp
    ngc_tmp=$(mktemp -d)
    wget -q --content-disposition \
      https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.21.0/files/ngccli_linux.zip \
      -O "${ngc_tmp}/ngccli_linux.zip" || return 1
    unzip -q -o "${ngc_tmp}/ngccli_linux.zip" -d "${ngc_tmp}" || return 1
    chmod u+x "${ngc_tmp}/ngc-cli/ngc"
    NGC="${ngc_tmp}/ngc-cli/ngc"
  else
    return 1
  fi
  # Without an org/team there is nothing to address, whatever the credentials.
  [[ -n "${NGC_ORG}" && -n "${NGC_TEAM}" ]] || return 1
  # The CLI takes its key and target from the environment, never from argv. A
  # freshly extracted binary has no ~/.ngc/config, so without these every query
  # below runs anonymously and 404s. NGC_API_KEY unset means "use an existing
  # `ngc config set` login" (local dev); org/team are forced either way.
  if [[ -n "${NGC_API_KEY:-}" ]]; then
    export NGC_CLI_API_KEY="${NGC_API_KEY}"
  fi
  export NGC_CLI_ORG="${NGC_ORG}" NGC_CLI_TEAM="${NGC_TEAM}"
  # Cheap credential smoke test, so an unauthenticated CLI is caught here rather
  # than by the 3x retry loop in every family fetch.
  "${NGC}" config current >/dev/null 2>&1 || return 1
  return 0
}

resolve_source() {
  case "${FETCH_SOURCE}" in
    none) return 0 ;;
    public) return 0 ;;
    ngc)
      ngc_is_usable || {
        echo "--source ngc needs BIOIR_NGC_ORG, BIOIR_NGC_TEAM and a usable ngc CLI" >&2
        exit 1
      }
      ;;
    auto)
      # Public sources are the normal case and need no announcement; the
      # resolved source is echoed below either way. Credentials are what
      # changes where the weights come from, so only that is worth a note.
      if ngc_is_usable; then
        FETCH_SOURCE="ngc"
        echo "NGC credentials found — preferring internal weights"
      else
        FETCH_SOURCE="public"
      fi
      ;;
  esac
}

resolve_source
echo "weight source: ${FETCH_SOURCE}"

# ---------------------------------------------------------------------------
# NGC fetch
# ---------------------------------------------------------------------------

query_ngc_json_field() {
  local field=$1 description=$2
  shift 2
  local attempt response value
  local filter="${field} | strings | select(length > 0)"
  for ((attempt = 1; attempt <= NGC_RETRY_ATTEMPTS; attempt++)); do
    if response=$("${NGC}" "$@" --format_type json) \
      && value=$(jq -er "${filter}" <<<"${response}"); then
      printf '%s\n' "${value}"
      return 0
    fi
    echo "NGC ${description} attempt" \
      "${attempt}/${NGC_RETRY_ATTEMPTS} failed" >&2
    ((attempt < NGC_RETRY_ATTEMPTS)) && sleep "${NGC_RETRY_DELAY_SECONDS}"
  done
  echo "NGC ${description} failed after ${NGC_RETRY_ATTEMPTS} attempts" >&2
  return 1
}

# Files of FAMILY this run actually needs: the active WEIGHTS rows plus every
# METADATA row of that family (stage_metadata resolves those by family, not by
# row, and hard-fails when one is missing).
#
# This is the *needed* set, not the family's full contents — a shard that names
# one weight does not fetch its siblings. It is what both the download filter and
# the completeness check key on, so the two cannot disagree.
family_needed_files() {
  local family=$1 row
  for row in ${ACTIVE_ROWS[@]+"${ACTIVE_ROWS[@]}"}; do
    [[ "$(row_family "${row}")" == "${family}" ]] && row_file "${row}"
  done
  for row in "${METADATA[@]}"; do
    [[ "$(meta_family "${row}")" == "${family}" ]] && meta_file "${row}"
  done
}

# True only if DIR holds every file this run needs from FAMILY (no filename has
# spaces). Target-aware on purpose: MODEL_CACHE_DIR persists (an on-node path on
# CI, the user's cache off it), so a dir populated by an earlier, narrower target
# must not be trusted for a later, wider one.
family_dir_complete() {
  local family=$1 dir=$2 f
  for f in $(family_needed_files "${family}"); do
    [[ -f "${dir}/${f}" ]] || return 1
  done
  return 0
}

# Fetch/refresh one family model's latest version into MODEL_CACHE_DIR, limited
# to the files this run needs. Self-contained so it can run as a background job —
# each family has its own cache files, so there are no shared writes between
# concurrent invocations.
fetch_family_ngc() {
  local family=$1
  echo "==== ${family} (ngc) ===="
  local version_file="${MODEL_CACHE_DIR}/${family}_version.txt"
  local dir_file="${MODEL_CACHE_DIR}/${family}_dir_path.txt"

  local latest
  latest=$(query_ngc_json_field '.latestVersionIdStr' "model info for ${family}" \
    registry model info "${NGC_ORG}/${NGC_TEAM}/${family}") || return 1

  if [[ -f "${version_file}" && -f "${dir_file}" ]]; then
    local cached_dir
    cached_dir=$(<"${dir_file}")
    if [[ "$(<"${version_file}")" == "${latest}" && -d "${cached_dir}" ]] \
      && family_dir_complete "${family}" "${cached_dir}"; then
      echo "${family} already up to date (version ${latest})"
      return 0
    fi
  fi

  # Pull only what this run needs. `--file` is repeatable and matches by name
  # within the version, so a shard that runs one model does not pay for its
  # siblings. Names are literal here, so no shell glob can leak in from the
  # WEIGHTS/METADATA tables.
  local -a file_args=() f
  for f in $(family_needed_files "${family}"); do
    file_args+=(--file "${f}")
  done
  if ((${#file_args[@]} == 0)); then
    echo "no files needed from ${family}" >&2
    return 1
  fi

  echo "downloading ${family}:${latest} ($((${#file_args[@]} / 2)) file(s))"
  local dl
  dl=$(query_ngc_json_field '.local_path' "download ${family}:${latest}" \
    registry model download-version "${NGC_ORG}/${NGC_TEAM}/${family}:${latest}" \
    --dest "${MODEL_CACHE_DIR}" "${file_args[@]}") || return 1
  if [[ ! -d "${dl}" ]] || ! family_dir_complete "${family}" "${dl}"; then
    echo "download incomplete for ${family} (missing needed files): ${dl}" >&2
    return 1
  fi
  printf '%s\n' "${dl}" >"${dir_file}"
  printf '%s\n' "${latest}" >"${version_file}"
}

# ---------------------------------------------------------------------------
# Public fetch
# ---------------------------------------------------------------------------

PUBLIC_ROOT="${MODEL_CACHE_DIR}/public"

# Download one public file, skipping when it is already complete. curl -C - so an
# interrupted multi-GB transfer resumes instead of restarting.
fetch_public_file() {
  local family=$1 filename=$2 url dest
  if ! url=$(public_url "${family}/${filename}"); then
    return 2 # no public source for this file
  fi
  dest="${PUBLIC_ROOT}/${family}/${filename}"
  mkdir -p "$(dirname "${dest}")"
  if [[ -s "${dest}" ]]; then
    echo "have ${family}/${filename}"
    return 0
  fi
  echo "downloading ${family}/${filename}"
  local -a auth=()
  # HuggingFace gated repos need a token; sending one to S3 would be pointless
  # but harmless, so scope it to huggingface.co.
  if [[ -n "${HF_TOKEN:-}" && "${url}" == *huggingface.co* ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  # A failed download is "unavailable", not fatal: gated repos (OpenFold3 answers
  # 401 without an accepted licence + HF_TOKEN) and transient network errors both
  # land here, and the tests that want the file skip themselves. Staging reports
  # everything it could not resolve.
  # --retry alone does not bound this: it covers a connection that fails, not one
  # the server accepts and then stops feeding, which would hang the job until its
  # own timeout. --speed-limit/--speed-time abort a transfer that drops under
  # 1 KiB/s for two minutes, and curl counts that as retryable, so a genuinely
  # slow link still gets its three attempts.
  if ! curl -fL --no-progress-meter --retry 3 --retry-delay 5 -C - \
    --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
    ${auth[@]+"${auth[@]}"} \
    -o "${dest}.part" "${url}"; then
    rm -f "${dest}.part"
    echo "could not download ${family}/${filename} (gated, or network) — skipping" >&2
    return 2
  fi
  mv "${dest}.part" "${dest}"
}

fetch_family_public() {
  local family=$1 row f rc
  local -a missing=()
  echo "==== ${family} (public) ===="
  for row in "${ACTIVE_ROWS[@]}" "${METADATA[@]}"; do
    [[ "$(row_family "${row}")" == "${family}" ]] || continue
    f=$(row_file "${row}")
    rc=0
    fetch_public_file "${family}" "${f}" || rc=$?
    ((rc == 0)) || missing+=("${f}")
  done
  if ((${#missing[@]})); then
    echo "not staged for ${family}: ${missing[*]} — see docs/ref/model-weights.md"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Resolution + staging
# ---------------------------------------------------------------------------

# jax_to_pt.py names its output after the DeepMind .npz stem, and the stem must
# not be renamed before conversion (the converter reads the model version out of
# it). The rows above carry the NGC names instead, so map one onto the other and
# accept either in ALPHAFOLD2_DIR.
alphafold2_alias() {
  local stem="${1%.pt}"
  case "${stem}" in
    alphafold2_multimer_*) printf 'params_model_%s_multimer_v3.pt\n' "${stem#alphafold2_multimer_}" ;;
    alphafold2_*) printf 'params_model_%s.pt\n' "${stem#alphafold2_}" ;;
    *) return 1 ;;
  esac
}

# Locate a weight file for a row, whichever source supplied it. An explicit
# ALPHAFOLD2_DIR lets a user point at their own converted AlphaFold2 params.
weight_src() {
  local row=$1 family filename candidate name
  family=$(row_family "${row}")
  filename=$(row_file "${row}")
  if [[ "${family}" == "alphafold2" && -n "${ALPHAFOLD2_DIR:-}" ]]; then
    local -a names=("${filename}")
    if name=$(alphafold2_alias "${filename}"); then names+=("${name}"); fi
    for name in "${names[@]}"; do
      candidate="${ALPHAFOLD2_DIR}/${name}"
      [[ -f "${candidate}" ]] && {
        printf '%s\n' "${candidate}"
        return 0
      }
    done
  fi
  candidate="${PUBLIC_ROOT}/${family}/${filename}"
  [[ -f "${candidate}" ]] && {
    printf '%s\n' "${candidate}"
    return 0
  }
  local dir_file="${MODEL_CACHE_DIR}/${family}_dir_path.txt"
  if [[ -f "${dir_file}" ]]; then
    candidate="$(<"${dir_file}")/${filename}"
    [[ -f "${candidate}" ]] && {
      printf '%s\n' "${candidate}"
      return 0
    }
  fi
  return 1
}

metadata_src() {
  local row=$1 family filename candidate
  family=$(meta_family "${row}")
  filename=$(meta_file "${row}")
  candidate="${PUBLIC_ROOT}/${family}/${filename}"
  [[ -f "${candidate}" ]] && {
    printf '%s\n' "${candidate}"
    return 0
  }
  local dir_file="${MODEL_CACHE_DIR}/${family}_dir_path.txt"
  if [[ -f "${dir_file}" ]]; then
    candidate="$(<"${dir_file}")/${filename}"
    [[ -f "${candidate}" ]] && {
      printf '%s\n' "${candidate}"
      return 0
    }
  fi
  return 1
}

# Directory hubs.local.resolve_cached_checkpoint() probes. Mirrors
# local_checkpoint_dir() exactly: BIOIR_CHECKPOINTS override, else
# cache_root/checkpoints — keep in sync with hubs/local.py.
checkpoints_cache_dir() {
  printf '%s\n' "${BIOIR_CHECKPOINTS:-$(cache_root)/checkpoints}"
}

# Directory hubs.metadata.resolve_cached_metadata() probes. Mirrors it exactly:
# BIOIR_METADATA override, else cache_root/metadata — keep in sync
# with hubs/metadata.py (metadata_cache_dir).
metadata_cache_dir() {
  printf '%s\n' "${BIOIR_METADATA:-$(cache_root)/metadata}"
}

# True if FAMILY is in the active set.
_family_active() {
  local f
  for f in "${ACTIVE_FAMILIES[@]}"; do [[ "${f}" == "$1" ]] && return 0; done
  return 1
}

# Env file the caller sources to pick up the staged paths. Staging into the probe
# dirs is what makes a *separate* pytest process resolve them; this file is for
# the caller's own children.
WEIGHTS_ENV_FILE="${WEIGHTS_ENV_FILE:-${MODEL_CACHE_DIR}/weights.env}"

# Symlink each weight file into <cache>/<probe_dir>/ so a separate pytest process
# resolves it via the cache probe without any *_CKPT env var. Symlink (not copy)
# — the weights are large and MODEL_CACHE_DIR stays put for the job's duration.
stage_checkpoints_into_cache() {
  local probe_root
  probe_root=$(checkpoints_cache_dir)
  local row src probe dest_dir staged=0 skipped=()
  for row in "${ACTIVE_ROWS[@]}"; do
    if ! src=$(weight_src "${row}"); then
      skipped+=("$(row_dir "${row}")")
      continue
    fi
    probe=$(row_dir "${row}")
    dest_dir="${probe_root}/${probe}"
    mkdir -p "${dest_dir}"
    # Wipe first: resolve_cached_checkpoint() returns the sorted-first match, so
    # a stale differently-named symlink could shadow the current one (ln -sfn
    # only overwrites the same basename).
    rm -f "${dest_dir}"/*.pt "${dest_dir}"/*.ckpt
    ln -sfn "${src}" "${dest_dir}/$(basename "${src}")"
    printf 'export %s=%q\n' "$(row_env "${row}")" "${src}" >>"${WEIGHTS_ENV_FILE}"
    echo "staged ${probe} -> ${dest_dir}/$(basename "${src}")"
    staged=$((staged + 1))
  done
  echo "staged ${staged}/${#ACTIVE_ROWS[@]} checkpoints"
  if ((${#skipped[@]})); then
    echo "unavailable (their tests will skip): ${skipped[*]}"
  fi
}

# Resolve metadata assets, then (a) symlink each into the metadata probe dir
# (named by env var) so a separate pytest process resolves it with no env var,
# and (b) record the env var for this run's own children. "file" rows resolve to
# the raw path; "tar" rows are extracted once and resolve to the extracted dir.
stage_metadata() {
  local probe_root row env kind stem extract_dir file_path resolved
  probe_root=$(metadata_cache_dir)
  for row in "${METADATA[@]}"; do
    _family_active "$(meta_family "${row}")" || continue
    env=$(meta_env "${row}")
    # Honour an existing override (e.g. developer pre-set the path).
    [[ -n "${!env:-}" ]] && {
      echo "${env} already set, skipping"
      continue
    }
    if ! file_path=$(metadata_src "${row}"); then
      echo "metadata unavailable for ${env} (its tests will skip)"
      continue
    fi
    kind=$(meta_kind "${row}")
    if [[ "${kind}" == "tar" ]]; then
      stem=$(meta_file "${row}")
      stem="${stem%.tar}"
      extract_dir="${MODEL_CACHE_DIR}/$(meta_family "${row}")_${stem}"
      # Unpack into a scratch dir and rename it into place. Extracting straight
      # into extract_dir would make the `-d` guard above trust whatever a run
      # killed mid-extract (SIGINT, OOM, an evicted node) left behind: the dir
      # exists, the top-level dir below resolves, and staging reports success
      # over a truncated tree. MODEL_CACHE_DIR persists, so that would survive
      # every later run. mv is atomic within one filesystem, and the scratch dir
      # is a sibling.
      if [[ ! -d "${extract_dir}" ]]; then
        echo "extracting $(meta_file "${row}") -> ${extract_dir}"
        local staging_dir="${extract_dir}.part.$$"
        rm -rf "${staging_dir}"
        mkdir -p "${staging_dir}"
        tar -xf "${file_path}" -C "${staging_dir}"
        mv "${staging_dir}" "${extract_dir}"
      fi
      # The archive unpacks a single top-level dir (e.g. mols/); resolve that.
      local unpacked
      unpacked=$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d | head -1)
      [[ -n "${unpacked}" ]] || {
        echo "metadata extract incomplete: ${extract_dir}" >&2
        return 1
      }
      resolved="${unpacked}"
    else
      resolved="${file_path}"
    fi
    mkdir -p "${probe_root}"
    ln -sfn "${resolved}" "${probe_root}/${env}"
    printf 'export %s=%q\n' "${env}" "${resolved}" >>"${WEIGHTS_ENV_FILE}"
    echo "staged ${env} -> ${resolved}"
  done
}

mkdir -p "${MODEL_CACHE_DIR}"
: >"${WEIGHTS_ENV_FILE}"

if [[ "${FETCH_SOURCE}" != "none" ]]; then
  # Fetch all families concurrently to saturate the network — each family is
  # independent, so a serial loop just idles the link. Collect PIDs and fail if
  # any fetch fails.
  fetch_pids=()
  for i in "${!ACTIVE_FAMILIES[@]}"; do
    if [[ "${FETCH_SOURCE}" == "ngc" ]]; then
      fetch_family_ngc "${ACTIVE_FAMILIES[$i]}" &
    else
      fetch_family_public "${ACTIVE_FAMILIES[$i]}" &
    fi
    fetch_pids["$i"]=$!
  done
  fetch_failed=0
  for i in "${!ACTIVE_FAMILIES[@]}"; do
    if ! wait "${fetch_pids[$i]}"; then
      echo "checkpoint fetch failed for ${ACTIVE_FAMILIES[$i]}" >&2
      fetch_failed=1
    fi
  done
  ((fetch_failed)) && exit 1
fi

echo "Staging checkpoints into the local cache probe dir..."
stage_checkpoints_into_cache

echo "Staging metadata into the local cache probe dir..."
stage_metadata

echo "wrote ${WEIGHTS_ENV_FILE}"
