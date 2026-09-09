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

import re
import sys
from datetime import datetime
from functools import cache
from os.path import splitext
from pathlib import Path

_LICENSE_PATH = Path(__file__).resolve().parents[1] / ".license-header.txt"
_SPDX = b"SPDX-License-Identifier:"
_YEAR = re.compile(r"(?<=Copyright \(c\) )\d{4}(?:-\d{4})?")
_MD = frozenset({".md", ".mdx"})
_CLOSE = {
    b"---": (re.compile(rb"^---\r?$", re.MULTILINE), "frontmatter"),
    b"<!--": (re.compile(rb"^-->\r?$", re.MULTILINE), "comment"),
}


def _current_year() -> int:
    return datetime.now().astimezone().year


@cache
def _license_for(year: int) -> bytes:
    text = _YEAR.sub(str(year), _LICENSE_PATH.read_text(encoding="utf-8"), count=1)
    return "".join(f"# {line}\n" if line else "#\n" for line in text.splitlines()).encode()


def _has_fields(inner: bytes) -> bool:
    return any(line.strip() and not line.lstrip().startswith(b"#") for line in inner.splitlines())


def _insert(path: Path) -> bool:
    data = path.read_bytes()
    nl = data.find(b"\n")
    opening = (data if nl < 0 else data[:nl]).rstrip(b"\r")
    spec = _CLOSE.get(opening)
    match = None
    inner = b""
    if spec is not None:
        close, kind = spec
        if nl < 0 or (match := close.search(data, nl + 1)) is None:
            raise ValueError(f"Unclosed {kind}: {path}")
        inner = data[nl + 1 : match.start()]
        if _SPDX in inner:
            if opening != b"---" or _has_fields(inner):
                return False
            out = data[: nl + 1] + inner + b"{}\n" + data[match.start() :]
            path.write_bytes(out)
            print(f"Updated {path}")
            return True

    header = _license_for(_current_year())
    if opening == b"---" and match is not None:
        extra = b"" if _has_fields(inner) else b"{}\n"
        out = data[: nl + 1] + header + inner + extra + data[match.start() :]
    else:
        out = b"---\n" + header + b"{}\n---\n" + (b"\n" + data if data else b"")

    path.write_bytes(out)
    print(f"Updated {path}")
    return True


def main(filenames: list[str] | None = None) -> int:
    changed = False
    for filename in filenames if filenames is not None else sys.argv[1:]:
        if splitext(filename)[1].lower() in _MD:
            changed = _insert(Path(filename)) or changed
    return int(changed)


if __name__ == "__main__":
    raise SystemExit(main())
