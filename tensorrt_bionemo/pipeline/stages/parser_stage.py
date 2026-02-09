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

import hashlib
from io import StringIO
from typing import Any, Dict, List, Optional, Type

from tensorrt_bionemo.data.parsers.a3m import parse_a3m_content
from tensorrt_bionemo.data.schemas import (InputRequest, MSARecord, Polymer,
                                           Template)
from tensorrt_bionemo.data.schemas.basic import (InputParsed, MSAParsed,
                                                 PolymerParsed, TemplateParsed)
from tensorrt_bionemo.pipeline.base import numpy_to_dict
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class FileContentCache:
    """Cache for file content with deduplication by content hash.

    This cache provides two levels of caching:
    1. Path-based: Maps file paths to content hashes (avoids re-reading files)
    2. Content-based: Maps content hashes to actual content (deduplicates identical content)

    This means:
    - Same file path referenced multiple times → read once
    - Different files/inline content with same content → stored once
    """

    def __init__(self):
        # Maps content hash -> actual content
        self._content_by_hash: Dict[str, str] = {}
        # Maps file path -> content hash
        self._hash_by_path: Dict[str, str] = {}

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute a hash for the given content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get_by_path(self, path: str) -> Optional[str]:
        """Get cached content for a file path."""
        content_hash = self._hash_by_path.get(path)
        if content_hash is not None:
            return self._content_by_hash.get(content_hash)
        return None

    def get_by_content(self, content: str) -> Optional[str]:
        """Check if content is already cached (by hash), return cached version."""
        content_hash = self._compute_hash(content)
        return self._content_by_hash.get(content_hash)

    def cache_content(self, content: str, path: Optional[str] = None) -> str:
        """Cache content by its hash, optionally associating with a path.

        Args:
            content: The content string to cache.
            path: Optional file path to associate with this content.

        Returns:
            The cached content (may be an existing cached version if duplicate).
        """
        content_hash = self._compute_hash(content)

        # Check if we already have this content
        if content_hash in self._content_by_hash:
            # Return existing cached content
            cached_content = self._content_by_hash[content_hash]
        else:
            # Store new content
            self._content_by_hash[content_hash] = content
            cached_content = content

        # Associate path with content hash if provided
        if path is not None:
            self._hash_by_path[path] = content_hash

        return cached_content

    def get_or_load(self, path: str) -> str:
        """Get cached content or load from file and cache it."""
        # Check if path is already cached
        cached = self.get_by_path(path)
        if cached is not None:
            return cached

        # Load from file
        with open(path, "r") as f:
            content = f.read()

        # Cache and return (will deduplicate if same content exists)
        return self.cache_content(content, path=path)

    def get_or_cache_content(self, content: str) -> str:
        """Get cached version of content or cache it.

        If identical content exists in cache, returns the cached version.
        Otherwise, caches the new content and returns it.
        """
        content_hash = self._compute_hash(content)

        if content_hash in self._content_by_hash:
            return self._content_by_hash[content_hash]

        self._content_by_hash[content_hash] = content
        return content

    def clear(self) -> None:
        """Clear all cached content."""
        self._content_by_hash.clear()
        self._hash_by_path.clear()

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "unique_contents": len(self._content_by_hash),
            "cached_paths": len(self._hash_by_path),
        }


class ParserUDF(StatefulStageUDF):
    """Parser UDF that converts InputRequest to InputParsed.

    This UDF includes a file content cache to avoid repeated disk reads
    when the same MSA or template file is referenced multiple times.
    The cache persists across multiple udf_for_item calls within the
    same UDF instance (i.e., within the same Ray actor).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance-level cache for file content
        self._file_cache = FileContentCache()

    def _get_content_with_cache(self, record: Dict[str, Any]) -> str:
        """Get content from record, using cache for file paths."""
        # If content is provided directly, return it
        content = record.get("content")
        if content is not None:
            return content

        # If path is provided, use cache
        path = record.get("path")
        if path is not None:
            return self._file_cache.get_or_load(path)

        raise ValueError("No content or path available in record")

    def _parse_msa(self, msa_data, cache: FileContentCache) -> MSAParsed:
        """Parse MSA data and return MSAParsed object.

        Uses cache for both file paths and content deduplication.
        Parses the A3M/MSA content into MSAParsed structure.

        Args:
            msa_data: MSARecord or dict with MSA data.
            cache: File content cache for deduplication.

        Returns:
            MSAParsed: Parsed MSA with sequences, raw, and descriptions.
        """
        if isinstance(msa_data, dict):
            msa = MSARecord(**msa_data)
        elif isinstance(msa_data, MSARecord):
            msa = msa_data
        else:
            raise ValueError(f"Unsupported MSA data type: {type(msa_data)}")

        content = msa.get("content")
        if content is not None:
            content = cache.get_or_cache_content(content)
        else:
            path = msa.get("path")
            if path is not None:
                content = cache.get_or_load(path)
            else:
                raise ValueError("MSARecord has no content or path")

        msa_format = msa.get("format", "a3m")
        if msa_format == "a3m":
            return parse_a3m_content(StringIO(content))
        else:
            # TODO: Add support for other MSA formats (e.g., sto, fasta, clustal)
            raise ValueError(f"Unsupported MSA format: '{msa_format}'. "
                             f"Currently only 'a3m' format is supported.")

    def _parse_template(self, template_data,
                        cache: FileContentCache) -> TemplateParsed:
        """Parse template data and return TemplateParsed object with content loaded.

        Uses cache for both file paths and content deduplication.
        If same content is provided multiple times (even inline),
        it will be deduplicated.

        Args:
            template_data: Template or dict with template data.
            cache: File content cache for deduplication.
        """
        if isinstance(template_data, dict):
            template = Template(**template_data)
        elif isinstance(template_data, Template):
            template = template_data
        else:
            raise ValueError(
                f"Unsupported template data type: {type(template_data)}")

        # Get content - use cache for both file loading and content deduplication
        content = template.get("content")
        if content is not None:
            # Inline content provided - deduplicate by hash
            content = cache.get_or_cache_content(content)
        else:
            path = template.get("path")
            if path is not None:
                # Load from file with caching
                content = cache.get_or_load(path)
            else:
                raise ValueError("Template has no content or path")

        return TemplateParsed(content=content,
                              format=template.get("format", "cif"))

    def _parse_polymer(self, polymer_data,
                       cache: FileContentCache) -> PolymerParsed:
        """Parse a single polymer and return PolymerParsed object.

        Args:
            polymer_data: Polymer or dict with polymer data.
            cache: File content cache for the current item.
        """
        if isinstance(polymer_data, dict):
            polymer = Polymer(**polymer_data)
        elif isinstance(polymer_data, Polymer):
            polymer = polymer_data
        else:
            raise ValueError(
                f"Unsupported polymer data type: {type(polymer_data)}")

        # Parse MSAs with cache
        msas_parsed = None
        msas = polymer.get("msas")
        if msas:
            msas_parsed = [self._parse_msa(msa, cache) for msa in msas]

        # Parse paired MSAs with cache
        paired_msas_parsed = None
        paired_msas = polymer.get("paired_msas")
        if paired_msas:
            paired_msas_parsed = [
                self._parse_msa(msa, cache) for msa in paired_msas
            ]

        # Parse templates with cache
        templates_parsed = None
        templates = polymer.get("templates")
        if templates:
            templates_parsed = [
                self._parse_template(t, cache) for t in templates
            ]

        return PolymerParsed(
            polymer_type=polymer.get("polymer_type"),
            chain_id=polymer.get("chain_id"),
            sequence=polymer.get("sequence"),
            msas=msas_parsed,
            paired_msas=paired_msas_parsed,
            templates=templates_parsed,
        )

    def _parse_input_request(self, input: InputRequest,
                             cache: FileContentCache) -> InputParsed:
        """Parse the entire input request and return InputParsed object.

        Args:
            input: InputRequest to parse.
            cache: File content cache for the current item.
        """
        polymers = input.get("polymers", [])
        # Deserialize numpy arrays to Python dicts
        # This because ray stage will be serialized by pyarrow
        polymers = numpy_to_dict(polymers)
        polymers_parsed: List[PolymerParsed] = [
            self._parse_polymer(p, cache) for p in polymers
        ]

        return InputParsed(
            input_id=input.get("input_id"),
            polymers=polymers_parsed,
        )

    async def udf_for_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single row and return parsed result.

        Uses instance-level file cache to avoid repeated disk reads
        when the same file is referenced multiple times across rows.
        """
        record = InputRequest(**row["record"])
        # Use the instance-level cache (persists across items in same actor)
        parsed = self._parse_input_request(record, self._file_cache)
        return {"parsed": parsed}

    def on_row_error(self, row: Dict[str, Any],
                     error: Exception) -> Dict[str, Any]:
        return {"parsed": None}


class ParserStage(StatefulStage):
    """
    A stage that parses the input.
    """

    fn: Type[StatefulStageUDF] = ParserUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {
            "record":
            "A record of the input. "
            "See tensorrt_bionemo.data.schemas.InputRequest for details."
        }
