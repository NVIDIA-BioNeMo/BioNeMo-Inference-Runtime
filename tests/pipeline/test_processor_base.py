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

import pickle
from typing import Any

import pytest

from bionemo_ir.pipeline.processor.base import Processor, ProcessorConfig, SerialProcessor, _ProcessorBase

_payload_executed = False


def _mark_payload() -> None:
    global _payload_executed
    _payload_executed = True


class _Payload:
    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return _mark_payload, ()


class _Dataset:
    def __init__(self, batch: dict[str, list[Any]]) -> None:
        self.batch = batch
        self.map_calls = 0

    def columns(self) -> list[str]:
        return list(self.batch)

    def map_batches(self, *args: Any, **kwargs: Any) -> "_Dataset":
        self.map_calls += 1
        return self


@pytest.fixture
def malicious_pickle() -> bytes:
    global _payload_executed
    _payload_executed = False
    return pickle.dumps(_Payload())


def _config() -> ProcessorConfig:
    return ProcessorConfig(model_source="test")


def _ray_processor() -> Processor:
    processor = Processor.__new__(Processor)
    _ProcessorBase.__init__(processor, _config(), [])
    return processor


def test_serial_processor_rejects_packed_input(malicious_pickle: bytes) -> None:
    processor = SerialProcessor(_config(), [])

    with pytest.raises(ValueError, match="Input column __data__ is reserved"):
        processor([{"__data__": malicious_pickle}])

    assert not _payload_executed


def test_ray_processor_rejects_packed_input(malicious_pickle: bytes) -> None:
    processor = _ray_processor()
    dataset = _Dataset({"__data__": [malicious_pickle]})

    with pytest.raises(ValueError, match="Input column __data__ is reserved"):
        processor(dataset)

    assert not _payload_executed
    assert dataset.map_calls == 0


def test_processors_accept_flat_input() -> None:
    records = [{"value": 1}, {"value": 2}]
    serial_processor = SerialProcessor(_config(), [])
    ray_processor = _ray_processor()
    dataset = _Dataset({"value": [1, 2]})

    assert serial_processor(records) == records
    assert ray_processor(dataset) is dataset
