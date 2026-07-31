# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project logger with the legacy TRT-LLM logger interface."""

import logging
import os
import sys

_LEVELS = {
    "internal_error": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "verbose": logging.DEBUG,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}
_PREFIXES = {
    "internal_error": "[F]",
    "error": "[E]",
    "warning": "[W]",
    "info": "[I]",
    "verbose": "[V]",
    "debug": "[D]",
    "trace": "[D]",
}


class Logger:
    """Thin logging adapter preserving the logger API used by the project."""

    def __init__(self) -> None:
        requested_level = os.environ.get("TENSORRT_BIONEMO_LOG_LEVEL",
                                         "error").lower()
        self._level = (requested_level
                       if requested_level in _LEVELS else "error")
        self._appeared_keys: set[object] = set()
        self._logger = logging.getLogger("TensorRT-BioNeMo")
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s] %(message)s",
                    datefmt="%m/%d/%Y-%H:%M:%S",
                ))
            self._logger.addHandler(handler)
        self._logger.setLevel(_LEVELS[self._level])

    @property
    def level(self) -> str:
        return self._level

    def set_level(self, level: str) -> None:
        level = level.lower()
        if level not in _LEVELS:
            raise ValueError(f"Unsupported log level: {level}")
        self._level = level
        self._logger.setLevel(_LEVELS[level])

    def log(self, level: str, *message: object) -> None:
        parts = ["[TensorRT-BioNeMo]"]
        parts.append(_PREFIXES[level])
        parts.extend(map(str, message))
        self._logger.log(_LEVELS[level], " ".join(parts))

    def log_once(self, level: str, *message: object, key: object) -> None:
        if key not in self._appeared_keys:
            self._appeared_keys.add(key)
            self.log(level, *message)

    def critical(self, *message: object) -> None:
        self.log("internal_error", *message)

    fatal = critical

    def error(self, *message: object) -> None:
        self.log("error", *message)

    def warning(self, *message: object) -> None:
        self.log("warning", *message)

    def info(self, *message: object) -> None:
        self.log("info", *message)

    def debug(self, *message: object) -> None:
        self.log("debug", *message)

    def critical_once(self, *message: object, key: object) -> None:
        self.log_once("internal_error", *message, key=key)

    fatal_once = critical_once

    def error_once(self, *message: object, key: object) -> None:
        self.log_once("error", *message, key=key)

    def warning_once(self, *message: object, key: object) -> None:
        self.log_once("warning", *message, key=key)

    def info_once(self, *message: object, key: object) -> None:
        self.log_once("info", *message, key=key)

    def debug_once(self, *message: object, key: object) -> None:
        self.log_once("debug", *message, key=key)


logger = Logger()

__all__ = ["Logger", "logger"]
