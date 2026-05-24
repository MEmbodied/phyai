"""Test-suite-wide engine config defaults.

Forces ``EngineConfig.device.target = "cpu"`` for every test so layer
construction defaults to CPU, matching the pre-device-default behaviour
the bulk of the suite was written against. CUDA-specific tests still
work — they either ``.cuda()`` the constructed module after the fact or
pass ``device="cuda"`` explicitly to the constructor.
"""

from __future__ import annotations

import pytest

from phyai.engine_config import (
    DeviceConfig,
    get_engine_config,
    set_engine_config,
)


@pytest.fixture(autouse=True)
def _engine_config_cpu_default():
    saved = get_engine_config()
    set_engine_config(
        saved.replace(
            device=DeviceConfig(target="cpu", params_dtype=saved.device.params_dtype)
        )
    )
    yield
    set_engine_config(saved)
