from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from vllm.v1.worker.gpu.split_dvi.draft_block_generator import (
    _activate_substep_loras,
)


def test_activate_substep_loras_uses_one_token_per_request() -> None:
    runner = MagicMock()
    runner.lora_config = object()
    lora_inputs = ((1, 2), (1, 2), {"adapter"})
    runner.lora_state.make_lora_inputs.return_value = lora_inputs
    input_batch = SimpleNamespace(
        req_ids=["req-0", "req-1"],
        idx_mapping_np=np.array([4, 7], dtype=np.int32),
    )

    _activate_substep_loras(runner, input_batch, 2)

    args = runner.lora_state.make_lora_inputs.call_args.args
    assert args[0] == ["req-0", "req-1"]
    np.testing.assert_array_equal(args[1], np.array([4, 7], dtype=np.int32))
    np.testing.assert_array_equal(args[2], np.ones(2, dtype=np.int32))
    runner._set_active_loras.assert_called_once_with(*lora_inputs)


def test_activate_substep_loras_is_noop_when_lora_is_disabled() -> None:
    runner = MagicMock()
    runner.lora_config = None
    input_batch = SimpleNamespace(
        req_ids=["req-0"],
        idx_mapping_np=np.array([0], dtype=np.int32),
    )

    _activate_substep_loras(runner, input_batch, 1)

    runner.lora_state.make_lora_inputs.assert_not_called()
    runner._set_active_loras.assert_not_called()
