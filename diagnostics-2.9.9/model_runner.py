"""
Otter Model Runner.
Inherits natively from GPUModelRunner to maintain the correct vLLM execution architecture.
Provides strict dummy shells for unsupported PyTorch primitives to reserve space for
future C++ implementations.
"""

import logging
import os
import types
from copy import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import otter_torch  # noqa: F401

from vllm.config import (
    VllmConfig,
    CUDAGraphMode,
)
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_model_runner import ExecuteModelState, PerLayerAttnMetadata
from .attention import OtterAttentionState, _resolve_kv_dtype
import vllm.v1.worker.gpu_model_runner as gpu_mr_mod
import vllm.model_executor.model_loader.weight_utils as weight_utils

from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.v1.utils import record_function_or_nullcontext
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    maybe_create_ubatch_slices,
)
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
)
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce

from dataclasses import dataclass, replace, fields, is_dataclass
import numpy as np

# logger = logging.getLogger(__name__)
logger = logging.getLogger(f"vllm.{__name__}")   # -> "vllm.otter_vllm.model_runner"


def _get_num_kv_heads(model_config) -> int:
    """num_kv_heads for the flat head-major slot index. Reads the text
    config (same source the attention impl's num_kv_heads ultimately comes
    from); the Llama-3-family GQA mapping heads % n_heads == 0 holds."""
    n_heads = int(model_config.hf_config.num_attention_heads)
    n_kv = getattr(model_config.hf_config, "num_key_value_heads", None)
    return max(1, (n_heads if n_kv is None else int(n_kv)) // get_tensor_model_parallel_world_size())

# import vllm.model_executor.layers.linear as _vllm_linear
# _vllm_linear.otter_graph_for_tp = otter_graph_for_tp

import vllm.model_executor.layers.vocab_parallel_embedding as vpe
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from otter_vllm.distributed.tp_runtime_context import use_tp, OtterGraphForTP
from otter_vllm.engine_config import (
    resolve_chunk_size,
    resolve_max_len,
    resolve_otter_graph_enabled,
)

def patch_apply():
    """Install the patches."""
    global PATCH_APPLIED
    assert PATCH_APPLIED is False, "patches are already installed!"

    _patch_embedding_init()
    _patch_embedding_forward()
    _patch_logits_gather()
    PATCH_APPLIED = True



def _patch_embedding_init():
    _orig_init = vpe.VocabParallelEmbedding.__init__

    def _init(self, *args, **kwargs):
        # Temporarily set tp size = 1 and tp rank = 0 to let VocabParallelEmbedding load whole embed/lmhead weight
        orig_get_tp_world_size = vpe.get_tensor_model_parallel_world_size
        orig_get_tp_rank = vpe.get_tensor_model_parallel_rank
        vpe.get_tensor_model_parallel_world_size = lambda: 1
        vpe.get_tensor_model_parallel_rank = lambda: 0
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            vpe.get_tensor_model_parallel_world_size = orig_get_tp_world_size
            vpe.get_tensor_model_parallel_rank = orig_get_tp_rank

    vpe.VocabParallelEmbedding.__init__ = _init


def _patch_embedding_forward():
    def _forward_native(self, input_):
        if self.tp_size > 1: # in _init we set self.tp_size = 1, so this will not be run.
            masked_input, input_mask = vpe.get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        if self.tp_size > 1: # in _init we set self.tp_size = 1, so this will not be run.
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            # full weight -> lookup is already complete; NO collective.
            output = output_parallel
        return output

    vpe.VocabParallelEmbedding.forward_native = _forward_native


def _patch_logits_gather():
    LogitsProcessor._gather_logits = lambda self, logits: logits


def _patch_get_masked_input_and_mask():
    def _get_masked_input_and_mask(
        input_: torch.Tensor,
        org_vocab_start_index: int,
        org_vocab_end_index: int,
        num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # torch.compile will fuse all of the pointwise ops below
        # into a single kernel, making it very fast
        org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        added_vocab_mask = (input_ >= added_vocab_start_index) & (
            input_ < added_vocab_end_index
        )
        added_offset = (
            added_vocab_start_index
            - (org_vocab_end_index - org_vocab_start_index)
            - num_org_vocab_padding
        )
        valid_offset = (org_vocab_start_index * org_vocab_mask.to(torch.int32)) + (
            added_offset * added_vocab_mask.to(torch.int32)
        )
        vocab_mask = org_vocab_mask | added_vocab_mask
        input_ = vocab_mask.to(torch.int32) * (input_ - valid_offset)
        return input_, ~vocab_mask
    
    vpe.get_masked_input_and_mask = _get_masked_input_and_mask


_patch_get_masked_input_and_mask()


PATCH_APPLIED = False
#   applied -> every rank holds the full vocab: no embedding all-reduce, and
#              compute_logits is a local matmul that graph 2 can capture.
#   not     -> vLLM's sharded vocab: one extra segment boundary at the
#              embedding, and compute_logits all-gathers, so it runs eagerly.
# patch_apply()


def _merge_cpu_gpu_buffers(
    buffers: list["CpuGpuBuffer"],
    device: torch.device,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-point a group of CpuGpuBuffers to alias one shared CPU arena and one
    shared GPU arena, so callers can copy_ the whole arena in a single command
    instead of one dispatch per buffer per decode step.

    Rewires .cpu/.gpu/.np views only -- shape, dtype, and object identity are
    unchanged, so existing reads/writes (e.g. `self.positions.np[...] = ...`)
    keep working. Caller must issue the single merged
    `arena_gpu.copy_(arena_cpu, non_blocking=True)` itself (see
    OtterModelRunner._prepare_inputs) instead of each buffer's own copy_to_gpu().
    """
    # Align offsets to the widest dtype itemsize so each buffer's .view(dtype)
    # below satisfies torch's "storage_offset() must be divisible by itemsize".
    arena_align = max(buf.cpu.element_size() for buf in buffers)

    offsets = []
    total_bytes = 0
    for buf in buffers:
        nbytes = buf.cpu.element_size() * buf.cpu.nelement()
        offsets.append(total_bytes)
        total_bytes += -(-nbytes // arena_align) * arena_align  # round up

    arena_cpu = torch.zeros(total_bytes, dtype=torch.uint8, pin_memory=pin_memory)
    arena_gpu = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

    for buf, offset in zip(buffers, offsets):
        nbytes = buf.cpu.element_size() * buf.cpu.nelement()
        shape = tuple(buf.cpu.shape)
        dtype = buf.cpu.dtype

        cpu_view = arena_cpu[offset : offset + nbytes].view(dtype).view(shape)
        gpu_view = arena_gpu[offset : offset + nbytes].view(dtype).view(shape)
        cpu_view.copy_(buf.cpu)  # preserve whatever the original buffer held

        buf.cpu = cpu_view
        buf.gpu = gpu_view
        if hasattr(buf, "np"):
            buf.np = buf.cpu.numpy()

    return arena_cpu, arena_gpu


def _is_same_storage(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """True when copying src into dst is provably a no-op: same device,
    address, geometry and dtype means dst already holds src's bytes. The
    device check matters because data_ptr() isn't comparable across devices.

    Worth checking because static buffers usually alias vLLM's own
    already-refreshed device buffers, making the copy a no-op that still
    costs a real otter memcpy command. `dst is src` can't catch this --
    scheduler_output is deserialised fresh each step -- so storage must be
    compared directly.
    """
    return (
        dst.device == src.device
        and dst.data_ptr() == src.data_ptr()
        and dst.dtype == src.dtype
        and dst.shape == src.shape
        and dst.stride() == src.stride()
    )


## for static input and static forward input
@dataclass
class StaticForwardContext:
    attn_metadata: PerLayerAttnMetadata
    num_tokens_padded: int
    num_tokens_across_dp: torch.Tensor | None
    cudagraph_mode: CUDAGraphMode
    batch_desc: BatchDescriptor
    ubatch_slices_padded: UBatchSlices | None
    slot_mappings: tuple[dict[int, torch.Tensor] | None, dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None]
    has_encoder_input: bool

    def _copy_structure(self, dst, src, _path="StaticForwardContext"):
        if isinstance(src, torch.Tensor):
            # Most leaves here already alias vLLM's live device buffers; see _is_same_storage.
            if _is_same_storage(dst, src):
                return
            dst.copy_(src)
        elif isinstance(src, dict):
            for k in src:
                self._copy_structure(dst[k], src[k], f"{_path}.{k}")
        elif isinstance(src, (list, tuple)):
            for i in range(len(src)):
                self._copy_structure(dst[i], src[i], f"{_path}[{i}]")
        else:
            pass

    def update(self, src: "StaticForwardContext"):
        self._copy_structure(self.attn_metadata, src.attn_metadata)
        self.num_tokens_padded = src.num_tokens_padded
        self._copy_structure(self.num_tokens_across_dp, src.num_tokens_across_dp)
        # self.cudagraph_mode = src.cudagraph_mode
        # self.batch_desc = src.batch_desc
        self._copy_structure(self.ubatch_slices_padded, src.ubatch_slices_padded)
        # slot_mappings dropped: OtterNPUImpl.forward never reads it (dense KV-cache write
        # uses _slot_index instead), so this was a per-step tensor copy of dead data.
        # self._copy_structure(self.slot_mappings, src.slot_mappings)
        self.has_encoder_input = src.has_encoder_input

@dataclass
class StaticInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    intermediate_tensors: IntermediateTensors | None = None
    inputs_embeds: torch.Tensor | None = None
    model_kwargs: dict | None = None

    def _copy_structure(self, dst, src, _path="StaticInput"):
        if isinstance(src, torch.Tensor):
            # Same buffer-aliasing situation as StaticForwardContext; see _is_same_storage.
            if _is_same_storage(dst, src):
                return
            dst.copy_(src)
        elif isinstance(src, dict):
            for k in src:
                self._copy_structure(dst[k], src[k], f"{_path}.{k}")
        elif isinstance(src, (list, tuple)):
            for i in range(len(src)):
                self._copy_structure(dst[i], src[i], f"{_path}[{i}]")
        elif is_dataclass(src) and not isinstance(src, type):
            for f in fields(src):
                self._copy_structure(getattr(dst, f.name), getattr(src, f.name), f"{_path}.{f.name}")
        else:
            # scalars / enums / None are graph constants baked in at capture;
            # the old `dst = src` here was a no-op (rebinds a local), so skip.
            pass

    def update(self, src: "StaticInput"):
        self._copy_structure(self.input_ids ,src.input_ids)
        self._copy_structure(self.positions ,src.positions)
        self._copy_structure(self.intermediate_tensors ,src.intermediate_tensors)
        self._copy_structure(self.inputs_embeds ,src.inputs_embeds)
        self._copy_structure(self.model_kwargs ,src.model_kwargs)


# =========================================================================
# Empty Shells for Asynchronous Primitives
# Since Otter NPU does not yet implement torch.cuda.Event and torch.cuda.Stream,
# we provide empty shells.
# =========================================================================
class OtterDummyEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, stream=None) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class OtterDummyStream:
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device", None)

    def wait_event(self, event):
        pass

    def record_event(self, event=None):
        return OtterDummyEvent()

    def synchronize(self):
        pass

    def wait_stream(self, stream):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# =========================================================================
# Monkey-patch vLLM's dummy weight initialization
# =========================================================================
_original_initialize_single_dummy_weight = weight_utils.initialize_single_dummy_weight


def _patched_initialize_single_dummy_weight(param, low, high, seed):
    if param.device.type == "phison_npu":
        # cpu_tensor = torch.empty_like(param, device="cpu") 
        cpu_tensor = torch.empty(
            param.shape,
            dtype=param.dtype,
            device="cpu",
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        cpu_tensor.uniform_(low, high, generator=generator)
        # param.data.copy_(cpu_tensor.to("phison_npu"))
        param.data.copy_(cpu_tensor.to(param.device)) #e.g. phison_npu:1
    else:
        _original_initialize_single_dummy_weight(param, low, high, seed)


weight_utils.initialize_single_dummy_weight = _patched_initialize_single_dummy_weight


# =========================================================================
# Native Model Runner
# =========================================================================
def _mask_bias_dtype(cache_config, model_config) -> torch.dtype:
    """dtype of the additive attention-mask bias: must equal the KV cache dtype
    (the QK^T scores it is added to are in that dtype). Mirrors OtterNPUImpl's
    kv dtype resolution, including its fp32 fallback."""
    try:
        return _resolve_kv_dtype(cache_config.cache_dtype, model_config)
    except ValueError:
        raise
    except Exception:
        return torch.float32


class OtterModelRunner(GPUModelRunner):
    """
    Native GPUModelRunner inheritance for Otter NPU.
    Replaces unsupported backend calls with python shells dynamically.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.device = device
        device_index = self.device.index if self.device.index is not None else 0
        torch.phison_npu.set_device(device_index)
        logger.info(
            "OtterModelRunner current device set: device=%s",
            self.device,
        )

        # -----------------------------------------------------------------
        # [CRITICAL FIX]
        # GPUModelRunner.__init__ hardcodes calls to torch.cuda.Stream()
        # and torch.cuda.Event() before we have a chance to replace them.
        # Since this machine has no NVIDIA drivers, these calls will crash.
        # We temporarily patch torch.cuda to return our Dummy shells.
        # -----------------------------------------------------------------
        original_stream = torch.cuda.Stream
        original_event = torch.cuda.Event

        torch.cuda.Stream = OtterDummyStream
        torch.cuda.Event = OtterDummyEvent

        try:
            # Initialize the parent GPUModelRunner directly with our NPU device
            super().__init__(vllm_config, self.device)
            logger.info("OtterModelRunner initialized with device: %s", self.device)
        finally:
            # Restore the original PyTorch CUDA objects to prevent polluting global scope
            torch.cuda.Stream = original_stream
            torch.cuda.Event = original_event

        # Replace the generated instances internally as well
        self._replace_with_dummy_events_and_streams()

        # Merge these 4 buffers' per-step host->device refresh into one shared
        # arena / one device command (see _merge_cpu_gpu_buffers below).
        # input_ids also rides the arena when the sampler runs on the host
        # (tp>1, sharded vocab), saving one H2D per step. With an NPU sampler
        # (tp=1, patch_apply()) it stays out: the device-side refresh would
        # be overwritten by a stale host value.
        self._ids_in_arena = (
            self.parallel_config.tensor_parallel_size > 1
            and not PATCH_APPLIED
        )
        self._merged_prepare_bufs = [
            self.positions,
            self.query_start_loc,
            self.seq_lens,
            self.discard_request_mask,
        ]

        # Carve the decode KV-write index into the same merged arena, so its
        # per-step H2D rides the single arena copy_. The slot values are
        # computed into the arena's CPU side in _prepare_inputs, strictly
        # before that H2D. Head-major layout needs max_num_seqs*num_kv_heads
        # entries (one per (request, kv head) D-row).
        _slot_rows_per_req = _get_num_kv_heads(vllm_config.model_config)
        self._slot_index_rows_per_req = _slot_rows_per_req
        _slot_entries = self.scheduler_config.max_num_seqs * _slot_rows_per_req
        self._slot_index_shim = types.SimpleNamespace(
            cpu=torch.zeros(
                _slot_entries,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            ),
            # _prepare_inputs no-ops copy_to_gpu on every merged buffer;
            # the shim has no device buffer of its own to copy.
            copy_to_gpu=lambda n=None: None,
        )
        self._merged_prepare_bufs.append(self._slot_index_shim)
        if self._ids_in_arena:
            self._merged_prepare_bufs.append(self.input_ids)

        # Token -> embedding lookup on the host (tp>1); the NPU index_select
        # is slow. Decode rows go into an arena slot that the decode graph's
        # static inputs_embeds aliases, so they ride the arena H2D. Prefill
        # uses _host_embed_buf, copied in StaticInput.update. tp=1 keeps the
        # NPU embedding.
        self.host_embed = self._ids_in_arena
        self._decode_embed_shim = None
        if self.host_embed:
            self._decode_embed_shim = types.SimpleNamespace(
                cpu=torch.zeros(
                    (self.scheduler_config.max_num_seqs, self.model_config.get_hidden_size()),
                    dtype=self.model_config.dtype,
                    pin_memory=self.pin_memory,
                ),
                copy_to_gpu=lambda n=None: None,
            )
            self._merged_prepare_bufs.append(self._decode_embed_shim)
        # rows of _decode_embed_shim filled this step; None = none (prefill)
        self._decode_embeds_n: int | None = None

        # The decode additive mask rides the same arena H2D. Host fills the
        # pinned CPU side in _compute_decode_slot_index; the arena's GPU view
        # is the fixed buffer every layer's attention reads, so no device
        # kernel touches it per step.
        bias_dtype = _mask_bias_dtype(vllm_config.cache_config, vllm_config.model_config)
        _kv = resolve_max_len(vllm_config)
        self._decode_bias_shim = types.SimpleNamespace(
            cpu=torch.zeros(
                (self.scheduler_config.max_num_seqs, 1, 1, _kv),
                dtype=bias_dtype,
                pin_memory=self.pin_memory,
            ),
            copy_to_gpu=lambda n=None: None,
        )
        self._merged_prepare_bufs.append(self._decode_bias_shim)
        # row for context length n is pat[kv - n : 2*kv - n]: n zeros then -inf
        pat = torch.zeros(2 * _kv, dtype=bias_dtype)
        pat[_kv:] = float("-inf")
        self._decode_bias_pat_cpu = pat

        self._merged_arena_cpu, self._merged_arena_gpu = _merge_cpu_gpu_buffers(
            self._merged_prepare_bufs, self.device, self.pin_memory
        )

        self._slot_index_cpu = self._slot_index_shim.cpu
        self._slot_index_cpu_np = self._slot_index_cpu.numpy()
        self._slot_index_arena_gpu = self._slot_index_shim.gpu
        from .attention import OtterNPUImpl

        self._decode_bias_cpu = self._decode_bias_shim.cpu
        self._decode_bias = self._decode_bias_shim.gpu
        OtterNPUImpl._shared_decode_bias = self._decode_bias

        # Same key construction as attention.py:_alloc_cache:
        # cfg.device_config.device is torch.device(device_type) with NO
        # index, and torch.device('phison_npu') != torch.device('phison_npu:0')
        # as dict keys -- so drop the index here too.
        _dev_key = torch.device(
            self.device.type
            if isinstance(self.device, torch.device)
            else torch.device(self.device).type
        )
        OtterNPUImpl._shared_slot_index[(_dev_key, self.scheduler_config.max_num_seqs)] = (
            self._slot_index_arena_gpu
        )

        # tp_segmented_replay: tp>1 -> segmented capture/replay; tp=1 -> whole model.
        # otter_graph_enabled comes from OtterPlatform.check_and_update_config (enforce_eager).
        self.otter_graph_enabled = resolve_otter_graph_enabled(vllm_config)
        self.tp_segmented_replay = get_tensor_model_parallel_world_size() > 1 if self.otter_graph_enabled else False
        # otter_graph_enabled and not tp_segmented_replay (tp=1) -> whole-model
        # graph replay; eager mode (otter_graph_enabled False) uses neither.
        # Use graph 2 (prefill + dcode) * layers * 2 (all reduce)
        self.otter_graph_for_tp_decode = OtterGraphForTP(base_graph_id=10, communicator=get_tp_group().device_communicator) if self.tp_segmented_replay else None
        self.otter_graph_for_tp_prefill = OtterGraphForTP(base_graph_id=1000,communicator=get_tp_group().device_communicator) if self.tp_segmented_replay else None

        self.chunk_size = resolve_chunk_size(vllm_config)
        self.max_len = resolve_max_len(vllm_config)

        # graph 0/1 are preserved for whole-model prefill/decode replay,
        # graph 2 is preserved for compute_logits (lm_head) graph, and
        # graph 10~(10+2*layer) are for segmented decode replay
        self.prefill_graph_id = 0
        self.decode_graph_id = 1
        self.logits_graph_id = 2

        # static buffers for the captured lm_head graph.
        self.static_logits_input = None         # for prefill
        self.static_logits_local = None         # for prefill
        self.static_logits_local_decode = None  # for decode
        # num_reqs the decode graph (and its folded lm_head) was captured with
        self.captured_decode_num_reqs = None

        # Separate from the tp_segmented buffers on purpose: _sample's greedy fast path
        # tells the two apart with `logits is self.static_logits_output_decode`.
        self.static_logits_output_decode = None
        self.static_greedy_token_decode = None

        # Resolved on the first chunked prefill: every layer's
        # static_prefill_offset IS one shared arena
        # (attention.py::_SharedStaticOffsetArena), so the per-chunk masks are
        # written ONCE, not once per layer. The resolve asserts the sharing.
        self._prefill_offset = None

        # Persistent working set for the chunked-prefill replay loop,
        # allocated in _dummy_run BEFORE any graph capture 
        self._prefill_static: dict | None = None
        self._causal_bias_pat: torch.Tensor | None = None

        # Host embedding table / prefill buffer (tp>1).
        self._host_embed_weight: torch.Tensor | None = None
        self._host_embed_buf: torch.Tensor | None = None

    def _replace_with_dummy_events_and_streams(self):
        """
        Replaces any residual Event or Stream instances initialized by the parent.
        """
        for attr_name, attr_value in vars(self).items():
            obj_type_name = type(attr_value).__name__
            if 'Event' in obj_type_name:
                setattr(self, attr_name, OtterDummyEvent())
            elif 'Stream' in obj_type_name:
                setattr(self, attr_name, OtterDummyStream(device=self.device))

    def _decode_sample_view(self, model_output, num_reqs):
        """
        The rows to feed lm_head for a decode step, as a metadata-only view.

        Decode schedules exactly one token per request, packed in request order,
        so the first num_reqs rows ARE those tokens -> a plain SLICE, never a
        gather. The slice emits no NPU command, which is precisely what lets
        lm_head be captured inside the decode graph reading the decode body's
        own output buffer.
        """
        if self.use_aux_hidden_state_outputs:
            # True when EAGLE 3 is used: (hidden_states, aux_hidden_states).
            model_output = model_output[0]
        return model_output[:num_reqs]

    def _dummy_run(self, *args, **kwargs):
        pad_len = self.chunk_size
        # Chunk-loop working set BEFORE any capture: tensors allocated after
        # capture_end can share pages the replayed graph writes (recycled
        # captured intermediates) and get clobbered by replays.
        self._alloc_prefill_static()
        self.static_input_prefill = StaticInput(
            input_ids=torch.full((pad_len,), fill_value=50, dtype=torch.int32, device='phison_npu'),
            positions=torch.arange(pad_len, dtype=torch.int32, device='phison_npu'),
            intermediate_tensors=None,
            inputs_embeds=(
                torch.zeros(
                    (pad_len, self.model_config.get_hidden_size()),
                    dtype=self.model_config.dtype, device='phison_npu',
                ) if self.host_embed else None
            ),
            model_kwargs={},
        )
        self.static_input_prefill.input_ids[0] = 128000

        from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData, CachedRequestData
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            stop=[],
            stop_token_ids=[128008, 128001],
            bad_words=[],
            max_tokens=3,
        )


        # whole_model_prefill_pad_len 個 token 會跨多個 KV block（block_size 通常 16 → 128 需 8 個 block）。
        # 只給一個 block 會讓 vLLM 建 slot_mapping 時越界，所以依長度配足 block。
        block_size = self.cache_config.block_size or 16
        num_blocks_needed = (pad_len + block_size - 1) // block_size
        dummy_block_ids = tuple([list(range(1, num_blocks_needed + 1))])
        new_request_data = NewRequestData(
            req_id='0-00000000',
            prompt_token_ids=self.static_input_prefill.input_ids.to('cpu').tolist(),
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=dummy_block_ids,
            num_computed_tokens=0,
            lora_request=None,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[new_request_data],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={'0-00000000': pad_len},
            total_num_scheduled_tokens=pad_len,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=set(),
        )

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(
                scheduler_output.preempted_req_ids
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            # self._update_states(scheduler_output)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = 1
            req_ids = ['0-00000000']
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices = torch.tensor([pad_len-1], device='phison_npu:0', dtype=torch.int32)

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = False
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # With spec decode, defer connector finalization until after the draft model.
        defer_kv_connector_finalize = self.speculative_config is not None

        self.static_forward_context_prefill = StaticForwardContext(
            attn_metadata=attn_metadata,
            num_tokens_padded=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_mode=cudagraph_mode,
            batch_desc=batch_desc,
            ubatch_slices_padded=ubatch_slices_padded,
            slot_mappings=slot_mappings,
            has_encoder_input=has_encoder_input,
        )

        # Alias onto vLLM's arena slice so StaticInput.update's D2D copy short-circuits
        # (positions is only used as an index operand, so int64 is fine).
        _decode_positions = self.positions.gpu[:1]
        # Same for input_ids: the async scatter writes this buffer before replay().
        _decode_input_ids = self.input_ids.gpu[:1]
        self.static_input_decode = StaticInput(
            input_ids=_decode_input_ids,
            positions=_decode_positions,
            intermediate_tensors=None,
            # Aliases the merged arena: refreshed by the single arena H2D.
            inputs_embeds=(
                self._decode_embed_shim.gpu[:1] if self.host_embed else None
            ),
            model_kwargs={},
        )
        self.static_input_decode.input_ids[0] = 128000

        decode_req = NewRequestData(
            req_id='0-00000001',
            prompt_token_ids=self.static_input_decode.input_ids.to('cpu').tolist(),
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        scheduler_output_d = SchedulerOutput(
            scheduled_new_reqs=[decode_req],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={'0-00000001': 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=set(),
        )

        num_reqs_d, num_tokens_d = 1, 1
        self.captured_decode_num_reqs = num_reqs_d
        logits_indices_d = torch.tensor([0], device='phison_npu:0', dtype=torch.int32)

        (
            cudagraph_mode_d,
            batch_desc_d,
            _,
            num_tokens_across_dp_d,
            _
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_d,
            num_reqs=num_reqs_d,
            num_scheduled_tokens_np=np.array([1], dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            num_encoder_reqs=0,
        )

        slot_mappings_by_group_d, slot_mappings_d = self._get_slot_mappings(
            num_tokens_padded=num_tokens_d, num_reqs_padded=num_reqs_d,
            num_tokens_unpadded=num_tokens_d, ubatch_slices=None,
        )

        attn_metadata_d, _ = (
            self._build_attention_metadata(
                num_tokens=num_tokens_d, num_reqs=num_reqs_d, max_query_len=1,
                ubatch_slices=None, logits_indices=logits_indices_d, use_spec_decode=False,
                num_scheduled_tokens=scheduler_output_d.num_scheduled_tokens,
                cascade_attn_prefix_lens=None, slot_mappings=slot_mappings_by_group_d,
            )
        )

        self.static_forward_context_decode = StaticForwardContext(
            attn_metadata=attn_metadata_d, num_tokens_padded=batch_desc_d.num_tokens,
            num_tokens_across_dp=num_tokens_across_dp_d, cudagraph_mode=cudagraph_mode_d,
            batch_desc=batch_desc_d, ubatch_slices_padded=None,
            slot_mappings=slot_mappings_d, has_encoder_input=False,
        )

        def _prefill_forward():
            with (
                set_forward_context(
                    self.static_forward_context_prefill.attn_metadata,
                    self.vllm_config,
                    num_tokens=self.static_forward_context_prefill.num_tokens_padded,
                    num_tokens_across_dp=self.static_forward_context_prefill.num_tokens_across_dp,
                    cudagraph_runtime_mode=self.static_forward_context_prefill.cudagraph_mode,
                    batch_descriptor=self.static_forward_context_prefill.batch_desc,
                    ubatch_slices=self.static_forward_context_prefill.ubatch_slices_padded,
                    slot_mapping=self.static_forward_context_prefill.slot_mappings,
                    skip_compiled=self.static_forward_context_prefill.has_encoder_input,
                ),
                record_function_or_nullcontext("gpu_model_runner: forward"),
                self.maybe_get_kv_connector_output(
                    scheduler_output,
                    defer_finalize=defer_kv_connector_finalize,
                ) as self.kv_connector_output,
            ):
                return self._model_forward(
                    input_ids=self.static_input_prefill.input_ids,
                    positions=self.static_input_prefill.positions,
                    intermediate_tensors=self.static_input_prefill.intermediate_tensors,
                    inputs_embeds=self.static_input_prefill.inputs_embeds,
                    **self.static_input_prefill.model_kwargs,
                )

        def _decode_forward():
            with set_forward_context(
                self.static_forward_context_decode.attn_metadata, self.vllm_config,
                num_tokens=batch_desc_d.num_tokens, num_tokens_across_dp=num_tokens_across_dp_d,
                cudagraph_runtime_mode=cudagraph_mode_d, batch_descriptor=batch_desc_d,
                ubatch_slices=None, slot_mapping=slot_mappings_d, skip_compiled=False,
            ):
                return self._model_forward(
                    input_ids=self.static_input_decode.input_ids,
                    positions=self.static_input_decode.positions,
                    intermediate_tensors=None,
                    inputs_embeds=self.static_input_decode.inputs_embeds,
                )

        def _tp_graph_preprocess(otter_graph_for_tp : OtterGraphForTP):
            otter_graph_for_tp.set_mode(OtterGraphForTP.CAPTURE)
            otter_graph_for_tp.begin_forward()

        def _tp_graph_postprocess(otter_graph_for_tp : OtterGraphForTP, name: str):
            otter_graph_for_tp.end_forward()
            logger.info(
                "Otter segmented %s replay: captured %d boundaries -> graphs %s",
                name,
                otter_graph_for_tp.segments.num_boundaries(),
                otter_graph_for_tp.segments.graph_ids(),
            )
            otter_graph_for_tp.set_mode(OtterGraphForTP.OFF)

        # capture decode
        if self.tp_segmented_replay:
            # Same consts the tp=1 branch below registers once; here they must
            # be re-registered per segment, so hand them to the segmenter.
            self.otter_graph_for_tp_decode.capture_consts = (
                [self.static_input_decode.positions]
                + self._kv_index_consts("_slot_index")
                + self._kv_index_consts("_positions")
            )
            _tp_graph_preprocess(self.otter_graph_for_tp_decode)

        elif self.otter_graph_enabled:
            otter_torch.capture_begin(self.decode_graph_id)
            # Index operands (rotary index_select, KV index_copy_, masked_fill_ positions)
            # must be replay consts, otherwise each layer is a LOCAL_PATCH cmd that
            # can't be packed (one descriptor send + interrupt per layer).
            otter_torch.capture_const(self.static_input_decode.positions)
            self._capture_const_kv_indices("_slot_index")
            self._capture_const_kv_indices("_positions")

        with use_tp(self.otter_graph_for_tp_decode):
            self.model_output_decode = _decode_forward()

        if self.tp_segmented_replay:
            # last RMS + lm head in a single graph
            lm_head = self.model.lm_head
            self.static_logits_local_decode = lm_head.quant_method.apply(
                lm_head, self._decode_sample_view(self.model_output_decode, num_reqs_d), bias=None
            )
            _tp_graph_postprocess(self.otter_graph_for_tp_decode, "decode")

        elif self.otter_graph_enabled:
            # Fold compute_logits into the same decode graph (tp=1 only).
            # Dedicated attribute (not the static_logits_local the tp_segmented
            # path uses): _sample's fast path pins itself to this replay via
            # `logits is self.static_logits_output_decode`.
            self.static_logits_output_decode = self.model.compute_logits(
                self._decode_sample_view(self.model_output_decode, num_reqs_d)
            )
            # int32 to match downstream consumers (GPUModelRunner.input_ids
            # is int32); int64 forces an implicit cast wherever it meets an
            # int32 site (e.g. async _prepare_input_ids's scatter).
            self.static_greedy_token_decode = self.static_logits_output_decode.argmax(dim=-1).to(torch.int32)
            otter_torch.capture_end()

        # capture prefill
        if self.tp_segmented_replay:
            self.otter_graph_for_tp_prefill.capture_consts = [
                self.static_input_prefill.positions
            ]
            _tp_graph_preprocess(self.otter_graph_for_tp_prefill)

        elif self.otter_graph_enabled:
            otter_torch.capture_begin(self.prefill_graph_id)
            otter_torch.capture_const(self.static_input_prefill.positions)
        
        if not self.otter_graph_enabled:
            for _md in self.static_forward_context_prefill.attn_metadata.values():
                _md.chunk_start = 0
                _md.chunk_len = pad_len

        with use_tp(self.otter_graph_for_tp_prefill):
            self.model_output_prefill = _prefill_forward()

        if self.tp_segmented_replay:
            _tp_graph_postprocess(self.otter_graph_for_tp_prefill, "prefill")

        elif self.otter_graph_enabled:
            otter_torch.capture_end()

        # for prefill lm_head
        if self.tp_segmented_replay:
            self.static_logits_input = torch.empty_like(self._decode_sample_view(self.model_output_decode, num_reqs_d))
            lm_head = self.model.lm_head
            otter_torch.capture_begin(self.logits_graph_id)
            # bias=None mirrors LogitsProcessor.forward's own call (no model
            # here passes embedding_bias). Replicated vocab -> this IS the full
            # row; sharded -> it is this rank's vocab/tp slice.
            self.static_logits_local = lm_head.quant_method.apply(
                lm_head, self.static_logits_input, bias=None
            )
            otter_torch.capture_end()

        # 清掉 warm-up / capture 期間寫入 dense KV cache 的殘值
        # 類似 otter/misc/capture_replay/demo_replay_llama3.py 的 reset_cache() 用 zero_() 去重置
        for _, module in self.model.named_modules():
            impl = getattr(module, "impl", None)
            if impl is not None and getattr(impl, "_key_cache", None) is not None:
                impl._key_cache.zero_()
                impl._value_cache.zero_()

    def _kv_index_consts(self, attr: str) -> list:
        """Every distinct persistent KV index buffer named `attr` across the
        attention layers (shared buffers appear once)."""
        out, seen = [], set()
        for _, module in self.model.named_modules():
            index = getattr(getattr(module, "impl", None), attr, None)
            if index is not None and index.data_ptr() not in seen:
                seen.add(index.data_ptr())
                out.append(index)
        return out

    def _logits_tail(self, logits_local: torch.Tensor) -> torch.Tensor:
        """Everything LogitsProcessor does AFTER the lm_head matmul.

        A replayed graph leaves this rank's share in `logits_local`
        (static_logits_local for prefill, static_logits_local_decode for
        decode); run the un-capturable gather and the elementwise
        tail here, in LogitsProcessor.forward's order. With a replicated vocab
        _gather_logits is the patched identity, so this is the same code for
        both modes.
        """
        lp = self.model.logits_processor
        logits = lp._gather_logits(logits_local)
        logits = logits[..., : lp.org_vocab_size]
        if lp.soft_cap is not None:
            logits = torch.tanh(logits / lp.soft_cap) * lp.soft_cap
        if lp.scale != 1.0:
            logits = logits * lp.scale
        return logits

    def _capture_const_kv_indices(self, attr: str) -> None:
        """
        Mark every attention layer's persistent KV-write index buffer as a
        replay const for the graph currently being captured.

        Must be called between capture_begin and the forward. The dense KV
        cache writes with index_copy_ (attention.py), and Capturer::end_cmd
        decides ID_INDEX_COPY's patch mode by looking the index tensor up in
        the graph's replay-const set: unregistered -> LOCAL_PATCH, which
        Capturer::end refuses to pack, so all 64 commands per step (2 per
        layer) pay their own descriptor send plus interrupt.
        """
        for index in self._kv_index_consts(attr):
            otter_torch.capture_const(index)

    def _update_decode_slot_index(self, num_reqs: int) -> None:
        """
        Host companion to write_decode_kv_index_copy's `_slot_index` buffer:
        computes the flat index of request r's new token into the dense KV
        cache and copies it into the buffer BEFORE replay() (host-side, or
        the index is one step stale).
        """
        for _, module in self.model.named_modules():
            dst = getattr(getattr(module, "impl", None), "_slot_index", None)
            if dst is not None:
                if dst.data_ptr() != self._slot_index_arena_gpu.data_ptr():
                    # Not our arena slice (attention.py did not adopt it,
                    # e.g. _shared_slot_index key mismatch): refresh it the
                    # legacy way instead of letting replay read stale data.
                    self._write_slot_index_cpu(num_reqs)
                    src = self._slot_index_cpu[: num_reqs * self._slot_index_rows_per_req]
                    dst[: src.shape[0]].copy_(src, non_blocking=True)
                break

    def _compute_decode_slot_index(self, num_reqs: int) -> None:
        """Compute slot indices into the arena's CPU side only (no device
        command). Called from _prepare_inputs after the native call refreshed
        seq_lens.np, before the merged arena H2D is issued."""
        self._write_slot_index_cpu(num_reqs)
        self._write_decode_bias_cpu(num_reqs)

    def _write_decode_bias_cpu(self, num_reqs: int) -> None:
        """Host fill of the decode additive mask (0 where position < seq_len,
        -inf elsewhere): a memcpy from the static pattern per request, picked
        up by the same arena H2D as the slot index."""
        kv = self._decode_bias_pat_cpu.shape[0] // 2
        for r, n in enumerate(self.seq_lens.np[:num_reqs].tolist()):
            n = min(max(n, 0), kv)
            self._decode_bias_cpu[r].copy_(self._decode_bias_pat_cpu[kv - n: 2 * kv - n].view(1, 1, kv))

    def _write_slot_index_cpu(self, num_reqs: int) -> None:
        """Fill _slot_index_cpu_np with the flat index each decode KV write
        targets. Head-major: R*Hkv entries in (r-major, h-inner) order,
        entry (r, h) = (r * Hkv + h) * max_len + pos -- one D-row per
        entry, matching write_decode_kv_index_copy's [S*Hkv*L, 1, D] flat
        view."""
        max_len = self.model_config.max_model_len
        pos = self.seq_lens.np[:num_reqs].astype(np.int64) - 1
        num_kv_heads = _get_num_kv_heads(self.model_config)
        r_term = (np.arange(num_reqs, dtype=np.int64) * num_kv_heads * max_len)[:, None]
        h_term = np.arange(num_kv_heads, dtype=np.int64)[None, :] * max_len
        self._slot_index_cpu_np[: num_reqs * num_kv_heads] = (
            (r_term + h_term + pos[:, None]).reshape(-1)
        )

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        if not self.host_embed:
            return
        embed = self.model.model.embed_tokens
        # Full table from the checkpoint; fall back to gathering the shards.
        full = self._load_host_embed_from_checkpoint(embed.weight.dtype)
        if full is None:
            full = self._gather_host_embed(embed)
        self._host_embed_weight = full[: embed.org_vocab_size].contiguous()
        self._check_host_embed_against_shard(embed)
        logger.info(
            "Host embedding enabled: table %s %s kept in host RAM",
            tuple(self._host_embed_weight.shape), self._host_embed_weight.dtype,
        )

    _HOST_EMBED_CKPT_NAME = "model.embed_tokens.weight"

    def _load_host_embed_from_checkpoint(self, dtype: torch.dtype) -> torch.Tensor | None:
        """Read just the embedding tensor from the local safetensors
        checkpoint; None if not found."""
        import glob
        import json

        from safetensors import safe_open

        path = self.model_config.model
        name = self._HOST_EMBED_CKPT_NAME
        if not os.path.isdir(path):
            return None
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as f:
                fname = json.load(f).get("weight_map", {}).get(name)
            files = [os.path.join(path, fname)] if fname else []
        else:
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        for fpath in files:
            with safe_open(fpath, framework="pt", device="cpu") as sf:
                if name in sf.keys():
                    logger.info("Host embedding: loading %s from %s", name, fpath)
                    return sf.get_tensor(name).to(dtype)
        return None

    def _gather_host_embed(self, embed) -> torch.Tensor:
        """Fallback: rebuild the full table from every rank's NPU shard
        (valid rows sit at the top of each padded shard, in rank order)."""
        logger.info("Host embedding: checkpoint not found, gathering shards")
        tp = gpu_mr_mod.get_tp_group()
        shard = embed.weight.detach().to("cpu").contiguous()
        si = embed.shard_indices
        n_valid = si.org_vocab_end_index - si.org_vocab_start_index
        shards = [torch.empty_like(shard) for _ in range(tp.world_size)]
        dist.all_gather(shards, shard, group=tp.cpu_group)
        valid = [None] * tp.world_size
        dist.all_gather_object(valid, n_valid, group=tp.cpu_group)
        return torch.cat([s[:v] for s, v in zip(shards, valid)])

    def _check_host_embed_against_shard(self, embed) -> None:
        """Check the host table against this rank's NPU shard (first / last
        valid row)."""
        si = embed.shard_indices
        lo, hi = si.org_vocab_start_index, si.org_vocab_end_index
        w = self._host_embed_weight
        assert w.shape[0] == embed.org_vocab_size, (tuple(w.shape), embed.org_vocab_size)
        for local, glob_row in ((0, lo), (hi - lo - 1, hi - 1)):
            npu_row = embed.weight[local].detach().cpu()
            assert torch.equal(npu_row, w[glob_row]), (
                f"host embed row {glob_row} != NPU shard row {local}"
            )

    def _host_embed_lookup(self, num_tokens: int, out: torch.Tensor | None = None) -> torch.Tensor:
        """Embed self.input_ids.cpu[:num_tokens] into `out` (default: the
        persistent prefill host buffer) and return it."""
        w = self._host_embed_weight
        if out is None and self._host_embed_buf is None:
            rows = (
                self.scheduler_config.max_num_batched_tokens
                + getattr(self, "chunk_size", 0)
            )
            self._host_embed_buf = torch.empty((rows, w.shape[1]), dtype=w.dtype)
        if out is None:
            out = self._host_embed_buf[:num_tokens]
        ids = self.input_ids.cpu[:num_tokens].long()
        torch.index_select(w, 0, ids, out=out)
        return out

    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        (
            input_ids, inputs_embeds, positions,
            intermediate_tensors, model_kwargs, ec_connector_output,
        ) = super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)
        if self.host_embed and inputs_embeds is None:
            # input_ids is kept: callers use it for shapes / routing.
            n = self._decode_embeds_n
            if n is not None and num_input_tokens <= n:
                # decode: looked up in _prepare_inputs, already on the NPU
                inputs_embeds = self._decode_embed_shim.gpu[:num_input_tokens]
            else:
                inputs_embeds = self._host_embed_lookup(num_input_tokens)
        return (
            input_ids, inputs_embeds, positions,
            intermediate_tensors, model_kwargs, ec_connector_output,
        )

    def profile_run(self, *args, **kwargs):
        """Shell: Skip slow profiling entirely."""
        pass

    def capture_model(self, kv_caches):
        """Shell: Bypass CUDA Graph capture."""
        pass

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        """
        Calls native GPUModelRunner._prepare_inputs, but collapses 4 of its
        unconditional per-step host->device transfers -- positions,
        query_start_loc, seq_lens, discard_request_mask -- into ONE device
        command via the shared arena built in __init__
        (_merge_cpu_gpu_buffers). Their own copy_to_gpu() calls are no-oped
        for the duration of the native call below (the writes into .np/.cpu
        still happen exactly as before, just landing in the shared arena's
        memory), then flushed with a single
        arena_gpu.copy_(arena_cpu, non_blocking=True).

        input_ids rides the arena too when self._ids_in_arena; otherwise it
        keeps its own copy_to_gpu (see __init__).
        """
        merged_bufs = self._merged_prepare_bufs
        orig_copy_to_gpu = [buf.copy_to_gpu for buf in merged_bufs]
        for buf in merged_bufs:
            buf.copy_to_gpu = lambda n=None: None

        # No-op two more H2D copies native _prepare_inputs issues
        # unconditionally: slot_mapping / block_table. Both feed vLLM's
        # paged KV-cache path, which this dense-cache backend replaces --
        # OtterNPUImpl.forward never reads OtterMetadata's slot_mapping/
        # block_tables fields. Only the device copy is skipped; .np/.cpu
        # writes still happen inside super()._prepare_inputs.
        block_tables = self.input_batch.block_table.block_tables
        orig_bt_copy = [bt.block_table.copy_to_gpu for bt in block_tables]
        orig_sm_copy = [bt.slot_mapping.copy_to_gpu for bt in block_tables]
        for bt in block_tables:
            bt.block_table.copy_to_gpu = lambda n=None: None
            bt.slot_mapping.copy_to_gpu = lambda n=None: None
        orig_ids_copy = self.input_ids.copy_to_gpu
        try:
            result = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
            # seq_lens.np was just refreshed by the native call and is not
            # written again until the next _prepare_inputs, so the slot-index
            # compute can move here -- strictly before the merged H2D it
            # must ride.
            self._compute_decode_slot_index(len(num_scheduled_tokens))
            self._decode_embeds_n = None
            if self.host_embed:
                n = int(num_scheduled_tokens.sum())
                # decode step: one token per request -> fits the arena slot
                if n == len(num_scheduled_tokens) and n <= self._decode_embed_shim.cpu.shape[0]:
                    self._host_embed_lookup(n, out=self._decode_embed_shim.cpu[:n])
                    self._decode_embeds_n = n
            self._merged_arena_gpu.copy_(self._merged_arena_cpu, non_blocking=True)
            return result
        finally:
            # restore input_ids first (its copy_to_gpu is no-oped above)
            self.input_ids.copy_to_gpu = orig_ids_copy
            for buf, orig in zip(merged_bufs, orig_copy_to_gpu):
                buf.copy_to_gpu = orig
            for bt, orig_bt, orig_sm in zip(block_tables, orig_bt_copy, orig_sm_copy):
                bt.block_table.copy_to_gpu = orig_bt
                bt.slot_mapping.copy_to_gpu = orig_sm

    def _prepare_input_ids(self, scheduler_output, total_num_scheduled_tokens, cu_num_tokens):
        """With input_ids in the arena and async scheduling on, write the
        previous step's sampled ids (a host tensor) into input_ids.cpu so the
        arena H2D carries them, instead of vLLM's extra copy_/scatter into
        input_ids.gpu. Every other case is vLLM's own code."""
        prev = self.input_batch.prev_sampled_token_ids
        if not self._ids_in_arena or prev is None:
            return super()._prepare_input_ids(
                scheduler_output, total_num_scheduled_tokens, cu_num_tokens
            )
        if prev.device.type != "cpu":
            raise RuntimeError(
                "input_ids is in the merged arena but the sampled token ids "
                f"are on {prev.device}: the arena H2D would overwrite them."
            )
        assert not scheduler_output.scheduled_spec_decode_tokens, (
            "spec decode is not supported with input_ids in the merged arena"
        )
        assert not self.enable_prompt_embeds
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        ids_cpu = self.input_ids.cpu
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            prev_index = prev_req_id_to_index.get(req_id)
            if prev_index is not None:
                # last scheduled token of this request = last step's sample
                ids_cpu[int(cu_num_tokens[cur_index]) - 1] = prev[prev_index, 0]

    def prepare_inputs(self, scheduler_output, batch_desc):
        """
        Calls the native GPUModelRunner.prepare_inputs, but ensures that the resulting
        input_ids and positions tensors are physically moved to the NPU.
        """
        device_index = self.device.index if self.device.index is not None else 0
        torch.phison_npu.set_device(device_index)

        input_batch = super().prepare_inputs(scheduler_output, batch_desc)

        # Move the newly constructed input tensors to the NPU
        if input_batch.input_ids is not None and getattr(input_batch.input_ids, "device", None) != self.device:
            input_batch.input_ids = input_batch.input_ids.to(self.device)

        if input_batch.positions is not None and getattr(input_batch.positions, "device", None) != self.device:
            input_batch.positions = input_batch.positions.to(self.device)

        return input_batch


    def _sample(self, logits, spec_decode_metadata):
        """
        Greedy fast path for whole-model decode replay: when logits is the
        folded decode-graph output and sampling is plain greedy (no spec
        decode, no penalties, no logprobs), the argmax was already computed
        inside the replay -- return it directly instead of dispatching
        sampler.py's eager argmax + casts.

        The `is` check is load-bearing: on the eager path both attrs are
        None so it short-circuits, and on prefill/tp_segmented_replay
        logits is a different object, so this always falls through to
        super()._sample. Mirrored in test_sample_greedy_guard.py.
        """
        sampling_metadata = self.input_batch.sampling_metadata
        if (
            self.static_greedy_token_decode is not None
            and logits is self.static_logits_output_decode
            and spec_decode_metadata is None
            and sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
        ):
            return gpu_mr_mod.SamplerOutput(
                sampled_token_ids=self.static_greedy_token_decode.unsqueeze(-1),
                logprobs_tensors=None,
            )
        return super()._sample(logits, spec_decode_metadata)

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # TP>1 samples on host so skip the base class's pinned staging copy + event sync.
        if sampled_token_ids.device.type == "cpu":
            return sampled_token_ids.tolist()
        return super()._to_list(sampled_token_ids)

    def _alloc_prefill_static(self) -> None:
        """Allocate the chunked-prefill replay loop's persistent working set.

        Must run before any otter graph capture (called from _dummy_run):
        after capture_end the allocator recycles the graph's pages, so
        tensors allocated later can alias replay-written pages and get
        clobbered. Holds everything the chunk loop keeps alive across
        replay(), sized to the max schedule, refreshed in place only.
        """
        cs = self.chunk_size
        kv = self.max_len
        dev = self.device
        # Scheduler never schedules more than this many tokens per step;
        # +cs keeps the staging buffers valid for the padded pad_len.
        max_sched = int(self.scheduler_config.max_num_batched_tokens) + cs
        kv_pos = torch.arange(kv, device=dev, dtype=torch.int32).view(kv, 1)
        self._prefill_static = {
            "ids": torch.zeros(max_sched, dtype=torch.int32, device=dev),
            "pos": torch.zeros(max_sched, dtype=torch.int64, device=dev),
            # kv_pos variants: one arange(max_len) storage, pre-shaped views
            # for the broadcast masks (onehot/valid) below.
            "kv_pos": kv_pos,
            "kv_pos_3d": kv_pos.view(kv, 1, 1),
        }

        # Prefill causal pattern, allocated before capture so it stays put.
        # Static 0/-inf, built once on the host: every per-step prefill mask is
        # a slice (view or one copy_) of it -- no masked_fill / compare on the
        # step path. (The decode mask is host-filled into the merged arena,
        # see __init__.)
        #   causal[q, c] = 0 iff c - kv <= q, so the mask of a chunk starting
        #   at `start` is causal[:, kv-start : 2*kv-start] (key k <= start + q
        #   allowed).
        from .attention import OtterNPUImpl

        bias_dtype = _mask_bias_dtype(self.cache_config, self.model_config)
        causal = torch.triu(
            torch.full((cs, 2 * kv), float("-inf"), dtype=bias_dtype), diagonal=kv + 1
        )
        self._causal_bias_pat = causal.to(dev)
        OtterNPUImpl._shared_causal_bias = self._causal_bias_pat

    @staticmethod
    def _pad_slot_structure(obj, pad_len):
        """把 slot_mappings (tuple/dict 內的張量) 沿 dim0 padding 到 pad_len。
        填入的值是 -1。

        現在 attention.py 的寫法沒有用到 slot_mapping，想像上忽略 slot_mappings 應該也能動。
        但以後有可能會再動到 attention.py 和 ops.py 的架構，
        所以這裡還是先沿用，把 slot_mappings 的大小調整到 static_forward_context_prefill 能夠正常 update 的長度。
        至於 padding 的值，填入的是 -1 (https://docs.vllm.ai/en/v0.11.2/api/vllm/v1/attention/backends/utils/ 的 PAD_SLOT_ID=-1)。
        """
        if isinstance(obj, torch.Tensor):
            if obj.shape[0] >= pad_len:
                return obj
            padded = obj.new_full((pad_len,) + tuple(obj.shape[1:]), -1)
            padded[:obj.shape[0]] = obj
            return padded
        if isinstance(obj, dict):
            return {k: OtterModelRunner._pad_slot_structure(v, pad_len) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(OtterModelRunner._pad_slot_structure(v, pad_len) for v in obj)
        return obj


    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.routed_experts_initialized:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(
                scheduler_output.preempted_req_ids
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            self._update_states(scheduler_output)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None

        # Route by attn_state (the shape heuristic mislabels multi-request decode as
        # prefill); fall back to the shape check only if it cannot be read.
        _am = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
        _meta = next(iter(_am.values()), None) if isinstance(_am, dict) else _am
        _attn_state = getattr(_meta, "attn_state", None)
        is_prefill = (
            _attn_state == OtterAttentionState.PREFILL
            if _attn_state is not None
            else input_ids.shape[0] != 1
        )

        # Phase: Prefill
        if is_prefill:
            # Pad input_ids/positions/slot_mappings up to a multiple of chunk_size so
            # every chunk has a fixed shape. seq_lens/query_start_loc are NOT padded:
            # the ops mask with the real length, passed in via the offset masks below.
            real_len = input_ids.shape[0]
            from vllm.forward_context import BatchDescriptor
            total_chunks = (real_len + self.chunk_size - 1) // self.chunk_size
            pad_len = total_chunks * self.chunk_size

            # Persistent chunk-loop working set: refresh in place.
            if self._prefill_static is None:
                self._alloc_prefill_static()
            st = self._prefill_static
            assert st is not None
            kv_pos = st["kv_pos"]
            kv_pos_3d = st["kv_pos_3d"]
            # pad (always stage through the persistent buffers so the
            # chunk loop below always slices the same stable allocation)
            st["ids"][:real_len].copy_(input_ids)
            input_ids = st["ids"][:pad_len]
            if inputs_embeds is not None:
                # zero the padding rows (stale tail of the host buffer)
                inputs_embeds = self._host_embed_buf[:pad_len]
                inputs_embeds[real_len:].zero_()
                if not self.otter_graph_enabled:
                    inputs_embeds = inputs_embeds.to(self.device)
            st["pos"][:real_len].copy_(positions)
            positions = st["pos"][:pad_len]
            if pad_len > real_len:
                slot_mappings = self._pad_slot_structure(slot_mappings, pad_len)

            # write_chunk_prefill_kv accumulates via add_, so slot 0 must start
            # clean per request or request N+1 sums onto request N's K/V
            for _, module in self.model.named_modules():
                impl = getattr(module, "impl", None)
                if impl is not None and getattr(impl, "_key_cache", None) is not None:
                    impl._key_cache[0].zero_()
                    impl._value_cache[0].zero_()
            for chunk_id in range(total_chunks):
                chunk_real_token_len = min(
                    self.chunk_size,
                    real_len - chunk_id * self.chunk_size
                )
                start = chunk_id * self.chunk_size
                end = (chunk_id + 1)* self.chunk_size
                
                chunk_num_tokens_unpadded = chunk_real_token_len
                chunk_max_num_scheduled_tokens = chunk_real_token_len
                chunk_batch_desc = BatchDescriptor(chunk_real_token_len)
                # Always chunk_size: the offset masks are chunk_size wide, and with
                # pad_attn False vLLM would truncate key/value to the real length.
                chunk_num_tokens_padded = self.chunk_size
                chunk_slot_mapping = {}
                chunk_group_slot_mapping = {}
                for k in slot_mappings:
                    chunk_slot_mapping[k] = slot_mappings[k][start: end]
                for k in slot_mappings_by_group:
                    chunk_group_slot_mapping[k] = slot_mappings_by_group[k][start: start + chunk_real_token_len]

                chunk_attn_metadata, chunk_spec_decode_common_attn_metadata = (
                    self._build_attention_metadata(
                        num_tokens=chunk_num_tokens_unpadded,
                        num_tokens_padded=chunk_num_tokens_padded,
                        num_reqs=num_reqs,
                        num_reqs_padded=num_reqs_padded if pad_attn else None,
                        max_query_len=chunk_max_num_scheduled_tokens,
                        ubatch_slices=ubatch_slices_attn,
                        logits_indices=logits_indices,
                        use_spec_decode=use_spec_decode,
                        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                        slot_mappings=chunk_group_slot_mapping,
                    )
                )
                for layer in attn_metadata:
                    chunk_attn_metadata[layer].max_actual_tokens = chunk_real_token_len
                    chunk_attn_metadata[layer].query_start_loc[1] = chunk_real_token_len
                    chunk_attn_metadata[layer].seq_lens[0] = chunk_real_token_len
                    chunk_attn_metadata[layer].slot_mapping = chunk_slot_mapping

                if self.otter_graph_enabled:
                    # update static context
                    next_forward_context_prefill = StaticForwardContext(
                        attn_metadata=chunk_attn_metadata,
                        num_tokens_padded=chunk_num_tokens_padded,
                        num_tokens_across_dp=num_tokens_across_dp,
                        cudagraph_mode=cudagraph_mode,
                        batch_desc=chunk_batch_desc,
                        ubatch_slices_padded=ubatch_slices_padded,
                        slot_mappings=chunk_slot_mapping,
                        has_encoder_input=has_encoder_input,
                    )
                    next_input_prefill = StaticInput(
                        input_ids=input_ids[start:end],
                        positions=positions[start:end],
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=(
                            inputs_embeds[start:end] if inputs_embeds is not None else None
                        ),
                        model_kwargs=model_kwargs
                    )
                    self.static_forward_context_prefill.update(next_forward_context_prefill)
                    self.static_input_prefill.update(next_input_prefill)
                if self.otter_graph_enabled:
                    # update write kv offset.
                    s = torch.arange(start, end, device=self.device, dtype=torch.int32).view(1, self.chunk_size)
                    n = chunk_real_token_len + start
                    e = start + chunk_attn_metadata[f'model.layers.{0}.self_attn.attn'].seq_lens
                    valid = ((kv_pos_3d >= ((n - 1) // self.chunk_size) * self.chunk_size) & (kv_pos_3d < n))
                    onehot = ((kv_pos == s) & (s < n)).to(torch.float32)
                    # causal slice of the static pattern; no `kv_pos >= e` term (it only
                    # matters for padded query rows, whose outputs are discarded and
                    # whose extra keys are zero-filled cache).
                    attn_bias = self._causal_bias_pat[:, self.max_len - start: 2 * self.max_len - start].view(
                        1, 1, self.chunk_size, self.max_len)

                    if self._prefill_offset is None:
                        layers = self.model.model.layers
                        self._prefill_offset = layers[0].self_attn.attn.impl.static_prefill_offset
                        assert all(
                            l.self_attn.attn.impl.static_prefill_offset is self._prefill_offset
                            for l in layers
                        ), (
                            "static_prefill_offset is no longer one shared arena -- "
                            "per-layer buffers need the per-layer write back"
                        )

                    self._prefill_offset.update(
                        seq_lens= e,
                        valid= valid,
                        onehot= onehot,
                        attn_bias = attn_bias,
                    )
                else:
                    for layer in chunk_attn_metadata:
                        chunk_attn_metadata[layer].chunk_start = start
                        chunk_attn_metadata[layer].chunk_len = chunk_real_token_len
                if self.tp_segmented_replay:
                    with use_tp(self.otter_graph_for_tp_prefill):
                        self.otter_graph_for_tp_prefill.set_mode(OtterGraphForTP.REPLAY)
                        self.otter_graph_for_tp_prefill.replay_forward()
                        self.otter_graph_for_tp_prefill.set_mode(OtterGraphForTP.OFF)
                elif self.otter_graph_enabled:
                    otter_torch.replay(self.prefill_graph_id)
                else:
                    with (
                        set_forward_context(
                            chunk_attn_metadata,
                            self.vllm_config,
                            num_tokens=chunk_num_tokens_padded,
                            num_tokens_across_dp=num_tokens_across_dp,
                            cudagraph_runtime_mode=cudagraph_mode,
                            batch_descriptor=chunk_batch_desc,
                            ubatch_slices=ubatch_slices_padded,
                            slot_mapping=chunk_slot_mapping,
                            skip_compiled=has_encoder_input,
                        ),
                        record_function_or_nullcontext("gpu_model_runner: forward"),
                        self.maybe_get_kv_connector_output(
                            scheduler_output,
                            defer_finalize=defer_kv_connector_finalize,
                        ) as self.kv_connector_output,
                    ):
                        self.model_output_prefill = self._model_forward(
                            input_ids=input_ids[start:end],
                            positions=positions[start:end],
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=(
                                inputs_embeds[start:end] if inputs_embeds is not None else None
                            ),
                            **model_kwargs,
                        )
        
        # Phase: Decode
        else:
            # update static_forward_context
            if not self.otter_graph_enabled and self.static_forward_context_decode is None:
                self.static_forward_context_decode = StaticForwardContext(
                    attn_metadata=attn_metadata,
                    num_tokens_padded=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_mode=cudagraph_mode,
                    batch_desc=batch_desc,
                    ubatch_slices_padded=ubatch_slices_padded,
                    slot_mappings=slot_mappings,
                    has_encoder_input=has_encoder_input,
                )
                self.static_input_decode = StaticInput(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    model_kwargs=model_kwargs
                )
            else:
                # update static forward context
                next_forward_context_decode = StaticForwardContext(
                    attn_metadata=attn_metadata,
                    num_tokens_padded=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_mode=cudagraph_mode,
                    batch_desc=batch_desc,
                    ubatch_slices_padded=ubatch_slices_padded,
                    slot_mappings=slot_mappings,
                    has_encoder_input=has_encoder_input,
                )
                self.static_forward_context_decode.update(next_forward_context_decode)

                # update static_input
                next_input_decode = StaticInput(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    model_kwargs=model_kwargs
                )
                self.static_input_decode.update(next_input_decode)

                # Refresh the decode KV-write index on the host and push it
                # into every layer's persistent buffer BEFORE replay() --
                # see _update_decode_slot_index for why the timing matters.
                if self.otter_graph_enabled:
                    self._update_decode_slot_index(num_reqs)

            if self.tp_segmented_replay:
                with use_tp(self.otter_graph_for_tp_decode):
                    self.otter_graph_for_tp_decode.set_mode(OtterGraphForTP.REPLAY)
                    self.otter_graph_for_tp_decode.replay_forward()
                    self.otter_graph_for_tp_decode.set_mode(OtterGraphForTP.OFF)
            elif self.otter_graph_enabled:
                otter_torch.replay(self.decode_graph_id)
            else:
                # eager decode
                with (
                    set_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_tokens_padded,
                        num_tokens_across_dp=num_tokens_across_dp,
                        cudagraph_runtime_mode=cudagraph_mode,
                        batch_descriptor=batch_desc,
                        ubatch_slices=ubatch_slices_padded,
                        slot_mapping=slot_mappings,
                        skip_compiled=has_encoder_input,
                    ),
                    record_function_or_nullcontext("gpu_model_runner: forward"),
                    self.maybe_get_kv_connector_output(
                        scheduler_output,
                        defer_finalize=defer_kv_connector_finalize,
                    ) as self.kv_connector_output,
                ):
                    self.model_output_decode = self._model_forward(
                        input_ids=self.static_input_decode.input_ids,
                        positions=self.static_input_decode.positions,
                        intermediate_tensors=self.static_input_decode.intermediate_tensors,
                        inputs_embeds=self.static_input_decode.inputs_embeds,
                        **self.static_input_decode.model_kwargs,
                    )

        # eager: last chunk's output wins
        model_output = self.model_output_prefill if is_prefill else self.model_output_decode

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = (
                    model_output[0].clone(),
                    model_output[1].clone(),
                )
            elif is_prefill:
                # Common case, prefill.
                hidden_states = model_output.clone()
                aux_hidden_states = None
            else:
                # Decode: model_output IS the decode replay graph's
                # persistent output buffer, so skip the .clone() -- safe
                # only because nothing downstream keeps hidden_states alive
                # past this call (EAGLE/pooling/prompt-logprobs are all off
                # for this deployment). Re-add .clone() if any of those
                # get enabled for decode.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = self.kv_connector_output
                    self.kv_connector_output = self.kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        self.kv_connector_output,
                    )

                # Rows to sample from. A gather (hidden_states[logits_indices]) breaks
                # under otter replay, so use slices:
                #  - decode: one token per request, packed in order -> first num_reqs rows.
                #  - prefill: the last chunk is padded to chunk_size, so the real last
                #    token is at (n % chunk_size) - 1, not -1. Single request only.
                if is_prefill:
                    chunk_num_tokens_unpadded = num_tokens_unpadded % self.chunk_size
                    if chunk_num_tokens_unpadded == 0:
                        chunk_num_tokens_unpadded = self.chunk_size
                    sample_hidden_states = hidden_states[chunk_num_tokens_unpadded - 1 : chunk_num_tokens_unpadded]
                else:
                    sample_hidden_states = hidden_states[:num_reqs]
                if self.tp_segmented_replay:
                    assert num_reqs == self.captured_decode_num_reqs, (
                        f"segmented graphs were captured for num_reqs="
                        f"{self.captured_decode_num_reqs}, got {num_reqs}; "
                        "they read fixed row ranges so a larger batch would "
                        "silently drop requests"
                    )
                    if is_prefill:
                        self.static_logits_input.copy_(sample_hidden_states)
                        otter_torch.replay(self.logits_graph_id)
                        logits = self._logits_tail(self.static_logits_local)
                    else:
                        # decode's last graph already contains rms + lm_head
                        logits = self._logits_tail(self.static_logits_local_decode)
                elif self.otter_graph_enabled and not is_prefill:
                    assert num_reqs == self.captured_decode_num_reqs, (
                        f"decode graph was captured for num_reqs="
                        f"{self.captured_decode_num_reqs}, got {num_reqs}; "
                        "the folded lm_head reads a fixed row range so a "
                        "larger batch would silently drop requests"
                    )
                    # compute_logits was folded into the decode graph capture
                    # above, so replaying decode_graph_id already refreshed
                    # static_logits_output_decode (and static_greedy_token_decode)
                    # -- no separate replay needed.
                    logits = self.static_logits_output_decode
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        # self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput | None") -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None

            if self.use_async_scheduling and gpu_mr_mod.get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()

            if not kv_connector_output:
                return None  # type: ignore[return-value]

            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state

        self.execute_model_state = None

        if grammar_output is not None:
            gpu_mr_mod.apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)

        with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        if self.use_async_scheduling:
            pp = gpu_mr_mod.get_pp_group()
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False

        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens <= self.effective_drafter_max_model_len
            )

            use_gpu_toks = (spec_config.use_eagle() or spec_config.uses_draft_model() or spec_config.uses_extract_hidden_states()) and not spec_config.disable_padded_drafter_batch

            if use_gpu_toks:
                assert isinstance(
                    self.drafter,
                    (
                        gpu_mr_mod.EagleProposer,
                        gpu_mr_mod.DraftModelProposer,
                        gpu_mr_mod.ExtractHiddenStatesProposer,
                    ),
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                        spec_decode_common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                    self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
                    self._draft_token_ids = torch.zeros(1, device=self.device, dtype=torch.int32).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

            elif spec_config.use_ngram_gpu() and not spec_config.disable_padded_drafter_batch:
                assert isinstance(self.drafter, gpu_mr_mod.NgramProposerGPU)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count, _ = self.drafter.update_token_ids_ngram(
                        sampled_token_ids,
                        self.input_batch,
                        self.token_ids_gpu_tensor,
                        self.num_tokens_no_spec_gpu,
                        self.discard_request_mask.gpu,
                    )
                    self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
                    self._draft_token_ids = torch.zeros(1, device=self.device, dtype=torch.int32).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if propose_drafts_after_bookkeeping:
            propose_draft_token_ids(valid_sampled_token_ids)

        if spec_config is not None:
            self.finalize_kv_connector()

        with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.routed_experts_initialized:
                capturer = gpu_mr_mod.RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )

        if not self.use_async_scheduling:
            return output

        with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: AsyncGPUModelRunnerOutput"):
            async_output = OtterAsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )

        with gpu_mr_mod.record_function_or_nullcontext("gpu_model_runner: set_async_sampled_token_ids"):
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output


# =========================================================================
# Async Output Wrapper (Replaces torch.cuda.stream with Synced Execution)
# =========================================================================
class OtterAsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    """
    Otter shell version of AsyncGPUModelRunnerOutput.
    Does not use torch.cuda.stream(...)
    Does not rely on CUDA events
    Copies outputs to CPU synchronously for now.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors,
        invalid_req_indices: list[int],
        async_output_copy_stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        self.async_copy_ready_event = OtterDummyEvent()

        self._sampled_token_ids = sampled_token_ids
        self._logprobs_tensors = logprobs_tensors
        self.vocab_size = vocab_size

        # Synced CPU Copy (instead of async stream)
        self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=False)
        self._logprobs_tensors_cpu = self._logprobs_tensors.to_cpu_nonblocking() if self._logprobs_tensors is not None else None

    def get_output(self) -> ModelRunnerOutput:
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = gpu_mr_mod.RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )
        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output
