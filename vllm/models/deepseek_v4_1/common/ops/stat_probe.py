# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project
"""Repeatedly-armable numerical breadcrumbs along the DSv4.1 forward.

Gated by VLLM_DSV41_SM120_PATH_CHECKS. Semantics:
  * an all-zero sample (profile dummy runs) never consumes the tag;
  * the first real clean sample logs once;
  * every non-finite sample logs (up to a small cap per tag) so the
    earliest stage that turns NaN at real decode/prefill steps is visible
    even after its one-shot budget was spent on the profile run.
Capture-safe: skips while a CUDA graph is being captured.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_probe_state: dict[str, int] = {}
_int_probe_seen: dict[str, set[tuple[int, int, int]]] = {}
_MAX_LOGS = 5
_WARN = 1
_CLEAN = 2


def stat_probe_once(tag: str, t: torch.Tensor | None) -> None:
    if t is None or not envs.VLLM_DSV41_SM120_PATH_CHECKS:
        return
    if t.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return
    state = _probe_state.get(tag, 0)
    if state >= _MAX_LOGS:
        return
    tf = t.detach().float()
    if tf.is_cuda:
        tf = tf.reshape(-1)
        if tf.numel() > 65536:
            tf = tf[:65536]
        nan = int(torch.isnan(tf).sum().item())
        inf = int(torch.isinf(tf).sum().item())
        fin = torch.nan_to_num(tf)
        absmax = float(fin.abs().max().item()) if tf.numel() else 0.0
        mean = float(fin.mean().item()) if tf.numel() else 0.0
    else:
        nan = inf = 0
        absmax = float(tf.abs().max().item()) if tf.numel() else 0.0
        mean = float(tf.mean().item()) if tf.numel() else 0.0
    nonfinite = nan + inf > 0
    all_zero = absmax == 0.0 and not nonfinite
    if all_zero and state:
        return  # zeros after the first sample tell us nothing new
    if state & _WARN and not nonfinite:
        return  # once NaN logged, clean repeats are noise
    if state & _CLEAN and not nonfinite and not all_zero:
        return  # one clean real sample is enough
    if nonfinite:
        state |= _WARN
    elif not all_zero:
        state |= _CLEAN
    _probe_state[tag] = state + 1
    log = logger.warning if nonfinite else logger.info
    log(
        "PATH_STATS %s #%d: shape=%s dtype=%s nan=%d inf=%d absmax=%.4g mean=%.4g",
        tag,
        state + 1,
        tuple(t.shape),
        t.dtype,
        nan,
        inf,
        absmax,
        mean,
    )
    if nonfinite and t.dim() >= 2 and t.is_cuda:
        mask = (
            torch.isnan(t.detach().float().reshape(t.shape[0], -1)).any(dim=1)
            .to(torch.uint8)
            .cpu()
        )
        bad = int(mask.sum().item())
        logger.warning(
            "PATH_ROWS %s #%d: rows=%d bad=%d mask=%s",
            tag,
            state + 1,
            t.shape[0],
            bad,
            "".join(map(str, mask[:64].tolist())),
        )


def int_probe_once(tag: str, t: torch.Tensor) -> None:
    """Index-domain counterpart of ``stat_probe_once`` for integer tensors
    (NaN/inf are meaningless there): reports min/max/invalid(-1) counts so
    garbage or all-invalid topk rows become visible. Same budget semantics as
    the float probe: a tag is consumed by *distinct* signatures only (profile
    dummy garbage/zeros no longer silences the real step), all-zero repeats
    never consume, up to ``_MAX_LOGS`` distinct reports per tag."""
    if not envs.VLLM_DSV41_SM120_PATH_CHECKS:
        return
    if t.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return
    if t.numel() == 0:
        return
    flat = t.reshape(-1)
    mn = int(flat.min().item())
    mx = int(flat.max().item())
    neg = int((flat < 0).sum().item())
    seen = _int_probe_seen.setdefault(tag, set())
    if len(seen) >= _MAX_LOGS:
        return
    all_zero = mn == 0 and mx == 0 and neg == 0
    if all_zero and seen:
        return  # zeros after a real sample tell us nothing new
    fp = (mn, mx, neg)
    if fp in seen:
        return
    seen.add(fp)
    all_invalid = neg == flat.numel()
    log = logger.warning if (all_invalid or mx < 0 or mn < -1) else logger.info
    log(
        "PATH_IDX %s #%d: shape=%s dtype=%s min=%d max=%d neg=%d",
        tag,
        len(seen),
        tuple(t.shape),
        t.dtype,
        mn,
        mx,
        neg,
    )
