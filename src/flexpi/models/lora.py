"""LoRA fine-tuning of the FlexPi MoT (both experts).

A LoRA run freezes every MoT parameter and trains low-rank adapters on the
``nn.Linear`` layers that ``lora.target_modules`` names, plus the MoT
parameters ``lora.full_train`` names, at full rank. Everything outside the MoT
trains exactly as in a full fine-tune: ``FlexPi.FROZEN_MODULES`` still decides
(VAE, text encoder, pointmap encoder and, by default, the DINO encoder frozen;
the DINO / pointmap / proprio heads trainable).

Adapters attach by swapping each target Linear's class in place, so ``weight``
and ``bias`` keep their parameter objects and state-dict names: a stock
checkpoint loads into a LoRA model by name, and ``merged_mot_state_dict``
returns a ``mot`` state dict with exactly the stock keys and shapes. The MoT
calls these Linears as modules (``block.self_attn.q(x)``, ``block.ffn(x)``),
never through their ``.weight``, so the adapter is on every training path.

Precision. The base keeps the model dtype (bf16 under ``mixed_precision:
bf16``). ``lora_A`` / ``lora_B`` are created fp32, so on the plain and DDP
backends AdamW keeps fp32 adapters and moments; DeepSpeed casts the whole module
to bf16 and keeps its own fp32 master copy of every trainable tensor. The
forward casts the adapters to the activation dtype, which under autocast is a
no-op and outside it (``Wan22Trainer.evaluate`` -> ``model.infer``) is what lets
fp32 adapters meet bf16 activations.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexpi.utils.logging_config import get_logger

logger = get_logger(__name__)

LORA_SUFFIXES = ("lora_A", "lora_B")


class LoRALinear(nn.Linear):
    """``nn.Linear`` plus ``scaling * B(A(x))``; built by ``_attach``, never directly."""

    lora_A: nn.Parameter  # [rank, in_features]
    lora_B: nn.Parameter  # [out_features, rank], zero at attach
    lora_scaling: float
    lora_dropout: float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        h = F.dropout(x, self.lora_dropout, self.training) if self.lora_dropout > 0 else x
        h = F.linear(F.linear(h, self.lora_A.to(h.dtype)), self.lora_B.to(h.dtype))
        return out + h * self.lora_scaling

    def delta_weight(self) -> torch.Tensor:
        """``scaling * B @ A`` in fp32."""
        return (self.lora_B.float() @ self.lora_A.float()) * self.lora_scaling


@dataclass
class LoRASpec:
    rank: int
    alpha: float
    dropout: float = 0.0
    # Per MoT expert ("video", "action"): a regex that must fullmatch a module
    # name inside the expert (e.g. "blocks.3.self_attn.q").
    target_modules: Dict[str, str] = field(default_factory=dict)
    # Per expert: a regex that must fullmatch a parameter name inside the
    # expert; matches stay trainable at full rank (and may be re-initialized
    # by a shape-mismatched warm start, unlike every other MoT tensor).
    full_train: Dict[str, str] = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)

    @classmethod
    def from_cfg(cls, cfg) -> Optional["LoRASpec"]:
        """None when the block is absent or ``enabled`` is false."""
        if cfg is None or not bool(cfg.get("enabled", False)):
            return None
        spec = cls(
            rank=int(cfg["rank"]),
            alpha=float(cfg["alpha"]),
            dropout=float(cfg.get("dropout", 0.0) or 0.0),
            target_modules={str(k): str(v) for k, v in dict(cfg.get("target_modules") or {}).items() if v},
            full_train={str(k): str(v) for k, v in dict(cfg.get("full_train") or {}).items() if v},
        )
        if spec.rank < 1 or spec.alpha <= 0 or not (0.0 <= spec.dropout < 1.0):
            raise ValueError(f"lora: rank >= 1, alpha > 0, 0 <= dropout < 1; got {spec}")
        if not spec.target_modules:
            raise ValueError("lora.enabled=true with no lora.target_modules")
        return spec

    def to_dict(self) -> dict:
        return {
            "rank": self.rank, "alpha": self.alpha, "dropout": self.dropout,
            "target_modules": dict(self.target_modules), "full_train": dict(self.full_train),
        }


def _never() -> re.Pattern:
    return re.compile(r"(?!)")


def _experts(mot) -> Dict[str, nn.Module]:
    return dict(mot.mixtures.items())


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in module.modules())


def _attach(lin: nn.Linear, rank: int, scaling: float, dropout: float, generator: torch.Generator) -> None:
    a = torch.empty(rank, lin.in_features, dtype=torch.float32)
    nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)  # peft's default init
    lin.__class__ = LoRALinear
    lin.lora_A = nn.Parameter(a.to(lin.weight.device))
    lin.lora_B = nn.Parameter(torch.zeros(lin.out_features, rank, dtype=torch.float32, device=lin.weight.device))
    lin.lora_scaling = scaling
    lin.lora_dropout = dropout


def inject_lora(model: nn.Module, spec: LoRASpec, seed: int = 0) -> dict:
    """Attach adapters to the MoT Linears ``spec.target_modules`` names.

    Sets ``model.lora_spec`` (read by the trainer's freeze and by
    ``FlexPi.save_checkpoint``) and applies the LoRA trainability. ``lora_A``
    is drawn from its own generator seeded with ``seed``, so every rank of a
    multi-GPU run attaches identical adapters and the global RNG stream the
    rest of training uses is not consumed. Raises when a pattern matches a
    non-Linear module or nothing at all, and when a ``full_train`` pattern
    matches no parameter.
    """
    if has_lora(model.mot):
        raise RuntimeError("inject_lora: the MoT already carries LoRA adapters")
    gen = torch.Generator().manual_seed(int(seed))
    summary = {}
    for name, expert in _experts(model.mot).items():
        pattern = spec.target_modules.get(name)
        if not pattern:
            continue
        rx = re.compile(pattern)
        hits = [(n, m) for n, m in expert.named_modules() if rx.fullmatch(n)]
        if not hits:
            raise ValueError(f"lora.target_modules.{name}={pattern!r} matches no module of the {name} expert")
        for n, m in hits:
            if type(m) is not nn.Linear:
                raise TypeError(f"lora.target_modules.{name} matches {n!r}, a {type(m).__name__}, not an nn.Linear")
            _attach(m, spec.rank, spec.scaling, spec.dropout, gen)
        summary[name] = {
            "modules": len(hits),
            "lora_params": sum(m.lora_A.numel() + m.lora_B.numel() for _, m in hits),
        }
    unknown = set(spec.target_modules) - set(_experts(model.mot))
    if unknown:
        raise ValueError(f"lora.target_modules names no MoT expert: {sorted(unknown)}")
    for name, pattern in spec.full_train.items():
        if name not in _experts(model.mot):
            raise ValueError(f"lora.full_train names no MoT expert: {name!r}")
        rx = re.compile(pattern)
        if not any(rx.fullmatch(n) for n, _ in _experts(model.mot)[name].named_parameters()):
            raise ValueError(f"lora.full_train.{name}={pattern!r} matches no parameter of the {name} expert")
    model.lora_spec = spec
    apply_lora_trainability(model)
    logger.info(
        "LoRA attached: rank=%d alpha=%g (scaling %.3g) dropout=%g | %s",
        spec.rank, spec.alpha, spec.scaling, spec.dropout,
        ", ".join(f"{k}: {v['modules']} Linears / {v['lora_params'] / 1e6:.2f}M" for k, v in summary.items()),
    )
    return summary


def apply_lora_trainability(model: nn.Module) -> None:
    """Freeze the MoT, then re-enable the adapters and the ``full_train`` matches.

    Runs at the end of ``Wan22Trainer._apply_dit_only_train_mode``, which
    re-enables the whole MoT every time it runs (train start, after every eval).
    """
    spec: LoRASpec = model.lora_spec
    model.mot.requires_grad_(False)
    for name, expert in _experts(model.mot).items():
        ft = re.compile(spec.full_train[name]) if spec.full_train.get(name) else _never()
        for n, p in expert.named_parameters():
            if n.endswith(LORA_SUFFIXES) or ft.fullmatch(n):
                p.requires_grad_(True)


def trainable_summary(model: nn.Module) -> str:
    lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.endswith(LORA_SUFFIXES))
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"trainable {total / 1e6:.2f}M = LoRA {lora / 1e6:.2f}M + full-rank {(total - lora) / 1e6:.2f}M"


def assert_base_loaded(model: nn.Module, payload: dict, spec: LoRASpec) -> None:
    """Every MoT tensor a LoRA run keeps frozen must come from the checkpoint.

    ``load_checkpoint`` keeps a missing or shape-mismatched key at its current
    value (``strict=False`` plus the shape filter) and drops unknown keys; under
    LoRA that value would stay frozen for the whole run. Only ``full_train``
    matches may be absent or re-shaped (a new embodiment's action head).
    """
    ckpt = payload.get("mot")
    if not isinstance(ckpt, dict):
        raise ValueError("lora: the warm-start checkpoint has no `mot` state dict")
    own = model.mot.state_dict()
    unexpected = sorted(k for k in ckpt if k not in own)
    bad = []
    for name, expert in _experts(model.mot).items():
        ft = re.compile(spec.full_train[name]) if spec.full_train.get(name) else _never()
        for n, t in expert.state_dict().items():
            if n.endswith(LORA_SUFFIXES) or ft.fullmatch(n):
                continue
            key = f"mixtures.{name}.{n}"
            if key not in ckpt:
                bad.append(f"{key}: missing")
            elif tuple(ckpt[key].shape) != tuple(t.shape):
                bad.append(f"{key}: ckpt {list(ckpt[key].shape)} vs model {list(t.shape)}")
    if bad or unexpected:
        raise ValueError(
            "lora: the frozen base does not match the checkpoint "
            f"({len(bad)} frozen tensor(s) not loaded, {len(unexpected)} unexpected key(s)); "
            f"first: {(bad + unexpected)[:8]}"
        )


@torch.no_grad()
def merged_mot_state_dict(mot: nn.Module) -> dict:
    """``mot.state_dict()`` with every adapter folded in: stock keys, stock dtypes.

    ``weight + scaling * B @ A`` is formed in fp32 and rounded once to the
    weight's dtype; merged tensors are moved to CPU as they are made, so the
    peak extra GPU memory is one fp32 weight. Logs how much of the adapters'
    change the rounded weights keep: the projection
    ``<round(W + dW) - W, dW> / ||dW||^2``, over all Linears and for the lowest
    one. A bf16 weight cannot take a change below half its ulp, so a small
    ``dW`` is kept only in part; above that scale the rounding adds the half-ulp
    noise every bf16 weight carries (the logged residual) and keeps the change.
    """
    sd = mot.state_dict()
    d2 = w2 = kept = e2 = 0.0
    lowest = (float("inf"), None)
    n = 0
    for name, m in mot.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        w = m.weight.float()
        delta = m.delta_weight()
        merged = (w + delta).to(m.weight.dtype)
        step = merged.float() - w
        dd = float(delta.pow(2).sum())
        kk = float((step * delta).sum())
        d2 += dd
        kept += kk
        e2 += float((step - delta).pow(2).sum())
        w2 += float(w.pow(2).sum())
        if dd > 0 and kk / dd < lowest[0]:
            lowest = (kk / dd, name)
        sd[f"{name}.weight"] = merged.cpu()
        del sd[f"{name}.lora_A"], sd[f"{name}.lora_B"]
        n += 1
    if n:
        logger.info(
            "LoRA merge: %d Linears, ||dW||/||W|| = %.3e, change kept by the bf16 weights = %.4f "
            "(lowest %.4f, %s), rounding residual ||round(W+dW)-(W+dW)||/||dW|| = %.3f",
            n, math.sqrt(d2 / max(w2, 1e-30)), kept / max(d2, 1e-30), lowest[0], lowest[1],
            math.sqrt(e2 / max(d2, 1e-30)),
        )
    return sd


def mot_state_dict_for_save(mot: nn.Module) -> dict:
    return merged_mot_state_dict(mot) if has_lora(mot) else mot.state_dict()


def base_fingerprint(model: nn.Module) -> list:
    """fp64 sums of squares of the first four adapted weights.

    Tells the base from a weights file with the adapters merged in (which moves
    each sum by about ``2 <W, dW> + ||dW||^2``); compared with a relative
    tolerance, since the reduction order differs between devices.
    """
    out = []
    for _, m in model.mot.named_modules():
        if isinstance(m, LoRALinear):
            out.append(float(m.weight.detach().double().pow(2).sum()))
            if len(out) == 4:
                break
    return out


def adapter_payload(model: nn.Module, step=None, base_ckpt=None) -> dict:
    """The adapters plus every other trainable tensor, keyed as in ``model.state_dict()``."""
    lora = {k: v.detach().cpu() for k, v in model.mot.state_dict().items() if k.endswith(LORA_SUFFIXES)}
    trainable = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad and not n.endswith(LORA_SUFFIXES)
    }
    return {
        "format": "flexpi-lora-v1",
        "lora_config": model.lora_spec.to_dict(),
        "lora": lora,
        "trainable": trainable,
        "base_ckpt": None if base_ckpt is None else str(base_ckpt),
        "base_fingerprint": base_fingerprint(model),
        "action_dim_mask_trained": getattr(model, "_action_dim_mask_trained", None),
        "step": step,
    }


def load_adapter(model: nn.Module, path_or_payload, seed: int = 0) -> dict:
    """Attach (if needed) and load an ``adapter_payload`` file into a model holding its base.

    Raises when the model already carries adapters with another ``lora`` config, or
    when its base is not the one the file was trained on (``base_fingerprint``:
    e.g. the run's merged weights, which would apply the change twice).
    """
    payload = path_or_payload if isinstance(path_or_payload, dict) else torch.load(path_or_payload, map_location="cpu")
    if payload.get("format") != "flexpi-lora-v1":
        raise ValueError("not a flexpi LoRA adapter file")
    if has_lora(model.mot):
        if model.lora_spec.to_dict() != payload["lora_config"]:
            raise ValueError(
                f"the model carries adapters with lora config {model.lora_spec.to_dict()}, "
                f"the file {payload['lora_config']}"
            )
    else:
        inject_lora(model, LoRASpec(**payload["lora_config"]), seed=seed)
    expected = payload.get("base_fingerprint")
    if expected is not None and not all(
        math.isclose(a, b, rel_tol=1e-7) for a, b in zip(base_fingerprint(model), expected)
    ):
        raise ValueError(
            "this model's base is not the one the adapter file was trained on "
            f"(base_ckpt={payload.get('base_ckpt')}); a merged weights file already holds the adapters"
        )
    missing, unexpected = model.mot.load_state_dict(payload["lora"], strict=False)
    lora_keys = {k for k in model.mot.state_dict() if k.endswith(LORA_SUFFIXES)}
    if set(payload["lora"]) != lora_keys or unexpected:
        raise ValueError("adapter file does not match this model's adapters")
    missing, unexpected = model.load_state_dict(payload["trainable"], strict=False)
    if unexpected:
        raise ValueError(f"adapter file carries unknown tensors: {unexpected[:8]}")
    if "action_dim_mask_trained" in payload:
        model._action_dim_mask_trained = payload["action_dim_mask_trained"]
    return payload
