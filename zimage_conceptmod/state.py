"""Product-owned declarations of execution modules and continuation identity."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math

import torch


def _class_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain(value):
    if isinstance(value, Enum):
        return _plain(value.value)
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous()
        return {"dtype": str(raw.dtype), "shape": list(raw.shape),
                "sha256": _digest(raw.reshape(-1).view(torch.uint8).numpy().tobytes())}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    if is_dataclass(value):
        return _plain(asdict(value))
    if callable(getattr(value, "to_dict", None)):
        return _plain(value.to_dict())
    if isinstance(getattr(value, "content", None), str):
        # tokenizers.AddedToken: retain normalization and whitespace behavior.
        return {key: getattr(value, key) for key in
                ("content", "single_word", "lstrip", "rstrip", "normalized", "special")}
    raise TypeError(f"unsupported behavior configuration value: {_class_name(value)}")


def _components(backend):
    pipeline = getattr(backend, "pipe", None)
    if pipeline is not None:
        values = getattr(pipeline, "components", None)
        if values is None:
            values = {name: value for name, value in vars(pipeline).items() if not name.startswith("_")}
        if not isinstance(values, Mapping):
            raise TypeError("pipeline components must be a mapping")
        return "pipeline", values
    # Native runtimes can own their modules directly rather than in a pipeline.
    return "backend", {name: value for name, value in vars(backend).items() if not name.startswith("_")}


def declared_modules(backend, modules):
    """Include frozen encoders, conditioners and VAE as well as edited modules."""
    modules = dict(modules)
    seen = {id(module) for module in modules.values()}
    prefix, components = _components(backend)
    for name, module in sorted(components.items()):
        if isinstance(module, torch.nn.Module) and id(module) not in seen:
            modules[f"{prefix}.{name}"] = module
            seen.add(id(module))
    return modules


def _tokenizer_identity(tokenizer):
    identity = {"class": _class_name(tokenizer)}
    for name in ("name_or_path", "model_max_length", "padding_side", "truncation_side",
                 "clean_up_tokenization_spaces", "add_bos_token", "add_eos_token", "legacy"):
        if hasattr(tokenizer, name):
            identity[name] = _plain(getattr(tokenizer, name))
    identity["special_tokens"] = _plain(getattr(tokenizer, "special_tokens_map_extended", {}))
    fast = getattr(tokenizer, "backend_tokenizer", None)
    if fast is not None and callable(getattr(fast, "to_str", None)):
        serialized = json.loads(fast.to_str())
        # Transformers sets these temporary backend fields for each call. The
        # wrapper settings above and explicit encode options define behavior.
        serialized.pop("padding", None)
        serialized.pop("truncation", None)
        identity["serialized_sha256"] = _digest(
            json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode())
    elif callable(getattr(tokenizer, "get_vocab", None)):
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
        identity["vocabulary_sha256"] = _digest(payload)
    sentencepiece = getattr(tokenizer, "sp_model", None)
    if callable(getattr(sentencepiece, "serialized_model_proto", None)):
        identity["sentencepiece_sha256"] = _digest(sentencepiece.serialized_model_proto())
    return identity


def runtime_metadata(backend):
    """Describe non-weight behavior; checkpoints separately save every declared module."""
    identity = {"settings": {}, "component_configs": {}, "adapter_configs": {}, "tokenizers": {}}
    for name in ("compute_dtype", "is_distilled", "dummy", "ctx_len", "spec",
                 "max_sequence_length", "default_negative", "patch_size", "vae_scale_factor"):
        if hasattr(backend, name):
            identity["settings"][name] = _plain(getattr(backend, name))
    for name, module in backend.training_modules().items():
        config = getattr(module, "config", None)
        if config is not None:
            identity["component_configs"][name] = _plain(config)
        adapters = getattr(module, "peft_config", None)
        if adapters:
            identity["adapter_configs"][name] = _plain(adapters)
    prefix, components = _components(backend)
    pipeline = getattr(backend, "pipe", None)
    if pipeline is not None:
        config = getattr(pipeline, "config", None)
        if config is not None:
            identity["pipeline_config"] = _plain(config)
    for name, component in sorted(components.items()):
        if not isinstance(component, torch.nn.Module):
            config = getattr(component, "config", None)
            if config is not None:
                identity["component_configs"][f"{prefix}.{name}"] = _plain(config)
        if callable(getattr(component, "get_vocab", None)):
            identity["tokenizers"][name] = _tokenizer_identity(component)
    padding = getattr(backend, "_padding_mask", None)
    if padding is not None:
        identity["padding_mask"] = _plain(padding)
    unconditional = getattr(backend, "_stored_uncond", None)
    if unconditional is not None:
        identity["stored_unconditional"] = {
            "embeds": _plain(unconditional.embeds), "mask": _plain(unconditional.mask)}
    return identity
