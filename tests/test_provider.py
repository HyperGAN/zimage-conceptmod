"""Provider discovery is lightweight; runtime contracts use the shared core."""
import importlib
import subprocess
import sys
from types import SimpleNamespace

from conceptmod.backends.base import Backend
from conceptmod.providers import BackendRegistry
from zimage_conceptmod.provider import PROVIDER


def test_provider_metadata_is_lightweight():
    code = """
import sys
from zimage_conceptmod.provider import PROVIDER
assert PROVIDER.name == "zimage"
assert PROVIDER.distribution == "zimage-conceptmod"
assert not set(sys.modules) & {"torch", "diffusers", "transformers", "peft"}
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_registry_resolves_lazy_factory(monkeypatch):
    marker = SimpleNamespace()
    options = {}
    def create(**kwargs):
        options.update(kwargs)
        return marker
    monkeypatch.setitem(sys.modules, "zimage_conceptmod.backend", SimpleNamespace(ZImageBackend=create))
    registry = BackendRegistry([PROVIDER], discover=False)
    assert registry.names() == ("zimage",)
    assert registry.load("zimage", device="cpu", model_id="fixture") is marker
    assert options == {"device": "cpu", "model_id": "fixture"}
    assert marker.capabilities == PROVIDER.capabilities


def test_runtime_implements_shared_backend_contract():
    cls = getattr(importlib.import_module("zimage_conceptmod.backend"), "ZImageBackend")
    assert issubclass(cls, Backend)
    for method in ("encode_text", "encode_text_grad", "predict_v", "partial_denoise",
                   "render", "generate", "attach_encoder_lora", "trainable_parameters",
                   "save_trained", "load_trained", "edit_components",
                   "training_modules", "training_identity", "invalidate_caches"):
        assert callable(getattr(cls, method))
    backend = object.__new__(cls)
    transformer, encoder = object(), object()
    backend.transformer = transformer
    backend.text_encoder = encoder
    backend.pipe = SimpleNamespace(text_encoder=encoder)
    backend.encoder_lora = False
    assert backend.edit_components() == {"transformer": transformer}
    backend.encoder_lora = True
    assert backend.edit_components() == {"transformer": transformer, "text_encoder": encoder}


def test_complete_bundle_contract_without_a_real_model(tmp_path):
    import torch
    from conceptmod.artifacts import read_edit_bundle

    cls = getattr(importlib.import_module("zimage_conceptmod.backend"), "ZImageBackend")
    def fixture():
        backend = object.__new__(cls)
        backend.device = "cpu"
        backend.model_id = "local-contract-fixture"
        backend.latent_shape = (2,)
        backend.lora_rank = None
        backend.transformer = torch.nn.Linear(2, 2)
        encoder = torch.nn.Linear(2, 2)
        backend.text_encoder = encoder
        backend.pipe = SimpleNamespace(text_encoder=encoder)
        backend.encoder_lora = True
        backend._text_cache = {"old": object()}
        backend.frozen = torch.nn.Linear(2, 2)
        return backend
    source, restored = fixture(), fixture()
    reference = {key: value.clone() for key, value in restored.frozen.state_dict().items()}
    path = tmp_path / "edit"
    source.save_trained(str(path))
    manifest = read_edit_bundle(path)
    assert set(manifest["components"]) == {"transformer", "text_encoder"}
    restored.load_trained(str(path))
    assert not restored._text_cache
    for name, module in source.edit_components().items():
        for key, value in module.state_dict().items():
            assert torch.equal(restored.edit_components()[name].state_dict()[key], value)
    for key, value in reference.items():
        assert torch.equal(restored.frozen.state_dict()[key], value)


class _TokenizerFixture:
    name_or_path = "local-tokenizer-fixture"
    model_max_length = 16
    padding_side = "right"
    truncation_side = "right"
    special_tokens_map_extended = {"bos_token": "<s>"}

    def __init__(self):
        self.vocabulary = {"red": 0, "blue": 1}

    def get_vocab(self):
        return dict(self.vocabulary)


def _state_fixture():
    import torch

    cls = getattr(importlib.import_module("zimage_conceptmod.backend"), "ZImageBackend")
    backend = object.__new__(cls)
    backend.device = "cpu"
    backend.model_id = "local-contract-fixture"
    backend.latent_shape = (2,)
    backend.lora_rank = None
    backend.transformer = torch.nn.Linear(2, 2)
    backend.text_encoder = torch.nn.Linear(2, 2)
    backend.vae = torch.nn.Linear(2, 2)
    backend.vae.config = {"scaling_factor": 0.5}
    backend.frozen = torch.nn.Linear(2, 2)
    backend.tokenizer = _TokenizerFixture()
    conditioner = torch.nn.Linear(2, 2)
    backend.pipe = SimpleNamespace(
        components={"transformer": backend.transformer, "text_encoder": backend.text_encoder,
                    "vae": backend.vae, "text_conditioner": conditioner,
                    "tokenizer": backend.tokenizer,
                    "scheduler": SimpleNamespace(config={"prediction_type": "flow_prediction"})},
        text_encoder=backend.text_encoder,
        config={"is_distilled": False},
    )
    backend.encoder_lora = True
    backend._text_cache = {}
    return backend, conditioner


def test_continuation_declares_frozen_execution_modules_without_aliases():
    backend, conditioner = _state_fixture()
    modules = backend.training_modules()
    assert modules["transformer"] is backend.transformer
    assert modules["text_encoder"] is backend.text_encoder
    assert modules["frozen"] is backend.frozen
    assert modules["pipeline.vae"] is backend.vae
    assert modules["pipeline.text_conditioner"] is conditioner
    assert len({id(module) for module in modules.values()}) == len(modules)
    assert not any(isinstance(module, _TokenizerFixture) for module in modules.values())


def test_continuation_identity_tracks_configuration_and_tokenizer_behavior():
    backend, _ = _state_fixture()
    first = backend.training_identity()
    assert "pipeline.vae" in first["module_graph"]
    assert "tokenizer" in first["metadata"]["runtime"]["tokenizers"]
    backend.tokenizer.vocabulary["red"] = 7
    assert backend.training_identity() != first
    second = backend.training_identity()
    backend.vae.config["scaling_factor"] = 0.75
    assert backend.training_identity() != second


def test_complete_bundle_leaves_execution_state_out_of_inference_weights(tmp_path):
    import torch
    from conceptmod.artifacts import read_edit_bundle

    backend, _ = _state_fixture()
    before = backend.training_identity()
    path = tmp_path / "edit"
    backend.save_trained(str(path))
    manifest = read_edit_bundle(path)
    assert set(manifest["components"]) == {"transformer", "text_encoder"}
    assert "pipeline.vae" in manifest["metadata"]["runtime"]["component_configs"]
    assert backend.training_identity() == before


def test_tokenizer_identity_ignores_temporary_fast_padding_state():
    import json
    from zimage_conceptmod.state import _tokenizer_identity

    tokenizer = _TokenizerFixture()
    serialized = {"model": {"vocab": {"red": 0}}, "padding": None, "truncation": None}
    tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(serialized))
    before = _tokenizer_identity(tokenizer)
    serialized["padding"] = {"strategy": "BatchLongest"}
    serialized["truncation"] = {"max_length": 16}
    assert _tokenizer_identity(tokenizer) == before
    serialized["model"]["vocab"]["red"] = 1
    assert _tokenizer_identity(tokenizer) != before


def test_unsupported_train_method_rejects_before_mutating_model():
    import pytest
    import torch

    backend, _ = _state_fixture()
    backend.lora_rank = 4
    backend.transformer.eval()
    first_parameter = next(backend.transformer.parameters())
    first_parameter.requires_grad_(False)
    flags = [parameter.requires_grad for parameter in backend.transformer.parameters()]
    for method in ("xattn", "full", "typo"):
        with pytest.raises(ValueError, match="train_method"):
            backend.trainable_parameters(method)
        assert not backend.transformer.training
        assert [parameter.requires_grad for parameter in backend.transformer.parameters()] == flags


def test_default_train_method_selects_product_lora_parameters():
    backend, _ = _state_fixture()
    backend.lora_rank = 4
    selected = backend.trainable_parameters(None)
    assert selected
    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in backend.transformer.parameters() if parameter.requires_grad}


def test_peft_configuration_identity_is_weights_only_checkpoint_safe():
    import io
    import torch
    from peft import LoraConfig

    backend, _ = _state_fixture()
    backend.transformer.peft_config = {
        "default": LoraConfig(r=2, lora_alpha=4, target_modules=["linear"])}
    identity = backend.training_identity()
    serialized = io.BytesIO()
    torch.save(identity, serialized)
    serialized.seek(0)
    restored = torch.load(serialized, weights_only=True)
    assert restored == identity
    assert type(restored["metadata"]["runtime"]["adapter_configs"]["transformer"]["default"]["peft_type"]) is str
