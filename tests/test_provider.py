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
                   "save_trained", "load_trained", "edit_components"):
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
