"""Lightweight installed-provider metadata; model code is loaded on creation."""
from conceptmod.contracts import BackendCapabilities
from conceptmod.providers import BackendProvider

PROVIDER = BackendProvider(
    name="zimage",
    factory="zimage_conceptmod.backend:ZImageBackend",
    distribution="zimage-conceptmod",
    capabilities=BackendCapabilities(),
    training_defaults=(('sample_steps', 8), ('sample_guidance', 0.0)),
)
