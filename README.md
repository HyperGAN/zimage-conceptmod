# zimage-conceptmod

Z-Image Turbo is a standalone model provider for [conceptmod](https://github.com/HyperGAN/conceptmod).
This repository owns the model's loading, adapters, encoding, sampling and
model-specific tests. Conceptmod core owns the DSL, training sessions, metrics
and complete edit-bundle format.

## Install and use

Install conceptmod 3 and this package in the same Python environment:

```bash
python -m pip install "conceptmod @ git+https://github.com/HyperGAN/conceptmod.git@main"
python -m pip install .
conceptmod backends
cat > edit.json <<'JSON'
{"backend": {"device": "cuda:0", "lora_rank": 16}}
JSON
conceptmod train --backend zimage --phrase "red=blue" --config edit.json --out outputs/edit
```

The entry point `conceptmod.backends:zimage` exports lightweight metadata;
listing installed providers does not import inference libraries or load weights.
Python callers can select this provider explicitly:

```python
from conceptmod.providers import BackendRegistry
from zimage_conceptmod.provider import PROVIDER

registry = BackendRegistry([PROVIDER], discover=False)
backend = registry.load("zimage", device="cuda:0")
image = backend.generate("a cat on a windowsill", seed=42)
backend.save_trained("outputs/edit-bundle")
```

The default base-model identifier is `Tongyi-MAI/Z-Image-Turbo`. Base weights are downloaded
only when constructing the real backend; no weights are shipped in this
package. Use the same model revision/local checkpoint when restoring an edit.
Diffusers is pinned to the supported `0.39.0` release; this package does not
install a floating Git checkout. The backend does not enable
`trust_remote_code`.

Training checkpoints declare the frozen execution modules as well as the
trainable adapters, and validate model configuration and tokenizer behavior.
Inference bundles contain the edited components.

## Validation

The CPU workflow builds and installs a wheel, then runs the offline tests from
a temporary directory with importlib mode so the checkout cannot shadow it.

```bash
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m build --wheel --no-isolation
```

CPU tests verify the provider contract and available local/stub behavior. They
do not establish pretrained-image quality or real-GPU training. The manual
`python scripts/smoke_zimage.py` path can load large checkpoints and requires
CUDA.
Run it deliberately on a machine with the required model access and memory.
No GPU test was run as part of this extraction.

## Origin

Extracted from `HyperGAN/conceptmod` at
`70f0cb2632190388eb4028ef80066558259494fc`. Model computations and tensor mappings were retained; package
imports, provider metadata and test routing were moved to this product.
The original MIT license and attribution are retained in [LICENSE](LICENSE).
See [docs/origin.md](docs/origin.md) for the transferred-file inventory.

Historical experiments remain in the immutable [source snapshot](https://github.com/HyperGAN/conceptmod/tree/70f0cb2632190388eb4028ef80066558259494fc/outputs).
