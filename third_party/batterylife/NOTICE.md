# Third-party code: BatteryLife

Files in this folder come from the BatteryLife repository (https://github.com/Ruifeng-Tan/BatteryLife,
commit 3b17bb53202e10711deae0448c91f84144bb3dd9), released under the MIT License (see `LICENSE`).

| File | Origin | Changes |
|---|---|---|
| `models/CPMLP.py`, `models/CPTransformer.py` | `models/` | none |
| `layers/*.py`, `utils/masking.py` | `layers/`, `utils/` | `layers/SelfAttention_Family.py`: comments translated to English; code unchanged |
| `curves.py` | `data_provider/data_loader.py`, class `Dataset_original` | the three curve methods only; changes marked `# [vendor]` (see the file's docstring) |
| `_stubs/reformer_pytorch/__init__.py` | new | placeholder for an optional dependency that CPMLP and CPTransformer never use |

The output head of both models is replaced at run time by a quantile head (`delta_cp/model.py`); the files here are
not modified for that.
