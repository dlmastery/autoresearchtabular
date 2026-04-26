"""Backbone registry."""
from .registry import BACKBONE_REGISTRY, register_backbone, create_backbone
from . import gbm, sklearn_baselines, mlp  # noqa: F401  registers self
try:
    from . import sota  # noqa: F401  registers TabM/FT-T/MLP-PLR/ResNet
except ImportError as _e:
    import warnings
    warnings.warn(f"SOTA backbones unavailable (missing tabm/rtdl): {_e!r}")
