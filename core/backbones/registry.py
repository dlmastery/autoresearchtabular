"""Backbone registry — single source of truth for which models exist."""
from __future__ import annotations
from typing import Callable, Dict

BACKBONE_REGISTRY: Dict[str, Callable] = {}


def register_backbone(name: str) -> Callable:
    def deco(cls):
        if name in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone {name!r} already registered")
        BACKBONE_REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def create_backbone(name: str, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Backbone {name!r} not registered. "
            f"Available: {sorted(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](**kwargs)
