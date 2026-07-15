"""Expand a compact QuaRot experiment matrix into independently valid configs."""

import copy
from typing import Any, Dict, Mapping

from repro.manifest import ConfigValidationError, validate_config


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def materialize_matrix(template: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Expand ``common`` plus named ``variants`` and validate every resulting config."""
    common = template.get("common")
    variants = template.get("variants")
    if not isinstance(common, Mapping) or not isinstance(variants, Mapping):
        raise ConfigValidationError("matrix requires object fields common and variants")

    result = {}
    for variant_id, override in variants.items():
        if not isinstance(variant_id, str) or not isinstance(override, Mapping):
            raise ConfigValidationError("matrix variant identifiers and overrides must be objects")
        config = _deep_merge(common, override)
        config["experiment_id"] = variant_id
        validate_config(config)
        result[variant_id] = config
    if not result:
        raise ConfigValidationError("matrix must contain at least one variant")
    return result
