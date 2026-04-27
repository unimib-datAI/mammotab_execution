import json
import os
from pathlib import Path
from typing import Optional, Tuple


ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = value.strip()
    return value or None


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    value = clean_optional(value)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False

    raise ValueError(f"Invalid boolean value: {value}")


def normalize_adapter_path(adapter_path: Optional[str]) -> Optional[str]:
    adapter_path = clean_optional(adapter_path)
    if adapter_path is None:
        return None

    path = Path(os.path.expandvars(adapter_path)).expanduser()
    if not path.is_dir():
        raise ValueError(f"Adapter path does not exist or is not a directory: {path}")

    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"Adapter path must contain adapter_config.json: {path}")

    if not any((path / file_name).is_file() for file_name in ADAPTER_WEIGHT_FILES):
        expected_files = ", ".join(ADAPTER_WEIGHT_FILES)
        raise ValueError(
            f"Adapter path must contain one of these weight files: {expected_files}"
        )

    return str(path)


def is_adapter_directory(value: str) -> bool:
    path = Path(os.path.expandvars(value)).expanduser()
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and not (path / "config.json").is_file()
    )


def infer_base_model_name(adapter_path: str) -> str:
    adapter_path = normalize_adapter_path(adapter_path)
    config_path = Path(adapter_path) / "adapter_config.json"

    with config_path.open("r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)

    base_model_name = clean_optional(adapter_config.get("base_model_name_or_path"))
    if base_model_name is None:
        raise ValueError(
            "MODEL_NAME is required because adapter_config.json does not include "
            "base_model_name_or_path."
        )

    return base_model_name


def resolve_model_inputs(
    model_name: Optional[str],
    tokenizer_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    model_name = clean_optional(model_name)
    tokenizer_name = clean_optional(tokenizer_name)

    if (
        adapter_path is None
        and model_name is not None
        and is_adapter_directory(model_name)
    ):
        adapter_path = model_name
        model_name = None

    adapter_path = normalize_adapter_path(adapter_path)

    if model_name is None and adapter_path is not None:
        model_name = infer_base_model_name(adapter_path)

    if model_name is None:
        raise ValueError(
            "MODEL_NAME is required. When using adapters, set MODEL_NAME to the "
            "base model or provide an adapter_config.json with base_model_name_or_path."
        )

    if tokenizer_name is None:
        tokenizer_name = model_name

    return model_name, tokenizer_name, adapter_path


def get_run_model_name(
    model_name: Optional[str], adapter_path: Optional[str] = None
) -> str:
    model_name = clean_optional(model_name)
    if model_name is None:
        raise ValueError("MODEL_NAME is required.")

    adapter_path = clean_optional(adapter_path)
    if adapter_path is None:
        return model_name

    return f"{model_name}+{Path(adapter_path).name}"
