from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from nova.paths import CONFIGS_DIR, LLM_CONFIGS_DIR


def get_dict_from_config(config_name: str, config_dir: Path = CONFIGS_DIR) -> DictConfig:
    """Get a DictConfig object from a Hydra config file.

    Args:
        config_name: The name of the config file (without extension).
        config_dir: The directory containing the config file, defaults to CONFIGS_DIR.

    Returns:
        A DictConfig object containing the configuration.
    """
    if GlobalHydra().is_initialized():
        hydra.utils.log.info(f"Hydra initialized, using config dir: {config_dir}")
        config = hydra.compose(config_name=config_name)
    else:
        with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = hydra.compose(config_name=config_name)
    return config


def get_model_config(model_name: str) -> DictConfig:
    """Gets the DictConfig object for a model of a given name.

    Args:
        model_name: The name of the model. A config file of the same name must exist in the LLM config folder.

    Returns:
        A DictConfig object containing the model configuration.
    """
    config = OmegaConf.load(LLM_CONFIGS_DIR / f"{model_name}.yaml")
    assert isinstance(config, DictConfig)
    return config


def save_config(config: DictConfig, config_path: Path) -> None:
    OmegaConf.save(config, config_path, resolve=True)
