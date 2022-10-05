import hydra
from omegaconf import DictConfig

from atif.tools import inference, optimization


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function.

    Args:
        cfg (DictConfig): configs.
    """
    if cfg.mode == "inference":
        inference(cfg.inference)
    else:
        optimization(cfg.optimization)


if __name__ == '__main__':
    main()
