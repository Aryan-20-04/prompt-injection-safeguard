import yaml
from pathlib import Path


class ConfigLoader:
    """
    Loads and validates benchmark configuration files.
    """

    REQUIRED_TOP_LEVEL = [
        "run",
        "datasets",
        "models",
        "labels"
    ]

    def load(self, config_path: str):

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self._validate(config)

        return config

    def _validate(self, config):

        for key in self.REQUIRED_TOP_LEVEL:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")

        if "output_dir" not in config["run"]:
            raise ValueError("run.output_dir missing")

        if not isinstance(config["datasets"], list):
            raise ValueError("datasets must be a list")

        if not isinstance(config["models"], list):
            raise ValueError("models must be a list")

        if not isinstance(config["labels"], list):
            raise ValueError("labels must be a list")