from omegaconf import DictConfig


class FileLogger:
    def __init__(self):
        super().__init__()
        self._file = None
        self.directory_image = None

    def setup(self, cfg: DictConfig):
        self._file = open(cfg.file_name, "a")
        self.directory_image = cfg.directory_image

    def info(self, value: str):
        """Process function."""
        self._file.write(value + "\n")

    def log(self, name: str, value: float):
        self._file.write(name + " - " + str(value) + "\n")

    def log_artifact(self, file_name: str):
        pass

    def start_logger(self, name_value: str) -> None:
        self._file.write(f"Start logger: {name_value}\n")

    def end_logger(self, name_value: str) -> None:
        self._file.write(f"End logger: {name_value}\n")
        self._file.close()
