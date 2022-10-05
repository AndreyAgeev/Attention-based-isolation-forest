from omegaconf import DictConfig
from hydra.utils import instantiate

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, average_precision_score

from atif.core import AbstractModel, AbstractDataset
from atif.logger import FileLogger


class Solution:
    def __init__(self, cfg: DictConfig):
        self._model: AbstractModel = instantiate(cfg.model.type_model, _recursive_=False)
        self._dataset: AbstractDataset = instantiate(cfg.dataset, _recursive_=False)
        self._logger = FileLogger(cfg.output_name)
        self._setup(cfg)

    def _setup(self, cfg: DictConfig):
        self._logger.setup()
        self._dataset.load()
        self._model.setup(cfg.model)

    def run(self):
        self._logger.start_logger("model: " + self._model.__str__())
        self._model.fit(self._dataset.X_train,
                        self._dataset.y_train)
        y_prediction = self._model.predict(self._dataset.X_test, self._dataset.y_test)
        self._dataset.plot_dataset(self._dataset.X_train, self._dataset.y_train,
                                   "/Users/andreyageev/PycharmProjects/ATIF/image_27_08/train_"
                                   + self._dataset.get_name() + ".jpg")
        self._dataset.plot_dataset(self._dataset.X_test, self._dataset.y_test,
                                   "/Users/andreyageev/PycharmProjects/ATIF/image_27_08/test_"
                                   + self._dataset.get_name() + ".jpg")
        self._dataset.plot_dataset(self._dataset.X_test, y_prediction,
                                   "/Users/andreyageev/PycharmProjects/ATIF/image_27_08/predict_"
                                   + self._model.get_name() + "_" + self._dataset.get_name() + ".jpg")
        self._log_metrics(y_prediction)

    def _log_metrics(self, y_prediction):
        confusion = confusion_matrix(self._dataset.y_test, y_prediction)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        normal = self._dataset.y_test[self._dataset.y_test == 0]
        anomalies = self._dataset.y_test[self._dataset.y_test == 1]
        F1 = f1_score(self._dataset.y_test, y_prediction)
        self._logger.info(
            f"Proportion anomalies/normal = {len(anomalies)}/{len(normal)} = {(len(anomalies) / len(normal)) * 100:.1f}%")
        self._logger.info(f"F1 score {F1:.4f}")
        self._logger.info(f"TPR score {TPR:.4f}, FPR {FPR:.4f}")
        self._logger.info(f"TP {TP:.4f}, FP {FP:.4f}")
        self._logger.info(f"FN {FN:.4f}, TN {TN:.4f}")
        self._logger.info(f"accuracy {accuracy_score(y_prediction, self._dataset.y_test):.4f}")

    def close(self):
        self._logger.end_logger("end model inference")


def start(cfg: DictConfig):
    """Main function.

    Args:
        cfg (DictConfig): config structure by .yaml.
    """
    solution_runner = Solution(cfg)
    solution_runner.run()
    solution_runner.close()
