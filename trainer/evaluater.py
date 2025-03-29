import torch
import logging
import jsonlines
from torch import nn
from tqdm import tqdm
from utils import format_metrics
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)


class Evaluater:

    def __init__(
        self, model: nn.Module, dataset: DataLoader, output_file: str = None
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.output_file = output_file

    def evaluate(self):
        self.model.eval()
        file_writer = jsonlines.open(self.output_file, "w") if self.output_file is not None else None
        tbar = tqdm(self.dataset, desc=f"Evaluating: ")
        loss_score = 0.0

        with torch.no_grad():
            for batch in tbar:
                model_output = self.model(**batch)
                loss_score += model_output["loss"].item()

                if file_writer is not None:
                    file_writer.write(self.model.make_output_human_readable(model_output))

                metrics = self.model.get_metrics()
                metrics["loss"] = model_output["loss"].item()
                tbar.set_postfix_str(format_metrics(metrics))

        metrics = self.model.get_metrics(reset=True)
        metrics["loss"] = loss_score / len(self.dataset)

        logger.info(format_metrics(metrics))
        if file_writer is not None:
            file_writer.close()
        self.model.train()

        return metrics
