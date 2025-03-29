from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from typing import List
from tqdm import tqdm
from .evaluater import Evaluater
import logging
import os
import shutil
from utils import format_metrics
logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self, model: nn.Module, datasets: List[DataLoader], optimizer: Optimizer,
        warmup_scheduler: LinearLR, decay_scheduler: LRScheduler,
        accumulation_steps: int, clip_grad_norm: float, metric: str, output_dir: str, epochs: int, patience: int
    ) -> None:
        self.model = model
        self.train_data = datasets[0]
        self.dev_data = datasets[1]
        self.test_data = datasets[2]

        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.decay_scheduler = decay_scheduler

        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm

        self.metric_name = metric[1:]
        self.metric_mode = metric[0]
        self.metric_score = -100000.0 if self.metric_mode == "+" else 100000.0

        self.output_dir = output_dir
        self.epochs = epochs
        self.patience = patience

        self.dev_evaluater = Evaluater(self.model, self.dev_data)
        self.test_evaluater = Evaluater(self.model, self.test_data)

        self.current_epoch = 0
        self.current_patience = 0
        self.current_step = 0
        self.loss_score = 0.0

    def train(self) -> None:
        logger.info("Training starts ...")

        self.optimizer.zero_grad()
        for _ in range(self.epochs):
            self.current_epoch += 1
            self.loss_score = 0.0
            self.model.train()

            tbar = tqdm(self.train_data, desc=f"Epoch {self.current_epoch:02d}: ")
            for mini_batch in tbar:
                self.current_step += 1
                model_output = self.model(**mini_batch)
                loss = model_output["loss"]
                loss.backward()

                self.loss_score += loss.item()
                metrics = self.model.get_metrics()
                metrics["loss"] = loss.item()
                tbar.set_postfix_str(format_metrics(metrics))

                if self.current_step % self.accumulation_steps == 0:
                    if self.clip_grad_norm > 0.:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.current_step // self.accumulation_steps <= self.warmup_scheduler.total_iters + 1:
                        self.warmup_scheduler.step()
                        if self.current_step % (self.accumulation_steps*100) == 0:
                            logger.info(f"lr is {self.warmup_scheduler.get_last_lr()[0]}")

            self.optimizer.step()
            self.optimizer.zero_grad()

            metrics = self.model.get_metrics(reset=True)
            metrics["loss"] = self.loss_score / len(self.train_data.dataset)
            logger.info(f"Epoch {self.current_epoch:02d}: train_set: " + format_metrics(metrics))

            logger.info("Evaluating on dev set: ")
            metrics = self.dev_evaluater.evaluate()

            main_metric = metrics[self.metric_name]
            if self.current_step > (self.warmup_scheduler.total_iters + 1) * self.accumulation_steps:
                self.decay_scheduler.step(main_metric)
                logger.info(f"lr is {self.decay_scheduler.get_last_lr()[0]}")

            metric_flag = self.metric_score <= main_metric if self.metric_mode == "+" else self.metric_score >= main_metric
            if metric_flag:
                self.metric_score = main_metric
                self.current_patience = 0

                model_save_dir = os.path.join(self.output_dir, "model")
                if os.path.exists(model_save_dir):
                    shutil.rmtree(model_save_dir)
                self.model.save_model(model_save_dir)
            else:
                self.current_patience += 1

            if self.current_patience > self.patience:
                break

        logger.info("Evaluating on test set: ")
        metrics = self.test_evaluater.evaluate()

        logger.info("Training ends ...")
