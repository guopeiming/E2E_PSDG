import os
import random
import torch
import shutil
import numpy as np
from argparse import Namespace
import torch.utils.data
from config import parse_args
from typing import List
from dataset import PipelineTextGeneraterDataset, DialogueLengthSampler, PipelineImageGeneraterDataset, JointGeneraterDataset
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from model import PipelineTextGenerater, PipelineImageGenerater, JointGenerater
from trainer import Trainer, Evaluater
import logging
logger = logging.getLogger(__name__)


def pre_experiment_init(args: Namespace) -> None:
    # ===== path init =====
    if args.debug:
        assert os.path.basename(args.output_dir) == "debug", "debug path error"
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        raise ValueError("output dir exists.")

    # ===== logging init =====
    logging.basicConfig(
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        filename=os.path.join(args.output_dir, f"{args.subcommand}.log"), level=logging.DEBUG
    )
    logger.addHandler(logging.StreamHandler())
    logging.getLogger('PIL').setLevel(logging.INFO)

    for k in vars(args):
        logger.info(f"{k}: {getattr(args, k)}")

    # ===== random seed init =====
    random.seed(args.seed+2024)
    np.random.seed(args.seed+6)
    torch.manual_seed(args.seed+15)

    # ===== gpu init =====
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    assert torch.cuda.device_count() == len(args.gpus), "set visible gpus error"


def train_model_init(args: Namespace) -> torch.nn.Module:
    if args.model_name == "pipeline_text_generater":
        model = PipelineTextGenerater(args.llm_dir, args.max_new_tokens, args.num_beams).to("cuda")
    elif args.model_name == "pipeline_image_generater":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
        torch.backends.cudnn.enabled = False
        model = PipelineImageGenerater(args.diffusion_dir).to("cuda")
    elif args.model_name == "joint_generater":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
        torch.backends.cudnn.enabled = False
        model = JointGenerater(args.llm_dir, args.max_new_tokens, args.num_beams, args.diffusion_dir).to("cuda")
    else:
        raise ValueError("model name error")

    return model


def dataset_init(args: Namespace) -> List[torch.utils.data.DataLoader]:
    train_dataloader, dev_dataloader, test_dataloader = None, None, None

    if args.model_name == "pipeline_text_generater":
        if hasattr(args, "train_data_path"):
            train_dataset = PipelineTextGeneraterDataset(dialogues_path=args.train_data_path)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.mini_batch_size, num_workers=2,
                sampler=DialogueLengthSampler(train_dataset.get_dialogues_length()),
                collate_fn=PipelineTextGeneraterDataset.collate_fn
            )
        if hasattr(args, "dev_data_path"):
            dev_dataloader = torch.utils.data.DataLoader(
                PipelineTextGeneraterDataset(dialogues_path=args.dev_data_path),
                batch_size=args.evaluate_batch_size, shuffle=False, drop_last=False, num_workers=2,
                collate_fn=PipelineTextGeneraterDataset.collate_fn
            )
        if hasattr(args, "test_data_path"):
            test_dataloader = torch.utils.data.DataLoader(
                PipelineTextGeneraterDataset(dialogues_path=args.test_data_path),
                batch_size=args.evaluate_batch_size, shuffle=False, drop_last=False, num_workers=2,
                collate_fn=PipelineTextGeneraterDataset.collate_fn
            )
    elif args.model_name == "pipeline_image_generater":
        if hasattr(args, "train_data_path"):
            train_dataset = PipelineImageGeneraterDataset(dialogues_path=args.train_data_path)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.mini_batch_size, num_workers=2,
                sampler=DialogueLengthSampler(train_dataset.get_images_description_length()),
                collate_fn=PipelineImageGeneraterDataset.collate_fn
            )
        if hasattr(args, "dev_data_path"):
            dev_dataloader = torch.utils.data.DataLoader(
                PipelineImageGeneraterDataset(dialogues_path=args.dev_data_path),
                batch_size=args.evaluate_batch_size, shuffle=False, drop_last=False, num_workers=2,
                collate_fn=PipelineImageGeneraterDataset.collate_fn
            )
        if hasattr(args, "test_data_path"):
            test_dataloader = torch.utils.data.DataLoader(
                PipelineImageGeneraterDataset(dialogues_path=args.test_data_path),
                batch_size=args.evaluate_batch_size, shuffle=False, drop_last=False, num_workers=2,
                collate_fn=PipelineImageGeneraterDataset.collate_fn
            )
    elif args.model_name == "joint_generater":
        if hasattr(args, "train_data_path"):
            train_dataset = JointGeneraterDataset(dialogues_path=args.train_data_path)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.mini_batch_size, num_workers=2,
                sampler=DialogueLengthSampler(train_dataset.get_dialogues_length()),
                collate_fn=JointGeneraterDataset.collate_fn
            )
        if hasattr(args, "dev_data_path"):
            dev_dataloader = torch.utils.data.DataLoader(
                JointGeneraterDataset(dialogues_path=args.dev_data_path),
                batch_size=args.evaluate_batch_size, shuffle=False, drop_last=False, num_workers=2,
                collate_fn=JointGeneraterDataset.collate_fn
            )
        if hasattr(args, "test_data_path"):
            test_dataloader = torch.utils.data.DataLoader(
                JointGeneraterDataset(dialogues_path=args.test_data_path),
                batch_size=args.evaluate_batch_size, shuffle=False, drop_last=False, num_workers=2,
                collate_fn=JointGeneraterDataset.collate_fn
            )
    else:
        raise ValueError("model name error")

    return train_dataloader, dev_dataloader, test_dataloader


def main():
    # ===== experiment init =====
    args = parse_args()
    pre_experiment_init(args)

    # ===== model init =====
    model = train_model_init(args)

    # ===== dataset init =====
    datasets = dataset_init(args)

    # ===== train/dev/test =====
    if args.subcommand == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        warmup_scheduler = LinearLR(optimizer, 0.01, 1.0, total_iters=args.warmup_steps)
        decay_scheduler = ReduceLROnPlateau(
            optimizer, mode="min" if args.metric[0] == "-" else "max",
            factor=0.5, patience=args.patience//3-1, min_lr=1e-9)

        trainer = Trainer(
            model, datasets, optimizer, warmup_scheduler, decay_scheduler,
            args.accumulation_steps, args.clip_grad_norm, args.metric, args.output_dir, args.epochs, args.patience
        )
        trainer.train()

    elif args.subcommand == "dev":
        model = model.load_model(args.model_path)
        evaluater = Evaluater(model, datasets[1], os.path.join(args.output_dir, "prediction.jsonl"))
        evaluater.evaluate()

    elif args.subcommand == "test":
        pass

    else:
        raise ValueError("subcommand error")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e
