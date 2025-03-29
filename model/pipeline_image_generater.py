import torch
import logging
from torch import nn
from typing import List, Dict
import torch.nn.functional as F
from utils import VisualMetrics
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
logger = logging.getLogger(__name__)


class PipelineImageGenerater(nn.Module):

    def __init__(self, diffusion_dir: str) -> None:
        super(PipelineImageGenerater, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(diffusion_dir, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(diffusion_dir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.bfloat16, use_safetensors=True)
        self.vae = AutoencoderKL.from_pretrained(diffusion_dir, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16, use_safetensors=True)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(diffusion_dir, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(diffusion_dir, subfolder="unet", variant="fp16", torch_dtype=torch.bfloat16, use_safetensors=True)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            diffusion_dir, tokenizer=self.tokenizer, text_encoder=self.text_encoder, vae=self.vae, unet=self.unet,
            torch_dtype=torch.bfloat16, safety_checker=None)

        self.metrics = VisualMetrics(verbose=False)

    def forward(self, descriptions: List[str], images: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if images is not None:
            images = images.to(torch.bfloat16).to("cuda")
        res = dict()

        if self.training:
            descriptions_ids = self._tokenize_descriptions(descriptions)
            descriptions_embeddings = self.text_encoder(**descriptions_ids, return_dict=True)["last_hidden_state"]

            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            noises = torch.randn_like(latents)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (images.size(0), ), dtype=torch.long, device=images.device)
            noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)

            noises_pred = self.unet(noisy_latents, timesteps, descriptions_embeddings, encoder_attention_mask=descriptions_ids["attention_mask"], return_dict=True)["sample"]
            # noises_pred = self.unet(noisy_latents, timesteps, descriptions_embeddings, return_dict=True)["sample"]
            loss = F.mse_loss(noises_pred, noises, reduction="mean")

            res["loss"] = loss

        else:
            if images is not None:
                pred_images = self.pipeline(
                    prompt=descriptions, height=512, width=512,
                    num_inference_steps=20, guidance_scale=7.5, num_images_per_prompt=1, return_dict=True)["images"]
                self.metrics(pred_images, images)
                res["loss"] = torch.tensor(0.)
            else:
                # TODO: test
                pass

        return res

    def _tokenize_descriptions(self, descriptions: List[str]) -> Dict[str, torch.Tensor]:
        descriptions_ids = self.tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
        descriptions_ids["input_ids"] = descriptions_ids["input_ids"].to("cuda")
        descriptions_ids["attention_mask"] = descriptions_ids["attention_mask"].to("cuda")

        return descriptions_ids

    def save_model(self, output_dir: str) -> None:
        self.pipeline.save_pretrained(output_dir)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.metrics.get_metric(reset)
        return metrics
