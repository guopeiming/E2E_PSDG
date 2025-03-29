import os
import torch
import logging
from torch import nn
from typing import List, Dict, Tuple
import torch.nn.functional as F
from utils import VisualMetrics
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import B_IMG_DESC, E_IMG_DESC, LLAMA_CHAT_TEMPLATE, SYSTEM_PROMPT
from utils import (
    sequence_cross_entropy_with_logits, sanitize_user_prompt_token, sanitize_image_description,
    BLEU, ROUGE
)
logger = logging.getLogger(__name__)


class JointGeneraterWordPiece(nn.Module):

    def __init__(self, llm_dir: str, max_new_tokens: int, num_beams: int, diffusion_dir: str) -> None:
        super(JointGeneraterWordPiece, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, padding_side="left")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_dir, torch_dtype=torch.bfloat16)
        self.tokenizer_user_agp_ids = None
        self._b_img_desc_id = None
        self._e_img_desc_id = None
        self._system_prompt_len = None
        self._token_transform_matrix = None
        self._tokenizer_and_llm_init()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=32, lora_alpha=32, lora_dropout=0.1, bias="none", use_rslora=False,
            # target_modules=["q_proj", "k_proj", "v_proj"], modules_to_save=["embed_tokens", "lm_head"]
            target_modules=["gate_proj", "up_proj", "down_proj"], modules_to_save=["embed_tokens", "lm_head"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        train_params, all_param = self.llm.get_nb_trainable_parameters()
        logger.info(
            f"trainable params: {train_params:,d} || all params: {all_param:,d} || trainable%: {100 * train_params / all_param}")

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.bleu = BLEU()
        self.rouge = ROUGE()

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(diffusion_dir, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(diffusion_dir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.bfloat16, use_safetensors=True)
        self.vae = AutoencoderKL.from_pretrained(diffusion_dir, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16, use_safetensors=True)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(diffusion_dir, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(diffusion_dir, subfolder="unet", variant="fp16", torch_dtype=torch.bfloat16, use_safetensors=True)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            diffusion_dir, tokenizer=self.clip_tokenizer, text_encoder=self.text_encoder, vae=self.vae, unet=self.unet,
            torch_dtype=torch.bfloat16, safety_checker=None)

        self.visual_metrics = VisualMetrics(verbose=False)

    def forward(
        self, dialogues: List[List[Dict[str, str]]], responses: List[List[str]] = None, images: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        res = dict()

        if images is not None:
            images = images.to(torch.bfloat16).to("cuda")

        if self.training:
            dialogues_ids = self._tokenize_prompts(dialogues)
            llm_output = self.llm(
                input_ids=dialogues_ids["input_ids"],
                attention_mask=dialogues_ids["attention_mask"],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True
            )

            text_loss = sequence_cross_entropy_with_logits(
                llm_output["logits"][:, :-1, :].contiguous(),
                dialogues_ids["input_ids"][:, 1:].contiguous(),
                dialogues_ids["attention_mask"][:, 1:].contiguous(),
                average="token",
                # label_smoothing=self.label_smoothing
            )

            imgs_desc_ids, imgs_desc_mask = self._tokenize_images_description(dialogues_ids["input_ids"], llm_output["logits"])
            imgs_desc_ids = torch.matmul(imgs_desc_ids, self._token_transform_matrix)
            imgs_desc_embeddings = self._get_images_description_embedding(imgs_desc_ids, imgs_desc_mask)[0]

            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            noises = torch.randn_like(latents)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (images.size(0), ), dtype=torch.long, device=images.device)
            noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)

            noises_pred = self.unet(noisy_latents, timesteps, imgs_desc_embeddings, return_dict=True)["sample"]
            # noises_pred = self.unet(noisy_latents, timesteps, descriptions_embeddings, encoder_attention_mask=img_descs_mask, return_dict=True)["sample"]
            image_loss = F.mse_loss(noises_pred, noises, reduction="mean")

            res["loss"] = text_loss + image_loss

        else:
            if responses is not None:
                res_loss_tensors, res_utterances = [], [[] for _ in range(len(responses))]

                for i in range(1, max([len(resp) for resp in responses])):
                    dialogues_contexts, gold_utterances, dialogues_indexes = [], [], []
                    for j, (dial, resp) in enumerate(zip(dialogues, responses)):
                        if i < len(resp):
                            dialogues_contexts.append(dial[:i])
                            gold_utterances.append(resp[i])
                            dialogues_indexes.append(j)

                    dialogues_ids = self._tokenize_prompts(dialogues_contexts)
                    llm_output = self.llm.generate(
                        input_ids=dialogues_ids["input_ids"],
                        attention_mask=dialogues_ids["attention_mask"],
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.num_beams,
                        num_return_sequences=1,
                        do_sample=False,
                        top_p=None,
                        temperature=None,
                        # do_sample=True,
                        # top_p=0.9,
                        # temperature=0.8,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                    pred_utterances = self.tokenizer.batch_decode(
                        llm_output["sequences"][:, dialogues_ids["input_ids"].size(1):],
                        skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    pred_utterances = sanitize_user_prompt_token(pred_utterances)

                    res_loss_tensors.append(llm_output["sequences_scores"])
                    for k, resp in enumerate(pred_utterances):
                        res_utterances[dialogues_indexes[k]].append(resp)

                    pred_utterances_bare, pred_images_descs = sanitize_image_description(pred_utterances)
                    gold_utterances_bare, gold_images_descs = sanitize_image_description(gold_utterances)

                    self.bleu(pred_utterances_bare, gold_utterances_bare)
                    self.rouge(pred_utterances_bare, gold_utterances_bare)

                    pred_images = [pred_descs[0] for pred_descs in pred_images_descs if len(pred_descs) > 0]
                    pred_images = self.pipeline(pred_images, num_inference_steps=20, num_images_per_prompt=1, return_dict=True)["images"] if len(pred_images) > 0 else None
                    gold_images = [images[i] for i in range(len(gold_images_descs)) if len(gold_images_descs[i]) > 0]
                    gold_images = torch.stack(gold_images, dim=0) if len(gold_images) > 0 else None
                    self.visual_metrics(predicted_images=pred_images, gold_images=gold_images)

                res["loss"] = -torch.cat(res_loss_tensors).sum() / len(res_loss_tensors)
                res["utterances"] = res_utterances

            else:
                # TODO: test
                pass

        return res

    def _get_images_description_embedding(self, imgs_desc_ids: torch.Tensor, imgs_desc_mask: torch.Tensor):
        clip_text_transf = self.text_encoder.text_model
        if imgs_desc_ids.size(1) > clip_text_transf.config.max_position_embeddings:
            imgs_desc_ids = imgs_desc_ids[:, :clip_text_transf.config.max_position_embeddings, :]
            imgs_desc_mask = imgs_desc_mask[:, :clip_text_transf.config.max_position_embeddings]
        # code is from CLIPTextTransformer

        # hidden_states = clip_text_transf.embeddings(input_ids=input_ids, position_ids=position_ids)
        seq_length = imgs_desc_ids.size(1)
        position_ids = clip_text_transf.embeddings.position_ids[:, :seq_length]
        position_embeddings = clip_text_transf.embeddings.position_embedding(position_ids)
        inputs_embeds = torch.matmul(imgs_desc_ids, clip_text_transf.embeddings.token_embedding.weight)
        hidden_states = inputs_embeds + position_embeddings

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            hidden_states.shape[:-1], hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if imgs_desc_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = clip_text_transf.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = clip_text_transf.final_layer_norm(last_hidden_state)

        return (last_hidden_state, ) + encoder_outputs[1:]

    def _tokenize_images_description(self, dialogues_input_ids: torch.Tensor, llm_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b_img_desc_id_pos = torch.where(dialogues_input_ids[:, self._system_prompt_len:]==self._b_img_desc_id)
        e_img_desc_id_pos = torch.where(dialogues_input_ids[:, self._system_prompt_len:]==self._e_img_desc_id)
        assert (b_img_desc_id_pos[0] - e_img_desc_id_pos[0]).sum().item() == 0, "size error"
        assert (b_img_desc_id_pos[0] - torch.arange(dialogues_input_ids.size(0))).sum().item() == 0, "size error"

        img_desc_len = (e_img_desc_id_pos[1] - b_img_desc_id_pos[1]).max().item() - 1
        imgs_desc_ids_list, imgs_desc_mask_list = [], []
        for i, (b_pos, e_pos) in enumerate(zip(b_img_desc_id_pos[1], e_img_desc_id_pos[1])):
            img_desc_logit = llm_logits[i, b_pos+self._system_prompt_len+1: e_pos+self._system_prompt_len, :]
            img_desc_prob = F.softmax(img_desc_logit, dim=-1)

            _, img_desc_ids_index = img_desc_prob.max(dim=-1, keepdim=True)
            img_desc_ids = torch.zeros_like(img_desc_prob).scatter_(-1, img_desc_ids_index, 1.0)
            img_desc_ids = img_desc_ids - img_desc_prob.detach() + img_desc_prob
            img_desc_mask = img_desc_ids.new_ones(img_desc_ids.size(0), dtype=torch.long)

            img_desc_pad_ids = img_desc_ids.new_zeros((img_desc_len - img_desc_ids.size(0), img_desc_ids.size(1)))
            img_desc_pad_ids[:, self.tokenizer.pad_token_id] = 1.0
            img_desc_pad_mask = img_desc_pad_ids.new_zeros(img_desc_pad_ids.size(0), dtype=torch.long)
            img_desc_ids = torch.cat([img_desc_ids, img_desc_pad_ids], dim=0)
            img_desc_mask = torch.cat([img_desc_mask, img_desc_pad_mask], dim=0)

            imgs_desc_ids_list.append(img_desc_ids)
            imgs_desc_mask_list.append(img_desc_mask)

        imgs_desc_ids = torch.stack(imgs_desc_ids_list, dim=0)
        imgs_desc_mask = torch.stack(imgs_desc_mask_list, dim=0)

        return imgs_desc_ids, imgs_desc_mask

    def _tokenize_prompts(self, dialogues: List[List[Dict[str, str]]]) -> Dict[str, torch.Tensor]:
        if self.training:
            # self.tokenizer.padding_side = "right"
            agp = False
        else:
            # self.tokenizer.padding_side = "left"
            if dialogues[0][-1]["role"] == "user":
                agp = True
            elif dialogues[0][-1]["role"] == "assistant":
                agp = False
            else:
                raise ValueError("prompt role error")

        dialogues = [[{"role": "system", "content": SYSTEM_PROMPT}]+dial for dial in dialogues]

        dialogues_ids = self.tokenizer.apply_chat_template(
            dialogues, add_generation_prompt=agp, tokenize=True, padding=True, return_tensors="pt", return_dict=True)

        if (not self.training) and dialogues[0][-1]["role"] == "assistant":
            user_agp_ids = self.tokenizer_user_agp_ids.repeat(len(dialogues), 1)
            user_apg_mask = torch.ones_like(user_agp_ids)
            dialogues_ids["input_ids"] = torch.cat([dialogues_ids["input_ids"], user_agp_ids], dim=-1)
            dialogues_ids["attention_mask"] = torch.cat([dialogues_ids["attention_mask"], user_apg_mask], dim=-1)

        dialogues_ids["input_ids"] = dialogues_ids["input_ids"].to("cuda")
        dialogues_ids["attention_mask"] = dialogues_ids["attention_mask"].to("cuda")
        return dialogues_ids

    def _tokenizer_and_llm_init(self) -> None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

        add_token_num = self.tokenizer.add_tokens([B_IMG_DESC, E_IMG_DESC])
        self._b_img_desc_id = self.tokenizer.vocab_size - 2
        self._e_img_desc_id = self.tokenizer.vocab_size - 1
        assert add_token_num == 2, "add tokens error"
        assert self._b_img_desc_id == self.tokenizer(B_IMG_DESC, add_special_tokens=False), "b_img_desc_id error"
        assert self._e_img_desc_id == self.tokenizer(E_IMG_DESC, add_special_tokens=False), "e_img_desc_id error"
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

        system_prompt_ids = self.tokenizer.apply_chat_template(
            [[{"role": "system", "content": SYSTEM_PROMPT}]],
            add_generation_prompt=False, tokenize=True, padding=True, return_dict=True)
        self._system_prompt_len = len(system_prompt_ids["input_ids"])

        assert self.tokenizer("<s>[INST]", add_special_tokens=False)["input_ids"] == [1, 518, 25580, 29962], "agp error"
        self.tokenizer_user_agp_ids = torch.tensor([1, 518, 25580, 29962], dtype=torch.long)

        self._token_transform_matrix = torch.zeros((self.tokenizer.vocab_size, self.clip_tokenizer.vocab_size))
        llm_vocab = self.tokenizer.get_vocab()
        clip_vocab = self.clip_tokenizer.get_vocab()
        for k, v in llm_vocab.items():
            if k.startswith("▁"):
                k = k[1:]

            if (k in clip_vocab) and (k+"</w>" in clip_vocab):
                self._token_transform_matrix[v, clip_vocab[k]] = 0.5
                self._token_transform_matrix[v, clip_vocab[k+"</w>"]] = 0.5
            elif (k in clip_vocab) and (k+"</w>" not in clip_vocab):
                self._token_transform_matrix[v, clip_vocab[k]] = 1.0
            elif (k not in clip_vocab) and (k+"</w>" in clip_vocab):
                self._token_transform_matrix[v, clip_vocab[k+"</w>"]] = 1.0
            elif (k not in clip_vocab) and (k+"</w>" not in clip_vocab):
                pass

        self._token_transform_matrix[self.tokenizer.pad_token_id, self.clip_tokenizer.pad_token_id] = 1.0
        self._token_transform_matrix[self.tokenizer.bos_token_id, self.clip_tokenizer.bos_token_id] = 1.0
        self._token_transform_matrix[self.tokenizer.eos_token_id, self.clip_tokenizer.eos_token_id] = 1.0

    def save_model(self, output_dir: str) -> None:
        self.llm.save_pretrained(os.path.join(output_dir, "llm"))
        self.pipeline.save_pretrained(output_dir)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        bleu = self.bleu.get_metric(reset)
        rouge = self.rouge.get_metric(reset)
        metrics = {"BLEU-1": bleu["BLEU-1"], "BLEU-2": bleu["BLEU-2"], "ROUGE-L": rouge["ROUGE-L"]}
        metrics.update(self.visual_metrics.get_metric(reset))
        return metrics
