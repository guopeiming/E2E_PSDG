import torch
import logging
from torch import nn
from typing import List, Dict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import B_IMG_DESC, E_IMG_DESC, LLAMA_CHAT_TEMPLATE, SYSTEM_PROMPT
from utils import (
    sequence_cross_entropy_with_logits, sanitize_user_prompt_token, sanitize_image_description,
    BLEU, ROUGE
)
logger = logging.getLogger(__name__)


class PipelineTextGenerater(nn.Module):

    def __init__(self, llm_dir: str, max_new_tokens: int, num_beams: int) -> None:
        super(PipelineTextGenerater, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, padding_side="left")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_dir, torch_dtype=torch.bfloat16)
        self.tokenizer_user_agp_ids = None
        self._tokenizer_and_llm_init()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=256, lora_alpha=512, lora_dropout=0.0, bias="none", use_rslora=False,
            target_modules=["q_proj", "k_proj", "v_proj"], modules_to_save=["embed_tokens", "lm_head"]
            # target_modules=["gate_proj", "up_proj", "down_proj"], modules_to_save=["embed_tokens", "lm_head"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        train_params, all_param = self.llm.get_nb_trainable_parameters()
        logger.info(
            f"trainable params: {train_params:,d} || all params: {all_param:,d} || trainable%: {100 * train_params / all_param}")

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.bleu = BLEU()
        self.rouge = ROUGE()

    def forward(
        self, dialogues: List[List[Dict[str, str]]], responses: List[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        res = dict()
        if self.training:
            dialogues_ids = self._tokenize_prompts(dialogues)
            llm_output = self.llm(
                input_ids=dialogues_ids["input_ids"],
                attention_mask=dialogues_ids["attention_mask"],
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )

            res["loss"] = sequence_cross_entropy_with_logits(
                llm_output["logits"][:, :-1, :].contiguous(),
                dialogues_ids["input_ids"][:, 1:].contiguous(),
                dialogues_ids["attention_mask"][:, 1:].contiguous(),
                average="token",
                # label_smoothing=self.label_smoothing
            )

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
                        # do_sample=False,
                        # top_p=None,
                        # temperature=None,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.6,
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

                    pred_utterances_bare, _ = sanitize_image_description(pred_utterances)
                    gold_utterances_bare, _ = sanitize_image_description(gold_utterances)
                    self.bleu(pred_utterances_bare, gold_utterances_bare)
                    self.rouge(pred_utterances_bare, gold_utterances_bare)

                res["loss"] = -torch.cat(res_loss_tensors).sum() / len(res_loss_tensors)
                res["utterances"] = res_utterances

            else:
                # TODO: test
                pass

        return res

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
        self.llm.resize_token_embeddings(len(self.tokenizer))
        assert add_token_num == 2, "add tokens error"

        assert self.tokenizer("<s>[INST]", add_special_tokens=False)["input_ids"] == [1, 518, 25580, 29962], "agp error"
        self.tokenizer_user_agp_ids = torch.tensor([1, 518, 25580, 29962], dtype=torch.long)

        self.tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

    def save_model(self, output_dir: str) -> None:
        self.llm.save_pretrained(output_dir)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        bleu = self.bleu.get_metric(reset)
        rouge = self.rouge.get_metric(reset)
        metrics = {"BLEU-1": bleu["BLEU-1"], "BLEU-2": bleu["BLEU-2"], "ROUGE-L": rouge["ROUGE-L"]}
        return metrics
