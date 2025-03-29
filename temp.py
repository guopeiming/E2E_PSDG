import PIL.Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from typing import List
from config import B_IMG_DESC, E_IMG_DESC
import jsonlines
from tqdm import tqdm
import logging
import os
import re
import PIL
import torch_fidelity
import numpy as np
from datasets import load_dataset
from torchvision.transforms.v2.functional import resize, center_crop
from transformers import AutoTokenizer, CLIPTokenizer


def filter_dialogues() -> List[List[str]]:
    for dataset in ["train", "dev", "test"]:
        with jsonlines.open(f"data/photochat/dialogues/{dataset}_old.jsonl", "r") as reader, jsonlines.open(f"data/photochat/dialogues/{dataset}.jsonl", "w") as writer:
            for line in tqdm(reader):
                if line["photo_description"].startswith("The photo has"):
                    reg_output = re.search(r"Objects in the photo.*", line["photo_description"])
                    assert reg_output is not None, "reg error"
                    line["photo_description"] = reg_output.group()
                writer.write(line)


def check_dialogues() -> List[List[str]]:
    num, total = 0, 0
    for dialogue_path in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        dialogue_path = os.path.join("data/photochat/dialogues", dialogue_path)
        with jsonlines.open(dialogue_path, "r") as reader:
            for line in tqdm(reader):
                if line["photo_description"][:22] != "Objects in the photo: ":
                    print(line["photo_description"])
                    num += 1
                total += 1
                if len(line["photo_description"]) > 150:
                    print(line["photo_description"])
    print(num, total)


def llama2_download():
    snapshot_download(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        allow_patterns=["*.json", "*.model", "*.safetensors"],
        ignore_patterns=["pytorch_model.bin.index.json"],
        local_dir="./Llama-2-7b-chat-hf",
        local_dir_use_symlinks=False
    )


def stable_diffusion_download():
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
        allow_patterns=["*.txt", "*.json", "*.model", "*.fp16.safetensors"],
        local_dir="./stable-diffusion-3-medium-diffusers",
        local_dir_use_symlinks=False
    )


def stable_diffusion_download():
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-2-1-base",
        allow_patterns=["*.txt", "*.json", "*.model", "*.fp16.safetensors"],
        local_dir="../llm/stable-diffusion-2-1-base",
        local_dir_use_symlinks=False
    )


def crop_images():
    files = os.listdir("./data/photochat/temp2/")
    for file in files:
        img = PIL.Image.open("./data/photochat/temp2/"+file).convert("RGB")
        img = resize(img, 512)
        img = center_crop(img, 512)
        img.save("./data/photochat/temp2_crop/"+file[:-3]+"png")


def fid_test():
    fidelity_outputs = torch_fidelity.calculate_metrics(
        input1="./data/photochat/temp2_crop",
        # input2="./data/photochat/temp2_crop",
        cuda=True,
        batch_size=8,
        isc=True,
        isc_splits=10,
        # fid=True,
        samples_find_deep=False,
        verbose=True
    )


def vocab_test():
    llama_tokenizer = AutoTokenizer.from_pretrained("../llm/Llama-2-7b-chat-hf")
    clip_tokenizer = CLIPTokenizer.from_pretrained("../llm/stable-diffusion-2-1-base", subfolder="tokenizer")
    print("▁")
    print(len(llama_tokenizer), len(clip_tokenizer))


def download_dialogcc():
    dataset = load_dataset("passing2961/dialogcc", cache_dir="./data/dialogcc/")


def test_sequence_len():
    llama_tokenizer = AutoTokenizer.from_pretrained("../llm/Llama-2-7b-chat-hf")
    utters_len = []
    with jsonlines.open("./data/dialogcc/dialogues/dev_mini.jsonl", "r") as reader:
        for line in reader:
            user_id = line["dialogue"][0]["user_id"]
            utter_list = []
            for utter in line["dialogue"]:
                if user_id != utter["user_id"]:
                    utters_len.append(len(llama_tokenizer(" ".join(utter_list))["input_ids"]))
                    utter_list = []
                if utter["share_photo"]:
                    utter_list.append(utter["photo_caption"])
                else:
                    utter_list.append(utter["message"])
    print(sum(utters_len)/len(utters_len))
    utters_len = np.array(utters_len)
    utters_len.sort()


if __name__ == "__main__":
    # filter_dialogues()
    # check_dialogues()
    # lama2_download()
    # stable_diffusion_download()
    # crop_images()
    # fid_test()
    # vocab_test()
    # download_dialogcc()
    test_sequence_len()
    print(1)

