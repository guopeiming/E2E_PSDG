import os
import torch
import jsonlines
from typing import List, Dict, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from config import B_IMG_DESC, E_IMG_DESC
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import (
    Compose, Resize, InterpolationMode, CenterCrop, RandomHorizontalFlip, ToDtype, Normalize)


class JointGeneraterDataset(Dataset):

    def __init__(self, dialogues_path: str) -> None:
        super(JointGeneraterDataset, self).__init__()
        self._dialogues: List[List[str]] = []
        self._images_filename: List[str] = []
        self._images_dir: str = os.path.normpath(os.path.join(os.path.dirname(dialogues_path), "../images"))
        self._load_dialogues_and_images(dialogues_path)

        self._image_transforms = Compose([
            Resize(512, interpolation=InterpolationMode.BILINEAR, antialias=True),
            CenterCrop(512),
            RandomHorizontalFlip(p=0.5),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self._dialogues)

    def __getitem__(self, index: int) -> Dict[str, Union[List[Dict[str, str]], List[str], torch.Tensor]]:
        dialogue_with_role, dialogue_str = [], []
        dialogue = self._dialogues[index]
        for i, utterance in enumerate(dialogue):
            role = "user" if i % 2 == 0 else "assistant"
            dialogue_with_role.append({"role": role, "content": utterance})
            dialogue_str.append(utterance)

        img = read_image(os.path.join(self._images_dir, self._images_filename[index]), mode=ImageReadMode.RGB)
        img = self._image_transforms(img)

        return {"dialogue": dialogue_with_role, "response": dialogue_str, "image": img}

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[List[Dict[str, str]], List[str], torch.Tensor]]]
    ) -> Dict[str, List[Union[List[Dict[str, str]], List[str], torch.Tensor]]]:
        return {
            "dialogues": [inst["dialogue"] for inst in batch],
            "responses": [inst["response"] for inst in batch],
            "images": torch.stack([inst["image"] for inst in batch], dim=0)
        }

    def _load_dialogues_and_images(self, dialogues_path: str) -> None:
        with jsonlines.open(dialogues_path, "r") as reader:
            for line in tqdm(reader):
                dialogue = []

                utterance_str, user_id = [], line["dialogue"][0]["user_id"]
                for utterance in line["dialogue"]:
                    if user_id != utterance["user_id"]:
                        dialogue.append(" ".join(utterance_str))
                        user_id = utterance["user_id"]
                        utterance_str = []

                    if utterance["share_photo"]:
                        utterance_str.append(B_IMG_DESC + " " + utterance["photo_caption"] + " " + E_IMG_DESC)

                        image_filename = f'{line["photo_id"]}.jpg'
                        assert os.path.exists(os.path.join(self._images_dir, image_filename)), "image missing"
                        self._images_filename.append(image_filename)
                    else:
                        utterance_str.append(utterance["message"])
                dialogue.append(" ".join(utterance_str))

                self._dialogues.append(dialogue)

    def get_dialogues_length(self) -> List[int]:
        return [len(" ".join(dial).split()) for dial in self._dialogues]
