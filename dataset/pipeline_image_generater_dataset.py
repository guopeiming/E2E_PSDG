import torch
import jsonlines
from typing import List, Dict, Union
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import (
    Compose, Resize, InterpolationMode, CenterCrop, RandomHorizontalFlip, ToDtype, Normalize)


class PipelineImageGeneraterDataset(Dataset):

    def __init__(self, dialogues_path: str) -> None:
        super(PipelineImageGeneraterDataset, self).__init__()
        self._images_description: List[str] = []
        self._images_dir = os.path.normpath(os.path.join(os.path.dirname(dialogues_path), "../images"))
        self._images_filename: List[str] = []
        self._load_images(dialogues_path)

        self._image_transforms = Compose([
            Resize(512, interpolation=InterpolationMode.BILINEAR, antialias=True),
            CenterCrop(512),
            RandomHorizontalFlip(p=0.5),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self._images_description)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        desc = self._images_description[index]
        img = read_image(os.path.join(self._images_dir, self._images_filename[index]), mode=ImageReadMode.RGB)
        img = self._image_transforms(img)
        return {"description": desc, "image": img}

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[str, torch.Tensor]]]
    ) -> Dict[str, List[Union[str, torch.Tensor]]]:
        return {
            "descriptions": [inst["description"] for inst in batch],
            "images": torch.stack([inst["image"] for inst in batch], dim=0)
        }

    def _load_images(self, dialogues_path: str) -> None:
        with jsonlines.open(dialogues_path, "r") as reader:
            for line in tqdm(reader):
                for utterance in line["dialogue"]:
                    if utterance["share_photo"]:
                        image_filename = f'{utterance["photo_id"]}.jpg'
                        assert os.path.exists(os.path.join(self._images_dir, image_filename)), "image missing"
                        self._images_description.append(utterance["photo_caption"])
                        self._images_filename.append(image_filename)

    def get_images_description_length(self) -> List[int]:
        return [len(desc.split()) for desc in self._images_description]
