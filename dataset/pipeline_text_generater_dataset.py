import jsonlines
from typing import List, Dict, Union, Iterator
from torch.utils.data import Dataset
from tqdm import tqdm
from config import B_IMG_DESC, E_IMG_DESC


class PipelineTextGeneraterDataset(Dataset):

    def __init__(self, dialogues_path: str) -> None:
        super(PipelineTextGeneraterDataset, self).__init__()
        self._dialogues: List[List[str]] = []
        self._load_dialogues(dialogues_path)

    def __len__(self) -> int:
        return len(self._dialogues)

    def __getitem__(self, index: int) -> Dict[str, Union[List[Dict[str, str]], List[str]]]:
        dialogue_with_role, dialogue_str = [], []
        dialogue = self._dialogues[index]
        for i, utterance in enumerate(dialogue):
            role = "user" if i % 2 == 0 else "assistant"
            dialogue_with_role.append({"role": role, "content": utterance})
            dialogue_str.append(utterance)
        return {"dialogue": dialogue_with_role, "response": dialogue_str}

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[List[Dict[str, str]], List[str]]]]
    ) -> Dict[str, List[Union[List[Dict[str, str]], List[str]]]]:
        return {
            "dialogues": [inst["dialogue"] for inst in batch],
            "responses": [inst["response"] for inst in batch]
        }

    def _load_dialogues(self, dialogues_path: str) -> None:
        with jsonlines.open(dialogues_path, "r") as reader:
            for line in tqdm(reader):
                dialogue = []
                for utterance_list in self.utterances_iterator(line["dialogue"]):
                    dialogue.append(" ".join(utterance_list))
                self._dialogues.append(dialogue)

    def utterances_iterator(self, dialogue: List[Dict[str, Union[str, bool, int]]]) -> Iterator[List[str]]:
        user_id = dialogue[0]["user_id"]
        utterance_list = []
        for utterance in dialogue:
            if user_id != utterance["user_id"]:
                yield utterance_list

                user_id = utterance["user_id"]
                utterance_list = []

            if utterance["share_photo"]:
                utterance_list.append(B_IMG_DESC + " " + utterance["photo_caption"] + " " + E_IMG_DESC)
            else:
                utterance_list.append(utterance["message"])
        return utterance_list

    def get_dialogues_length(self) -> List[int]:
        return [len(" ".join(dial).split()) for dial in self._dialogues]
