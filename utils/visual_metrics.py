import torch
from typing import Dict, List, Optional
import tempfile
import PIL.Image
import torch_fidelity
from torch_fidelity import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD, KEY_METRIC_FID


class VisualMetrics:

    def __init__(self, verbose: bool = False) -> None:
        self._tempdir_pred = None
        self._tempdir_gold = None
        self._verbose = verbose

    def reset(self) -> None:
        if self._tempdir_pred is not None:
            self._tempdir_pred.cleanup()
            self._tempdir_gold.cleanup()

            self._tempdir_pred = None
            self._tempdir_gold = None

    def __call__(
        self,  # type: ignore
        predicted_images: Optional[List[PIL.Image.Image]] = None,
        gold_images: Optional[torch.Tensor] = None,
    ) -> None:

        if predicted_images is not None:
            if self._tempdir_pred is None:
                self._tempdir_pred = tempfile.TemporaryDirectory(prefix="gpmtmp_pred_", dir="./")

            for pred in predicted_images:
                tmpfile = tempfile.mkstemp(suffix=".png", prefix="pred_", dir=self._tempdir_pred.name, text=False)
                pred.save(tmpfile[1])

        if gold_images is not None:
            if self._tempdir_gold is None:
                self._tempdir_gold = tempfile.TemporaryDirectory(prefix="gpmtmp_gold_", dir="./")

            gold_images = (gold_images.cpu().permute(0, 2, 3, 1).to(torch.float32).numpy() / 2 + 0.5).clip(0, 1)
            gold_images = (gold_images * 255).round().astype("uint8")
            gold_images = [PIL.Image.fromarray(image) for image in gold_images]
            for gold in gold_images:
                tmpfile = tempfile.mkstemp(suffix=".png", prefix="gold_", dir=self._tempdir_gold.name, text=False)
                gold.save(tmpfile[1])

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if (self._tempdir_pred is not None) and (self._tempdir_gold is not None):

            fidelity_outputs = torch_fidelity.calculate_metrics(
                input1=self._tempdir_pred.name,
                input2=self._tempdir_gold.name,
                cuda=True,
                batch_size=32,
                isc=True,
                isc_splits=10,
                fid=True,
                samples_find_deep=False,
                verbose=self._verbose
            )

            metrics = {
                "IS": fidelity_outputs[KEY_METRIC_ISC_MEAN],
                "IS_STD": fidelity_outputs[KEY_METRIC_ISC_STD],
                "FID": fidelity_outputs[KEY_METRIC_FID]
            }

        else:
            metrics = dict()

        if reset:
            self.reset()

        return metrics

