from collections import Counter
import math
from .utils import ngrams
from typing import Iterable, Tuple, Dict, List


class BLEU:
    """
    modified from allennlp: https://github.com/allenai/allennlp/blob/main/allennlp/training/metrics/bleu.py
    """

    def __init__(
        self,
        ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        self._ngram_weights = ngram_weights
        self._precision_matches: Dict[int, int] = Counter()
        self._precision_totals: Dict[int, int] = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    def reset(self) -> None:
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    def _get_precision_counts(
        self,
        predicted_tokens: List[str],
        reference_tokens: List[str],
        ngram_size: int,
    ) -> Tuple[int, int]:
        clipped_matches = 0
        total_predicted = 0
        predicted_ngram_counts = ngrams(predicted_tokens, ngram_size)
        reference_ngram_counts = ngrams(reference_tokens, ngram_size)

        for ngram, count in predicted_ngram_counts.items():
            clipped_matches += min(count, reference_ngram_counts[ngram])
            total_predicted += count
        return clipped_matches, total_predicted

    def _get_brevity_penalty(self) -> float:
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    def __call__(
        self,  # type: ignore
        predictions: List[str],
        references: List[str]
    ) -> None:
        for prediction, reference in zip(predictions, references):
            prediction = prediction.split()
            reference = reference.split()

            for ngram_size, _ in enumerate(self._ngram_weights, start=1):
                precision_matches, precision_totals = self._get_precision_counts(
                    prediction, reference, ngram_size
                )
                self._precision_matches[ngram_size] += precision_matches
                self._precision_totals[ngram_size] += precision_totals

            self._prediction_lengths += len(prediction)
            self._reference_lengths += len(reference)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()

        bleu_score_list = []
        brevity_penalty = self._get_brevity_penalty()
        for i, weight in enumerate(self._ngram_weights, start=1):
            ngram_scores = math.log(self._precision_matches[i] + 1e-13) - math.log(self._precision_totals[i] + 1e-13)
            bleu_score = brevity_penalty * math.exp(ngram_scores) * 100.
            metrics[f"BLEU-{i}"] = bleu_score
            bleu_score_list.append(weight * bleu_score)

        metrics["BLEU"] = brevity_penalty * math.exp(sum(bleu_score_list))

        if reset:
            self.reset()

        return metrics
