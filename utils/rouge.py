from collections import Counter
from typing import Tuple, Dict, List
from .utils import ngrams


class ROUGE:
    """
    modified from allennlp: https://github.com/allenai/allennlp/blob/main/allennlp/training/metrics/rouge.py
    """

    def __init__(self, ngram_size: int = 2) -> None:
        self._ngram_size = ngram_size
        self._recall_matches: Dict[int, int] = Counter()
        self._recall_totals: Dict[int, int] = Counter()

        self._total_rouge_l_f1 = 0.0
        self._total_sequence_count = 0

    def reset(self) -> None:
        self._recall_matches = Counter()
        self._recall_totals = Counter()

        self._total_rouge_l_f1 = 0.0
        self._total_sequence_count = 0

    def _longest_common_subsequence(self, seq_1: List[str], seq_2: List[str]) -> int:
        m = len(seq_1)
        n = len(seq_2)

        # Slightly lower memory usage by iterating over the longer sequence in outer loop
        # and storing previous lcs for the shorter sequence
        if m < n:
            seq_1, seq_2 = seq_2, seq_1
            m, n = n, m

        prev_lcs = [0 for _ in range(n+1)]
        for i in range(1, m+1, 1):

            cur_lcs = [0 for _ in range(n+1)]
            for j in range(1, n+1, 1):
                if seq_1[i-1] == seq_2[j-1]:
                    cur_lcs[j] = prev_lcs[j-1] + 1
                else:
                    cur_lcs[j] = max(cur_lcs[j-1], prev_lcs[j])
            prev_lcs = cur_lcs

        return prev_lcs[-1]

    def _get_rouge_l_score(
        self, predicted_tokens: List[str], reference_tokens: List[str]
    ) -> float:
        lcs = self._longest_common_subsequence(predicted_tokens, reference_tokens)
        if lcs == 0:
            f1 = 0.0
        else:
            recall_lcs = lcs / len(predicted_tokens)
            precision_lcs = lcs / len(reference_tokens)
            f1 = 2 * recall_lcs * precision_lcs / (recall_lcs + precision_lcs)
        return f1

    def _get_recall_counts(
        self,
        predicted_tokens: List[str],
        reference_tokens: List[str],
        ngram_size: int,
    ) -> Tuple[int, int]:
        clipped_matches = 0
        total_reference = 0
        predicted_ngram_counts = ngrams(predicted_tokens, ngram_size)
        reference_ngram_counts = ngrams(reference_tokens, ngram_size)

        for ngram, count in reference_ngram_counts.items():
            clipped_matches += min(count, predicted_ngram_counts[ngram])
            total_reference += count
        return clipped_matches, total_reference

    def __call__(
        self,  # type: ignore
        predictions: List[str],
        references: List[str],
    ) -> None:
        for prediction, reference in zip(predictions, references):
            prediction = prediction.split()
            reference = reference.split()

            # ROUGE-N
            for n in range(1, self._ngram_size + 1):
                recall_matches, recall_totals = self._get_recall_counts(prediction, reference, n)
                self._recall_matches[n] += recall_matches
                self._recall_totals[n] += recall_totals

            # ROUGE-L
            self._total_rouge_l_f1 += self._get_rouge_l_score(prediction, reference)

        self._total_sequence_count += len(predictions)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()

        # ROUGE-N
        for i in range(1, self._ngram_size+1):
            metrics[f"ROUGE-{i}"] = 0.0 if self._recall_totals[i] == 0 else self._recall_matches[i] / self._recall_totals[i] * 100.

        # ROUGE-L
        metrics["ROUGE-L"] = 0.0 if self._total_sequence_count == 0 else self._total_rouge_l_f1 / self._total_sequence_count * 100.

        if reset:
            self.reset()

        return metrics

