import math
from types import SimpleNamespace
import pytest
import torch
from serving.perplexity import evaluate_tokens


class UniformModel:
    def eval(self):
        return self

    def __call__(self, input_ids, labels):
        logits = torch.zeros((*input_ids.shape, 10))
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


@pytest.mark.parametrize("length,stride", [(2, 4), (7, 4), (17, 4), (32, 7)])
def test_each_causal_target_counted_once(length, stride):
    result = evaluate_tokens(UniformModel(), [i % 10 for i in range(length)], window=8, stride=stride)
    assert result["scored_tokens"] == length - 1
    assert result["perplexity"] == pytest.approx(10)


def test_no_overlap_rejected():
    with pytest.raises(ValueError):
        evaluate_tokens(UniformModel(), [1, 2], window=8, stride=8)
