import datasets
import re
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def p_r_f1_score(prediction, ground_truth):
    prediction_tokens = prediction
    ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return [precision, recall, f1]


_CITATION = """\
N/A
"""

_DESCRIPTION = """\
N/A
"""

_KWARGS_DESCRIPTION = """
Compute GLUE evaluation metric associated to each GLUE dataset.
Args:
    predictions: list of predictions to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
Returns: depending on the GLUE subset, one or several of:
    "accuracy": Accuracy
    "f1": F1 score
    "pearson": Pearson Correlation
    "spearmanr": Spearman Correlation
    "matthews_correlation": Matthew Correlation
Examples:
    >>> glue_metric = datasets.load_metric('glue', 'sst2')  # 'sst2' or any of ["mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0}
    >>> glue_metric = datasets.load_metric('glue', 'mrpc')  # 'mrpc' or 'qqp'
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0, 'f1': 1.0}
    >>> glue_metric = datasets.load_metric('glue', 'stsb')
    >>> references = [0., 1., 2., 3., 4., 5.]
    >>> predictions = [0., 1., 2., 3., 4., 5.]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print({"pearson": round(results["pearson"], 2), "spearmanr": round(results["spearmanr"], 2)})
    {'pearson': 1.0, 'spearmanr': 1.0}
    >>> glue_metric = datasets.load_metric('glue', 'cola')
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'matthews_correlation': 1.0}
"""


def simple_accuracy(preds, labels):
    return (preds == labels).mean().item()



@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class dstc10_intent_metric(datasets.Metric):
    def _info(self):
        if self.config_name not in [
            "dstc10"
        ]:
            raise KeyError(
                "Not dstc10_intent Task."
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64" if self.config_name != "stsb" else "float32"),
                    "references": datasets.Value("int64" if self.config_name != "stsb" else "float32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        if self.config_name == "dstc10":
            scores = p_r_f1_score(predictions, references)
            acc = simple_accuracy(predictions, references)
            return {"precision": scores[0], "recall": scores[1], "f1": scores[2], "accuracy": acc}
        else:
            raise KeyError(
                "Not dstc10_intent Task."
            )
