from typing import List

from keywords.extractors import KeywordExtractor


def average_precision_k(
        actual: List[str],
        predicted: List[str],
        k: int
) -> float:
    pass


def apk(
        actual: List[str],
        predicted: List[str],
        k: int
) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def recall(
        actual: List[str],
        predicted: List[str],
        k: int
) -> float:
    pass


class Evaluator(object):
    """
    The job of an evaluator is to evaluate different Keyword Extraction algorithm
    on a specific dataset
    """

    def __init__(self, dataset_name: str):
        """Load the dataset into this Evaluator

        Arguments:
            object {[type]} -- [description]
            dataset_name {str} -- [description]
        """
        self.texts = []  # List of strings
        self.keywords = []  # List of list of strings (keywords)
        pass

    def evaluate(
            self,
            keyword_extractor: KeywordExtractor,
            metric: str = 'precision_k',
            topn: int = 100
    ) -> float:
        """Evaluate a keyword extractor with a specific metric

        Arguments:
            keyword_extractor {KeywordExtractor} -- [description]

        Keyword Arguments:
            metric {str} -- [description] (default: {'Precision'})
            topn {int} -- [description] (default: {100})

        Returns:
            float -- [description]
        """
        results = []
        for t, kws in zip(self.texts, self.keywords):
            preds = keyword_extractor.predict(t)
            precision = apk(kws, preds, topn)
            results.append(precision)
        return sum(results)/len(results)