from typing import List, Tuple
import yake
from keybert import KeyBERT
from nltk import edit_distance
from rake_nltk import Rake
import pke


class KeywordExtractor(object):

    def __init__(self,  document: str, **kwargs):
        self._keyword_score_list = None
        self._document = document
        self.set_keyword_score_list(**kwargs)

    def filter_keywords_with_index(self, index):
        """
        check the present of keywords in index by the ratio of matching > 90% of the current keyword
        (if document break into list of words)
        """

        return [
            (keyword, score)
            for keyword, score in self._keyword_score_list
            if any(fuzzy_match(keyword, indexed_keyword) for indexed_keyword in index)
        ]

    def get_n_best(self, n: int = 10, index: List[str] = None) -> List[Tuple[str, float]]:
        """Given a document, return a ranking list of keywords
               Arguments:
                   document {str} -- the input document

               Returns:
                   List[dict] -- list of keywords
        """
        if index is not None:
            self._keyword_score_list = self.filter_keywords_with_index(index)

        return self._keyword_score_list[:n]

    def set_keyword_score_list(self, **kwargs):
        pass


class YakeExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        # language = kwargs.get('language', 'en')
        # max_ngram_size = 3
        # deduplication_thresold = 0.4
        # deduplication_algo = 'seqm'
        # windowSize = 1
        # numOfKeywords = len(self._document)
        # custom_kw_extractor = yake.KeywordExtractor(
        #     lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
        #     dedupFunc=deduplication_algo, windowsSize=windowSize,
        #     top=numOfKeywords, features=None
        # )
        # self._keyword_score_list = custom_kw_extractor.extract_keywords(self._document)
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.YAKE(),
            corpus=self._document,
            **kwargs,
        )


class TfIdfExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.TfIdf(),
            corpus=self._document,
            **kwargs,
        )


class TextRankExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.TextRank(),
            corpus=self._document,
            **kwargs,
        )


class SingleRankExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.SingleRank(),
            corpus=self._document,
            **kwargs,
        )


class TopicalPageRankExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.SingleRank(),
            corpus=self._document,
            **kwargs,
        )


class TopicRankExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.TopicRank(),
            corpus=self._document,
            **kwargs,
        )


class PositionRankExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.PositionRank(),
            corpus=self._document,
            **kwargs,
        )


class MultipartiteRankExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.MultipartiteRank(),
            corpus=self._document,
            **kwargs,
        )


class KPMinerExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        self._keyword_score_list = extract_with_pke(
            model=pke.unsupervised.KPMiner(),
            corpus=self._document,
            **kwargs,
        )


class KeyBertExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        stop_words = kwargs.get('stop_words', 'english')
        self._keyword_score_list = extractor.extract_keywords(self._document, keyphrase_ngram_range=(1, 4), stop_words=stop_words)[:len(self._document)]


class RakeExtractor(KeywordExtractor):
    def set_keyword_score_list(self, **kwargs):
        r = Rake(**kwargs)  # Uses stopwords for english from NLTK, and all puntuation characters.
        r.extract_keywords_from_text(self._document)
        self._keyword_score_list = [(v, k) for k, v in r.get_ranked_phrases_with_scores()][:len(self._document)]
        # To get keyword phrases ranked highest to lowest.


def extract_with_pke(model, corpus, **kwargs) -> list:
    # load the content of the document, here document is expected to be in raw
    # format (i.e. a simple text file) and preprocessing is carried out using spacy
    language = kwargs.get('language', 'en')
    normalization = kwargs.get('normalization', 'stemming')

    model.load_document(input=corpus, language=language, normalization=normalization)

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    model.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    model.candidate_weighting()

    # N-best selection, keyphrases contains the n best ranked candidates as
    # (keyphrase, score) tuples
    return model.get_n_best(n=len(model.candidates))


# Helper functions
def fuzzy_match(a: str, b: str, threshold=.9):
    """Returns true if the first string fuzzy matches the second.
    """
    dist = edit_distance(a, b)
    dist /= max(len(a), len(b))
    return (1.0 - dist) > threshold
