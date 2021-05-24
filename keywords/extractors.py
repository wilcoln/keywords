from typing import List
import yake
from nltk.corpus import stopwords
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from rake_nltk import Rake
import pke
from google.cloud import language_v1


class KeywordExtractor(object):

    def train(self, documents: List[str], **kwargs):
        """Unsupervised training of the keyword extractor on a list of documents

        Arguments:
            documents {List[str]} -- [description]
        """
        pass

    def predict(self, document: str, top: int = 10) -> List[dict]:
        """Given a document, return a ranking list of keywords
        Example:
        Input: document = "what is machine learning and what is the difference between deep learning and neural networks"
        Output: [
            {
                "keyword": "machine learning",
                "score": "0.353"
            },
            {
                "keyword": "deep learning",
                "score": "0.21"
            },
            ...
        ]

        Arguments:
            document {str} -- the input document

        Returns:
            List[dict] -- list of keywords
        """
        pass

    def batch_predict(self, documents: List[str], top: int = 100) -> List[List[dict]]:
        """Same as predict but for batch of documents instead of a single document

        Arguments:
            documents {List[str]} -- [description]

        Keyword Arguments:
            top {int} -- [description] (default: {100})

        Returns:
            List[List[dict]] -- [description]
        """

        return [self.predict(document, top) for document in documents]


class TwoStepKeywordExtractor(KeywordExtractor):

    def __init__(self, n_gram, keywords_index_size, documents):
        self.n_gram = n_gram
        self.keywords_index_size = keywords_index_size
        self.keywords_index = []
        self.train(documents)

    @staticmethod
    def document_keywords_selection(document_keywords, top):
        """
        order the collected keywords by score.

        :param document_keywords:
        :param top:
        :return:
        """
        tmp = sorted(document_keywords, key=lambda e: e['score'], reverse=True)[:top]
        return tmp

    def match_keywords_in_document(self, document):
        """
        check the present of keywords in document by the ratio of matching >90% of the current keyword
        (if document break into list of words)
        :param document:
        :return:
        """

        return [
            {'keyword': kw, 'score': score}
            for kw, score in self.keywords_index
            if len(set(kw.split()) & set(document.split())) / len(kw.split()) >= 0.90
        ]

    # general processing function for all of the steps show up before
    def predict(self, document: str, top: int = 10) -> List[dict]:
        """Given a document, return a ranking list of keywords
               Example:
               Input: document = "what is machine learning and what is the difference between deep learning and neural networks"
               Output: [
                   {
                       "keyword": "machine learning",
                       "score": "0.353"
                   },
                   {
                       "keyword": "deep learning",
                       "score": "0.21"
                   },
                   ...
               ]

               Arguments:
                   document {str} -- the input document

               Returns:
                   List[dict] -- list of keywords
               """

        document_keywords = self.match_keywords_in_document(document)
        return self.document_keywords_selection(document_keywords, top)


class YakeExtractor(TwoStepKeywordExtractor):

    # order the collected keywords by score
    def document_keywords_selection(self, document_keywords, top):
        return sorted(document_keywords, key=lambda e: e['score'], reverse=False)[:top]

    def train(self, documents, **kwargs):
        """Unsupervised train the keyword extractor on a list of documents
        Arguments:
            documents {List[str]} -- [description]
        """
        language = kwargs.get('language', 'en')
        max_ngram_size = self.n_gram
        deduplication_thresold = 0.4
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = self.keywords_index_size

        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                    dedupFunc=deduplication_algo, windowsSize=windowSize,
                                                    top=numOfKeywords, features=None)
        self.keywords_index = custom_kw_extractor.extract_keywords('\n'.join(documents))


class TfIdfExtractor(TwoStepKeywordExtractor):

    # Function for sorting tf_idf in descending order
    @staticmethod
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @staticmethod
    def extract_top_from_vector(feature_names, sorted_items, top=10):
        """get the feature names and tf-idf score of top n items"""

        # use only top items from vector
        sorted_items = sorted_items[:top]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature,score
        # results = zip(feature_vals,score_vals)
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

    def train(self, documents, **kwargs):
        corpus = '\n'.join(documents)
        language = kwargs.get('language', 'english')
        stop_words = set(stopwords.words(language))
        cv = CountVectorizer(max_df=0.8,  # set up the document frequency threshold
                             stop_words=stop_words,
                             max_features=10000,
                             ngram_range=(1, 4))  # bi-gram and tri-gram
        X = cv.fit_transform(corpus.split('\n'))
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        tf_idf_vector = tfidf_transformer.transform(cv.transform(corpus.split('\n')))

        # sort the tf-idf vectors by descending order of scores
        sorted_items = TfIdfExtractor.sort_coo(tf_idf_vector.tocoo())

        feature_names = cv.get_feature_names()
        self.keywords_index = [(k, v) for k, v in TfIdfExtractor.extract_top_from_vector(
            feature_names,
            sorted_items,
            self.keywords_index_size,
        ).items()]


class TextRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.TextRank(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class SingleRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.SingleRank(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class TopicalPageRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.SingleRank(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class TopicRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.TopicRank(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class PositionRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.PositionRank(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class MultipartiteRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.MultipartiteRank(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class KPMinerExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.keywords_index = extract_with_pke(
            extractor=pke.unsupervised.KPMiner(),
            corpus='\n'.join(documents),
            top=self.keywords_index_size,
            **kwargs,
        )


class KeyBertExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        stop_words = kwargs.get('stop_words', 'english')
        self.keywords_index = extractor.extract_keywords('\n'.join(documents),  stop_words=stop_words)[:self.keywords_index_size]


class RakeExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        r = Rake(**kwargs)  # Uses stopwords for english from NLTK, and all puntuation characters.
        r.extract_keywords_from_text('\n'.join(documents))
        self.keywords_index = [(v, k) for k, v in r.get_ranked_phrases_with_scores()][:self.keywords_index_size]
        # To get keyword phrases ranked highest to lowest.


def extract_with_pke(extractor, corpus, top, **kwargs) -> list:
    # load the content of the document, here document is expected to be in raw
    # format (i.e. a simple text file) and preprocessing is carried out using spacy
    language = kwargs.get('language', 'en')
    normalization = kwargs.get('normalization', 'stemming')

    extractor.load_document(input=corpus, language=language, normalization=normalization)

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    return extractor.get_n_best(n=top)


class GoogleCloudExtractor(KeywordExtractor):
    def predict(self, document: str, top: int = 10) -> List[str]:
        # Instantiates a client
        client = language_v1.LanguageServiceClient()

        document = language_v1.Document(content=document, type_=language_v1.Document.Type.PLAIN_TEXT)
        entities = client.analyze_entities(request={'document': document}).entities
        # TODO extract keyword list from entities
        return entities[:top]
        #GOOGLE_APPLICATION_CREDENTIALS
        #jobs.ai.institute@gmail.com /
        #!Lxk?d@$J4zH6jby