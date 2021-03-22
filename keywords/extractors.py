from typing import List
import yake
import nltk
from nltk.corpus import stopwords
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from rake_nltk import Rake
import pke

nltk.download('stopwords')
nltk.download('universal_tagset')


class KeywordExtractor(object):

    def __init__(self):
        self.kw_extractor = None

    def train(self, documents: List[str], **kwargs):
        """Unsupervised train the keyword extractor on a list of documents

        Arguments:
            documents {List[str]} -- [description]
        """
        pass

    def predict(self, text: str, topn: int = 10) -> List[dict]:
        """Given a text, return a ranking list of keywords
        Example:
        Input: text = "what is machine learning and what is the difference between deep learning and neural networks"
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
            text {str} -- the input text

        Returns:
            List[dict] -- list of keywords
        """
        pass

    def batch_predict(
            self,
            texts: List[str],
            topn: int = 100
    ) -> List[List[dict]]:
        """Same as predict but for batch of texts instead of a single text

        Arguments:
            texts {List[str]} -- [description]

        Keyword Arguments:
            topn {int} -- [description] (default: {100})

        Returns:
            List[List[dict]] -- [description]
        """

        return [self.predict(text, topn) for text in texts]


class TwoStepKeywordExtractor(KeywordExtractor):

    def __init__(self, n_gram, total_keywords_in_training, documents):
        self.n_gram = n_gram
        self.total_keywords_in_training = total_keywords_in_training
        self.the_total_keywords = []
        self.train(documents)

    # order the collected keywords by score
    def sent_keyword_selection(self, sent_keyword, n_keywords):
        tmp = sorted(sent_keyword, key=lambda e: e['score'], reverse=True)[:n_keywords]
        return tmp

    # check the present of keywords in sentence by the ratio of matching >90% of the current keyword (if pharse break into list of words)
    def match_keyword_in_sent_v2(self, KW, ST):
        sentence_keywords = []
        tmpdict = {}
        for keys in KW:
            tmpdict[keys[0]] = [keys[0].split(), keys[-1]]  # [ ['a','b','c'],0.5]
        tmp_KW = ''
        for sub_keys, sub_words in tmpdict.items():
            count1 = 0
            count2 = 0
            for tmp_words in sub_words[0]:
                if tmp_words in ST:
                    count1 += 1
                if tmp_words in tmp_KW:
                    count2 += 1
            if count1 / len(tmpdict[sub_keys][0]) >= 0.90 and count2 / len(tmpdict[sub_keys][0]) < 0.9:
                sentence_keywords.append({'keyword': sub_keys, 'score': sub_words[1]})
                tmp_KW = tmp_KW + ' ' + sub_keys
        return sentence_keywords

    # general processing function for all of the steps show up before
    def predict(self, text: str, topn: int = 10) -> List[dict]:
        """Given a text, return a ranking list of keywords
               Example:
               Input: text = "what is machine learning and what is the difference between deep learning and neural networks"
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
                   text {str} -- the input text

               Returns:
                   List[dict] -- list of keywords
               """

        sent_key_collection = self.match_keyword_in_sent_v2(self.the_total_keywords, text)
        tmp_final_collection = self.sent_keyword_selection(sent_key_collection, topn)

        return tmp_final_collection


class YakeExtractor(TwoStepKeywordExtractor):

    # order the collected keywords by score
    def sent_keyword_selection(self, sent_keyword, n_keywords):
        tmp = sorted(sent_keyword, key=lambda e: e['score'], reverse=False)[:n_keywords]
        return tmp

    def train(self, documents, **kwargs):
        """Unsupervised train the keyword extractor on a list of documents
        Arguments:
            documents {List[str]} -- [description]
        """

        total_data = ' '.join(documents)
        language = kwargs.get('language', 'en')
        max_ngram_size = self.n_gram
        deduplication_thresold = 0.4
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = self.total_keywords_in_training

        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                    dedupFunc=deduplication_algo, windowsSize=windowSize,
                                                    top=numOfKeywords, features=None)
        self.the_total_keywords = custom_kw_extractor.extract_keywords(total_data)


class TfIdfExtractor(TwoStepKeywordExtractor):

    # Function for sorting tf_idf in descending order
    @staticmethod
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @staticmethod
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        # use only topn items from vector
        sorted_items = sorted_items[:topn]

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
        corpus = ' '.join(documents)
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
        self.the_total_keywords = [(k, v) for k, v in TfIdfExtractor.extract_topn_from_vector(
            feature_names,
            sorted_items,
            self.total_keywords_in_training,
        ).items()]


class TextRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.TextRank(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class SingleRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.SingleRank(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class TopicalPageRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.SingleRank(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class TopicRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.TopicRank(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class PositionRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.PositionRank(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class MultipartiteRankExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.MultipartiteRank(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class KPMinerExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        self.the_total_keywords = extract_with_pke(
            extractor=pke.unsupervised.KPMiner(),
            corpus=' '.join(documents),
            top=self.total_keywords_in_training,
            **kwargs,
        )


class KeyBertExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        stop_words = kwargs.get('stop_words', 'english')
        self.the_total_keywords = extractor.extract_keywords(' '.join(documents),  stop_words=stop_words)[:self.total_keywords_in_training]


class RakeExtractor(TwoStepKeywordExtractor):
    def train(self, documents, **kwargs):
        r = Rake(**kwargs)  # Uses stopwords for english from NLTK, and all puntuation characters.
        r.extract_keywords_from_text(' '.join(documents))
        self.the_total_keywords = [(v, k) for k, v in r.get_ranked_phrases_with_scores()][:self.total_keywords_in_training]  # To get keyword phrases ranked highest to lowest.


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

