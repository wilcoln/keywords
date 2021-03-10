import yake
import nltk
import spacy
from nltk.corpus import stopwords
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from rake_nltk import Rake
from gensim.summarization import keywords as gs_keywords
from google.cloud import language_v1
import pke

nltk.download('stopwords')
nltk.download('universal_tagset')


class KeywordExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        pass


class YakeExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        kw_extractor = yake.KeywordExtractor(top=top, **kwargs)
        extracted_keywords = kw_extractor.extract_keywords(corpus)

        return [word for word, _ in extracted_keywords]


class TfIdfExtractor(KeywordExtractor):
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

    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        stop_words = set(stopwords.words("english"))
        cv = CountVectorizer(max_df=0.8,  # set up the document frequency threshold
                             stop_words=stop_words,
                             max_features=10000,
                             ngram_range=(1, 3))  # bi-gram and tri-gram
        X = cv.fit_transform(corpus.split('\n'))
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        tf_idf_vector = tfidf_transformer.transform(cv.transform(corpus.split('\n')))

        # sort the tf-idf vectors by descending order of scores
        sorted_items = TfIdfExtractor.sort_coo(tf_idf_vector.tocoo())

        feature_names = cv.get_feature_names()
        # extract only the top n; n here is 10
        keywords = TfIdfExtractor.extract_topn_from_vector(feature_names, sorted_items, top)
        return [word for word in keywords]


class TextRankExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.TextRank(), corpus=corpus, top=top, **kwargs)


class SingleRankExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.SingleRank(), corpus=corpus, top=top, **kwargs)


class TopicalPageRankExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.TopicalPageRank(), corpus=corpus, top=top, **kwargs)


class TopicRankExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.TopicRank(), corpus=corpus, top=top, **kwargs)


class PositionRankExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.PositionRank(), corpus=corpus, top=top, **kwargs)


class MultipartiteRankExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.MultipartiteRank(), corpus=corpus, top=top, **kwargs)


class KeaExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.supervised.Kea(), corpus=corpus, top=top, **kwargs)


class WINGNUSExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.supervised.WINGNUS(), corpus=corpus, top=top, **kwargs)


class SpacyExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        nlp = spacy.load("en_core_web_sm")
        return list(set([str(ent) for ent in nlp(corpus).ents]))[:top]


class KPMinerExtractor:
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return extract_with_pke(extractor=pke.unsupervised.KPMiner(), corpus=corpus, top=top, **kwargs)


class GensimExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        return gs_keywords(corpus, split=True)[:top]


class GoogleCloudExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        # Instantiates a client
        client = language_v1.LanguageServiceClient()

        document = language_v1.Document(content=corpus, type_=language_v1.Document.Type.PLAIN_TEXT)

        # Detects the sentiment of the text
        entities = client.analyze_entities(request={'document': document}).entities

        # TODO extract keyword list from entities
        return entities


class KeyBertExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        extractor = KeyBERT('distilbert-base-nli-mean-tokens')
        stop_words = kwargs.get('stop_words', 'english')
        extracted_keywords = extractor.extract_keywords(corpus,  stop_words=stop_words)[:top]
        return [word for word, _ in extracted_keywords]


class RakeExtractor(KeywordExtractor):
    @staticmethod
    def extract(corpus, top, **kwargs) -> list:
        r = Rake(**kwargs)  # Uses stopwords for english from NLTK, and all puntuation characters.
        r.extract_keywords_from_text(corpus)
        return r.get_ranked_phrases()[:top]  # To get keyword phrases ranked highest to lowest.


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
    extracted_keywords = extractor.get_n_best(n=top)
    return [word for word, _ in extracted_keywords]

