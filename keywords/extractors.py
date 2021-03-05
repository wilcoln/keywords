import yake


class YakeExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        max_ngram_size = 3  # we want max 3 words per keyword phrase, no more.

        kw_extractor = yake.KeywordExtractor(lan=lang, n=max_ngram_size, top=top)
        extracted_keywords = kw_extractor.extract_keywords(corpus)

        return {word: score for word, score in extracted_keywords}


class TFIDFExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass


class TextRankExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass


class TopicRankExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass


class SpacyExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass


class KeyBertExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass


class ExactMatchExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass


class IntersectionJoinExtractor:
    @staticmethod
    def extract(corpus=None, top=None, lang=None) -> dict:
        pass
