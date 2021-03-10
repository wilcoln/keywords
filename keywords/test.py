from keywords import datasets, extractors


dataset = datasets.load('500N-KPCrowd-v1.1')
corpus = dataset['corpora'][0]
print(corpus['text'])
print('Ground truth')
print(corpus['keywords'])

test_extractors = [
    # Unsupervised
    extractors.YakeExtractor,
    extractors.TextRankExtractor,
    extractors.KPMinerExtractor,
    extractors.TfIdfExtractor,
    extractors.RakeExtractor,
    extractors.GensimExtractor,
    extractors.SpacyExtractor,
    extractors.TopicRankExtractor,
    extractors.SingleRankExtractor,
    extractors.PositionRankExtractor,
    extractors.MultipartiteRankExtractor,
    extractors.KeyBertExtractor,
    extractors.GoogleCloudExtractor,
    # Supervised
    extractors.KeaExtractor,
    extractors.WINGNUSExtractor,
]

for extractor in test_extractors:
    print(extractor)
    keywords = extractor.extract(corpus=corpus['text'], top=10)
    print(keywords)
