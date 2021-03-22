from keywords import datasets, extractors


dataset = datasets.load('500N-KPCrowd-v1.1')
corpora = dataset['corpora']

training = [corpus for corpus in corpora[:1]]

test_extractors = [
    # Unsupervised
    extractors.YakeExtractor,
    extractors.TextRankExtractor,
    extractors.KPMinerExtractor,
    extractors.TfIdfExtractor,
    extractors.RakeExtractor,
    extractors.TopicRankExtractor,
    extractors.SingleRankExtractor,
    extractors.PositionRankExtractor,
    extractors.MultipartiteRankExtractor,
    extractors.KeyBertExtractor,
]

for extractor in test_extractors:
    print(extractor)
    model = extractor(n_gram=4, total_keywords_in_training=300, documents=[corpus['text'] for corpus in training])
    print(model.predict(text=training[0]['text'], topn=10))

