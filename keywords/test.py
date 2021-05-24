from keywords import datasets, extractors, evaluation


evaluation.visual_bench(dataset='500N-KPCrowd-v1.1', num_docs=2, k=20)

# dataset = datasets.load('500N-KPCrowd-v1.1')
# documents = dataset['documents']
# test_extractors = [
#     # Unsupervised
#     extractors.YakeExtractor,
#     extractors.TextRankExtractor,
#     extractors.KPMinerExtractor,
#     extractors.TfIdfExtractor,
#     extractors.RakeExtractor,
#     extractors.TopicRankExtractor,s
#     extractors.SingleRankExtractor,
#     extractors.PositionRankExtractor,
#     extractors.MultipartiteRankExtractor,
#     extractors.KeyBertExtractor,
# ]
#
# for extractor in test_extractors:
#     model = extractor(n_gram=4, keyword_index_size=300, documents=[documents[0]['text']])
#     print(model)
#     print(model.predict(document=documents[0]['text'], top=10))

