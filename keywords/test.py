import json
import pandas as pd
from csv import reader


# evaluation.visual_bench(dataset='500N-KPCrowd-v1.1', num_docs=2, k=20)

# dataset = datasets.load('500N-KPCrowd-v1.1')
from keywords import extractors


def read_topics(content_filename, taxonomy_filename):
    _topics = []

    with open(content_filename, "r") as content_file:
        content_lines = content_file.readlines()
        with open(taxonomy_filename, 'r') as taxonomy_file:
            read_obj = reader(taxonomy_file)
            next(read_obj)  # pass the header
            for i, (topic_name, parent_id, line_span) in enumerate(read_obj):
                text = ''
                if line_span:
                    line_span = line_span.split('-')
                    start, end = int(line_span[0]) - 1, int(line_span[1]) - 1
                    text = ''.join(content_lines[start: end])

                _topics.append(
                    {
                        'id': i + 1,
                        'name': topic_name,
                        'text': text,
                        'parent_id': int(parent_id) if parent_id else 0,
                    }
                )

    return _topics


topics = read_topics('../data/reagan/content.md', '../data/reagan/taxonomy.csv')


keyword_index = list(pd.read_csv('../data/reagan/keyword_index.csv')['keyword'])

test_extractors = [
    # Unsupervised
    extractors.YakeExtractor,
    extractors.TextRankExtractor,
    extractors.KPMinerExtractor,
    extractors.TfIdfExtractor,
    # extractors.RakeExtractor,
    # extractors.TopicRankExtractor,
    # extractors.SingleRankExtractor,
    # extractors.PositionRankExtractor,
    # extractors.MultipartiteRankExtractor,
    # extractors.KeyBertExtractor,
]


def dict_to_json_file(adict, filename):
    with open(filename, 'w') as outfile:
        json.dump(adict, outfile)


for topic in topics:
    topic['keywords'] = {
        model.__class__.__name__: {
            'raw': model.get_n_best(n=10),
            'filtered': model.get_n_best(n=10, index=keyword_index)
        }
        for model in [extractor(document=topic['text']) for extractor in test_extractors]
    }


dict_to_json_file(topics, '../data/reagan/tree.json')
