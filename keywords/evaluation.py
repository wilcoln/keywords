from typing import List

import numpy as np
import pandas as pd

import plotly.express as px

from keywords import datasets, extractors


def get_cleaned_documents(dataset: str, num_docs: int = None):
    """

    This function will fetch the data based on the name of the dataset. It will:
       1. load the dataset
       2. get rid of records in which # of keywords > # of words in the original text
       3. get a numbers of documents to use, based on 2nd parameter: num

        Args:
           - dataset: The name of the data to load
           - num_docs: The number of document to load

        Returns: A list of document (a documents)

    """
    data = datasets.load(dataset)

    documents = data['documents']

    # get rid of data in which # of keywords > # words in text
    documents = [
        document for document in documents
        if len(document['text'].split()) > len(document['keywords']) + 30
    ]

    return documents[:num_docs]


def get_output(document: str, top: int = None):
    """
    This function will initialize all algorithms,
    and output the results from them. It then wrap
    all results into a dictionary.

    Notice:
        If new algorithms has been added in, then new
        initialization code needs to be added here
    """

    return {
        extractor: [
            keyword for keyword, _ in extractor().get_n_best(document=document, n=top)
        ]
        for extractor in [
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
    }


# %%
# average precison
def apk(actual: List[str], predicted: List[str], k: int):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
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


# mean average precision
def mapk(actual: List[str], predicted: List[str], k: int):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def visual_bench(dataset: str, num_docs: int, k: int = None):
    """
    Visualize mean average precision for each algorithm given random data
    Parameters
    ----------
    dataset: the name of the original dataset

    k : int, optional
        The maximum number of predicted elements, if it's none, then it will be set equals to # of goldkeys for each pair

    num_docs: int
        The number of document that needs to be choose

    Returns
    -------
     map scores for each algorithms : Pandas dataframe
            A pandas dataframe containing map scores for each algorithm
    """
    documents = get_cleaned_documents(dataset, num_docs)
    doc_output_list = [
        get_output(
            document=doc['text'],
            top=k if k is not None else len(doc['keywords']),
        ) for doc in documents
    ]

    # get map scores for each data
    doc_scores_list = [{
        extractor:
        mapk(actual=documents[doc_idx]['keywords'],
             predicted=extracted_keywords,
             k=k if k is not None else len(documents[doc_idx]['keywords']))
        for extractor, extracted_keywords in results.items()
    } for doc_idx, results in enumerate(doc_output_list)]

    # formalize scores
    scores_ = [doc_scores.values() for doc_scores in doc_scores_list]

    # retrieve name of extraction algorithm
    algorithms = [extractor.__name__ for extractor in doc_scores_list[0].keys()]

    # create a dataframe
    df = pd.DataFrame(data=scores_, index=[doc['id'] for doc in documents], columns=algorithms).reset_index() \
        .rename(columns={'index': 'document'})

    # visualization
    df_ = pd.melt(df, id_vars=['document'], var_name='algorithm', value_name='MAP value')
    fig = px.bar(
        df_,
        x='document',
        color='algorithm',
        y='MAP value',
        title="Mean Average Precision",
        barmode='group',
    )

    fig.show()

    return df
