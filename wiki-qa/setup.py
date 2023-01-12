"""
This examples demonstrates the setup for Question-Answer-Retrieval.

You can input a query or a question. The script then uses semantic search
to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).

As model, we use: nq-distilbert-base-v1

It was trained on the Natural Questions dataset, a dataset with real questions from Google Search
together with annotated data from Wikipedia providing the answer. For the passages, we encode the
Wikipedia article tile together with the individual text passages.

Google Colab Example: https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing
"""
from sentence_transformers import util
import os

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = 'nq-distilbert-base-v1'

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

wikipedia_filepath = 'data/simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

# To speed things up, pre-computed embeddings are downloaded.
# The provided file encoded the passages with the model 'nq-distilbert-base-v1'
if model_name == 'nq-distilbert-base-v1':
    embeddings_filepath = 'simplewiki-2020-11-01-nq-distilbert-base-v1.pt'
    if not os.path.exists(embeddings_filepath):
        util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01-nq-distilbert-base-v1.pt', embeddings_filepath)
