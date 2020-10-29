import pathlib 
from io import StringIO

import streamlit as st
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering
from whatlies.language import CountVectorLanguage
from whatlies.transformers import Pca, Umap
from whatlies import EmbeddingSet, Embedding

import sentencepiece as spm
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

with tf.Session() as sess:
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/1")
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))


def process_to_IDs_in_sparse_format(sp, sentences):
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return values, indices, dense_shape


hash_funcs={'_thread.RLock' : lambda _: None,
            '_thread.lock' : lambda _: None,
            'builtins.PyCapsule': lambda _: None,
            '_io.TextIOWrapper' : lambda _: None,
            'builtins.weakref': lambda _: None,
            'builtins.dict' : lambda _:None}


@st.cache(hash_funcs=hash_funcs)
def calculate_embeddings(messages):
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(
            encodings,
            feed_dict={input_placeholder.values: values,
                       input_placeholder.indices: indices,
                       input_placeholder.dense_shape: dense_shape})

    return message_embeddings

st.sidebar.markdown("Made with love over at [Rasa](https://rasa.com/).")
st.sidebar.image("https://rasahq.github.io/rasa-nlu-examples/square-logo.svg", width=100)
uploaded = st.sidebar.file_uploader("Upload a `.txt` file for clustering. Each utterance should appear on a new line.")
if not uploaded:
    txt = pathlib.Path("nlu.md").read_text()
    texts = list(set([t for t in txt.split("\n") if len(t) > 0]))
else:
    bytes_data = uploaded.read()
    stringio = StringIO(bytes_data.decode("utf-8"))
    string_data = stringio.read()
    texts = [t.replace(" - ", "") for t in string_data.split("\n") if len(t) > 0 and t[0] != "#"]

method = st.sidebar.selectbox("Select Embedding Method", ["CountVector SVD", "Lite Sentence Encoding"])
if method == "CountVector SVD":
    n_svd = st.sidebar.slider("Number of SVD components", min_value=2, max_value=100, step=1)
    min_ngram, max_ngram = st.sidebar.slider("Range of ngrams", min_value=1, max_value=5, step=1, value=(2, 3))

reduction_method = st.sidebar.selectbox('Reduction Method', ('Umap', 'Pca'))
if reduction_method == "Umap":
    n_neighbors = st.sidebar.slider("Number of UMAP neighbors", min_value=1, max_value=100, value=15, step=1)
    min_dist = st.sidebar.slider("Minimum Distance for UMAP", min_value=0.01, max_value=0.99, value=0.8, step=0.01)
    reduction = Umap(2, n_neighbors=n_neighbors, min_dist=min_dist)
else:
    reduction = Pca(2)

st.markdown("# Simple Text Clustering")
st.markdown("Let's say you've gotten a lot of feedback from folks on social media. You might like to be able to distill main topics. It might even inspire some intents that will be used in a virtual assistant! This tool will help you discover them. This app will attempt to cluster whatever text you give it. The chart will try to clump text together and you can explore underlying patterns.")

if method == "CountVector SVD":
    lang = CountVectorLanguage(n_svd, ngram_range=(min_ngram, max_ngram))
    embset = lang[texts]
if method == "Lite Sentence Encoding":
    embset = EmbeddingSet(*[Embedding(t, v) for t, v in zip(texts, calculate_embeddings(texts))])


# cluster = st.sidebar.checkbox("Assign Clusters")
# if cluster:
#     linkage = st.sidebar.selectbox(
#         'Linkage Method',
#         ('average', 'complete', 'ward', 'single'))
#     n_cluster = st.sidebar.slider("Number of clusters", min_value=2, max_value=100, step=1, value=20)
#
#     model = AgglomerativeClustering(linkage=linkage, n_clusters=n_cluster)
#     X = embset.to_X()
#     clusters = {n: str(c) for n, c in zip([e.name for e in embset], model.fit_predict(X))}
#     embset = embset.assign(group=lambda d: clusters[d.name])

p = (embset
     .transform(reduction)
     .plot_interactive(annot=False)
     .properties(width=500, height=500, title=""))

st.write(p)
