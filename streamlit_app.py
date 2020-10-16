import numpy as np
import streamlit as st
import matplotlib.pylab as plt
import pathlib 
from whatlies.language import CountVectorLanguage
from whatlies.transformers import Pca, Umap

txt = pathlib.Path("nlu.md").read_text()
texts = [t.replace(" - ", "") for t in txt.split("\n") if len(t) > 0 and t[0] != "#"]

n_svd = st.sidebar.slider("Number of SVD components", min_value=2, max_value=100, step=1)
min_ngram, max_ngram = st.sidebar.slider("Range of ngrams", min_value=1, max_value=5, step=1, value=(2, 3))


st.markdown("# Simple Text Clustering")
lang = CountVectorLanguage(n_svd, ngram_range=(min_ngram, max_ngram))

p = (lang[texts]
     .transform(Umap(2))
     .plot_interactive(annot=False)
     .properties(width=500, height=500))

st.write(p)


