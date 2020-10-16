import numpy as np
import streamlit as st
import matplotlib.pylab as plt
import pathlib 
from whatlies.language import CountVectorLanguage
from whatlies.transformers import Pca, Umap

txt = pathlib.Path("nlu.md").read_text()
texts = [t.replace(" - ", "") for t in txt.split("\n") if len(t) > 0 and t[0] != "#"]

lang = CountVectorLanguage(10)

st.write(lang[texts].transform(Umap(2)).plot_interactive().properties(width=500, height=500))