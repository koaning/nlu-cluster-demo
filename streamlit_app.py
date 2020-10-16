import numpy as np
import streamlit as st
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

n = 1000
np.random.seed(42)
x = np.linspace(0, 6, n)
X = np.linspace(0, 6, n)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.random(n) * 0.3

n_est = st.sidebar.slider("n_est", min_value=1, max_value=5_000, step=1)

@st.cache
def make_predictions(n_est):
    mod1 = DecisionTreeRegressor(max_depth=4)
    y1 = mod1.fit(X,y).predict(X)
    y2 = AdaBoostRegressor(mod1, n_estimators=n_est).fit(X, y).predict(X)
    return y1, y2

y1, y2 = make_predictions(n_est=n_est)

if st.sidebar.checkbox("Toggle ScatterChart"):
    plt.scatter(x, y, alpha=0.1)
plt.plot(x, y1, label="just a tree")
plt.plot(x, y2, label=f"adaboost-{n_est}")
plt.legend()

st.pyplot()