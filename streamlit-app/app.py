import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
import pickle
from scipy import stats
import umap

st.title("Embedding explorer")
st.markdown("Interpreting the UMAP plot")

with open("document_embeddings_1.pkl", "rb") as f:
    embeddings_data = pickle.load(f)

embeddings_all = embeddings_data["embeddings"]
all_titles = embeddings_data["titles"]
all_arxivid = embeddings_data["arxivid"]
all_links = embeddings_data["links"]

umap_reducer = umap.UMAP(n_components=2, random_state=42)
embedding = umap_reducer.fit_transform(embeddings_all)


def density_estimation(m1, m2, xmin=0, ymin=0, xmax=15, ymax=15):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

# TODO: implement this query feature later
# st.sidebar.markdown(
#     'This is a widget that allows you to look for papers containing specific phrases in the dataset and show it as a heatmap. Enter the phrase of interest, then change the size and opacity of the heatmap as desired to find the high-density regions. Hover over blue points to see the details of individual papers.')
# st.sidebar.markdown(
#     '`Note`: (i) if you enter a query that is not in the corpus of abstracts, it will return an error. just enter a different query in that case. (ii) there are some empty tooltips when you hover, these correspond to the underlying hexbins, and can be ignored.')
#
# st.sidebar.text_input("Search query", key="phrase", value="Quenching")
# alpha_value = st.sidebar.slider("Pick the hexbin opacity", 0.0, 1.0, 0.81)
# size_value = st.sidebar.slider("Pick the hexbin gridsize", 10, 50, 20)

# phrase = st.session_state.phrase

source = ColumnDataSource(data=dict(
    x=embedding[0:, 0],
    y=embedding[0:, 1],
    title=all_titles,
    link=all_links,
))

TOOLTIPS = """
<div style="width:300px;">
ID: $index
($x, $y)
@title <br>
@link <br> <br>
</div>
"""

p = figure(width=700, height=583, tooltips=TOOLTIPS, x_range=(0, 15), y_range=(2.5, 15),
           title="UMAP projection of embeddings for the given embeddings")

# phrase_flags = np.zeros((len(all_titles),))


# p.hexbin(embedding[phrase_flags==1,0],embedding[phrase_flags==1,1], size=size_value,
#          palette = np.flip(OrRd[8]), alpha=alpha_value)

p.circle('x', 'y', size=3, source=source, alpha=0.3)
st.bokeh_chart(p)


fig = plt.figure(figsize=(10.5, 9 * 0.8328))
plt.scatter(embedding[0:, 0], embedding[0:, 1], s=2, alpha=0.1)
# plt.hexbin(embedding[phrase_flags==1,0],embedding[phrase_flags==1,1],
# gridsize=size_value, cmap = 'viridis', alpha=alpha_value,extent=(-1,16,1.5,16),mincnt=10)
# plt.title("UMAP localization of heatmap keyword: " + phrase)
plt.axis([0, 15, 2.5, 15])
# clbr = plt.colorbar()
# clbr.set_label('# papers')
plt.axis('off')
st.pyplot(fig)
