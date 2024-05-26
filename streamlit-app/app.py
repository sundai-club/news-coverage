import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import TapTool, CustomJS
from bokeh.palettes import OrRd
import pickle
from scipy import stats
import umap
import json


# Function to read the JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return json.loads(content)


st.title("Embedding explorer")
st.markdown("Interpreting the UMAP plot")


def density_estimation(m1, m2, xmin=0, ymin=0, xmax=15, ymax=15):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


with st.sidebar:
    button_html = """
            <style>
                .btn-custom {
                    color: white; 
                    background-color: #D97884; 
                    border: none; 
                    padding: 10px 20px; 
                    font-size: 16px;
                    border-radius: 8px; /* Rounded edges */
                    transition: all 0.3s; /* Smooth transition for hover effects */
                }

                .btn-custom:hover {
                    background-color: #5779ff; /* Change background on hover */
                    color: black; /* Change text color on hover */
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Add shadow on hover */
                }
            </style>
            <form action="https://sundai.club" target="_blank">
                <button type="submit" class="btn-custom">
                    <i class="fa fa-rocket"></i> Visit SundAI.club
                </button>
            </form>
            """

    st.markdown(button_html, unsafe_allow_html=True)

    st.markdown(
        'Query items containing specific phrases in the dataset and show it as a heatmap. Enter the phrase of interest, then change the size and opacity of the heatmap as desired to find the high-density regions. Hover over blue points to see the details of individual papers.')
    st.markdown(
        '`Note`: (i) if you enter a query that is not in the corpus of abstracts, it will return an error. just enter a different query in that case. (ii) there are some empty tooltips when you hover, these correspond to the underlying hexbins, and can be ignored.')

    search_query = st.text_input("Search query", key="phrase", value="bee")

    alpha_value = st.slider("Pick the hexbin opacity", 0.0, 1.0, 0.81)
    size_value = st.slider("Pick the hexbin gridsize", 0.5, 5.0, 1.0)


def get_embeddings_from_file(file_path='embeddings_1.json'):
    embeddings_json = read_json(file_path)
    all_titles = []
    all_arxivid = []
    all_links = []
    embeddings_all = []

    for i in range(0, len(embeddings_json['embeddings'])):
        title = embeddings_json['embeddings'][i]['title']
        source = embeddings_json['embeddings'][i]['source']

        link = embeddings_json['embeddings'][i]['link']
        embedding_i = embeddings_json['embeddings'][i]['embedding']

        all_titles.append(title)
        all_arxivid.append(source)
        all_links.append(link)
        embeddings_all.append(embedding_i)

    # todo: Just a hack for testing coloring for differnet JSON files
    if file_path == 'embeddings_1.json':
        all_titles = all_titles[:40]
        all_arxivid = all_arxivid[:40]
        all_links = all_links[:40]
        embeddings_all = embeddings_all[:40]
    else:
        all_titles = all_titles[40:]
        all_arxivid = all_arxivid[40:]
        all_links = all_links[40:]
        embeddings_all = embeddings_all[40:]

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = umap_reducer.fit_transform(embeddings_all)

    source = ColumnDataSource(data=dict(
        x=embedding[0:, 0],
        y=embedding[0:, 1],
        title=all_titles,
        data_source=all_arxivid,
        link=all_links,
    ))

    return source, all_titles


all_titles = []

# todo: this is a hack, replace with 4 actual embedding files.
source_1, titles_1 = get_embeddings_from_file('embeddings_1.json')
all_titles += titles_1

source_2, titles_2 = get_embeddings_from_file('embeddings_2.json')
all_titles += titles_2

TOOLTIPS = """
<div style="width:300px;">
ID: $index <br>
($x, $y) <br>
@title <br>
Click to open @link <br> <br>
</div>
"""

phrase = st.session_state.phrase

p = figure(width=700, height=583, tooltips=TOOLTIPS, x_range=(0, 15), y_range=(2.5, 15),
           title="UMAP projection of embeddings for the given embeddings")

# Add TapTool to enable clicking on dots
taptool = p.select(type=TapTool)
p.add_tools(TapTool())

# Add JavaScript callback to open link on click
p.js_on_event('tap', CustomJS(args=dict(source=source), code="""
    var indices = source.selected.indices;
    if (indices.length > 0) {
        var link = source.data['link'][indices[0]];
        window.open(link);
    }
"""))

# TODO: change this with actual semantic search - the embedding distance basically
phrase_flags = np.zeros((len(all_titles),))

phrase_flags = np.zeros((len(all_titles),))
for i in range(len(all_titles)):
    if phrase.lower() in all_titles[i].lower():
        phrase_flags[i] = 1

# TODO: add a summarization to the HEXBIN items so that when hovering a bin can see a summary of the items within
# p.hexbin(embedding[phrase_flags == 1, 0], embedding[phrase_flags == 1, 1], size=size_value,
#          palette=np.flip(OrRd[8]), alpha=alpha_value)


p.circle('x', 'y', size=3, source=source_1, alpha=0.3, color='blue')
p.circle('x', 'y', size=3, source=source_2, alpha=0.3, color='red')

st.bokeh_chart(p)

fig = plt.figure(figsize=(10.5, 9 * 0.8328))
# plt.scatter(embedding[0:, 0], embedding[0:, 1], s=2, alpha=0.1)
# plt.hexbin(embedding[phrase_flags == 1, 0], embedding[phrase_flags == 1, 1],
#            gridsize=int(10 * size_value), cmap='viridis', alpha=alpha_value, extent=(-1, 16, 1.5, 16), mincnt=1)
# plt.title("UMAP localization of heatmap keyword: " + phrase)
plt.axis([0, 15, 2.5, 15])
# clbr = plt.colorbar()
# clbr.set_label('# papers')
plt.axis('off')
st.pyplot(fig)
