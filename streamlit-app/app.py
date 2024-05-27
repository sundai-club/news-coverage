import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import TapTool, CustomJS
from bokeh.palettes import OrRd, Greens, Reds
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

    st.markdown('#')
    st.markdown(
        'Query items containing specific phrases in the dataset and show it as a heatmap. Enter the phrase of interest, then change the size and opacity of the heatmap as desired to find the high-density regions. Hover over blue points to see the details of individual papers.')
    st.markdown(
        '`Note`: (i) if you enter a query that is not in the corpus of abstracts, it will return an error. just enter a different query in that case. (ii) there are some empty tooltips when you hover, these correspond to the underlying hexbins, and can be ignored.')

    search_query = st.text_input("Search query", key="phrase", value="bee")

    alpha_value = st.slider("Pick the hexbin opacity", 0.0, 1.0, 0.5)
    size_value = st.slider("Pick the hexbin gridsize", 0.5, 5.0, 1.0)


def get_embeddings_from_file():
    embeddings_json = read_json("embeddings.json")

    all_titles = []
    all_arxivid = []
    all_links = []
    embeddings_all = []

    for i in range(0, len(embeddings_json['embeddings'])):
        title = embeddings_json['embeddings'][i]['title']
        source = embeddings_json['embeddings'][i]['type']
        link = embeddings_json['embeddings'][i]['link']
        embedding_i = embeddings_json['embeddings'][i]['embedding']

        all_titles.append(title)
        all_arxivid.append(source)
        all_links.append(link)
        embeddings_all.append(embedding_i)

    # TODO: make sure the UMAP is ran on all the embeddings at the end
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    final_2d_embeddings = umap_reducer.fit_transform(embeddings_all)

    sources_df = pd.DataFrame.from_dict(data=dict(
        x=final_2d_embeddings[0:, 0],
        y=final_2d_embeddings[0:, 1],
        title=all_titles,
        data_source=all_arxivid,
        link=all_links,
    ))

    return sources_df, all_titles, final_2d_embeddings


sources_df, all_titles, final_2d_embeddings = get_embeddings_from_file()
source = ColumnDataSource(sources_df)

TOOLTIPS = """
<div style="width:300px;">
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

# Add JavaScript callback to open link on click
p.add_tools(TapTool())

# TODO: change this with actual semantic search - the embedding distance basically
phrase_flags = np.zeros((len(all_titles),))

for i in range(len(all_titles)):
    if phrase.lower() in all_titles[i].lower():
        phrase_flags[i] = 1

# TODO: create a hexbin manually with the needed description and number of points...
# p.hexbin(final_2d_embeddings[:, 0], final_2d_embeddings[:, 1], size=0.5,
#          palette=np.flip(OrRd[9]), alpha=alpha_value)

# TODO: add a summarization to the HEXBIN items so that when hovering a bin can see a summary of the items within
# p.hexbin(embedding[phrase_flags == 1, 0], embedding[phrase_flags == 1, 1], size=size_value,
#          palette=np.flip(OrRd[8]), alpha=alpha_value)

type_to_color = {'paper': 'red', 'article': 'green', 'reddit': 'blue'}
for source_type, color in type_to_color.items():
    curr_source = ColumnDataSource(sources_df[sources_df["data_source"] == source_type])
    p.circle('x', 'y', size=3, source=curr_source, alpha=0.3, color=color, legend_label=source_type)
    p.js_on_event('tap', CustomJS(args=dict(source=curr_source), code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var link = source.data['link'][indices[0]];
            window.open(link);
        }
    """))

    if source_type == 'paper':
        p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=0.25,
                 palette=np.flip(Greens[9]), alpha=alpha_value)
    if source_type == 'article':
        p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=0.25,
                 palette=np.flip(Reds[9]), alpha=alpha_value)

st.bokeh_chart(p)

# fig = plt.figure(figsize=(10.5, 9 * 0.8328))

# plt.hexbin(embedding[phrase_flags == 1, 0], embedding[phrase_flags == 1, 1],
#            gridsize=int(10 * size_value), cmap='viridis', alpha=alpha_value, extent=(-1, 16, 1.5, 16), mincnt=1)
# plt.title("UMAP localization of heatmap keyword: " + phrase)
# plt.axis([0, 15, 2.5, 15])
# clbr = plt.colorbar()
# clbr.set_label('# papers')
# plt.axis('off')
# st.pyplot(fig)
