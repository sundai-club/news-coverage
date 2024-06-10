import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import TapTool, CustomJS, HoverTool
from bokeh.palettes import OrRd, Greens, Reds
import pickle
from scipy import stats
import umap
import json
import sendgrid
from sendgrid.helpers.mail import *

from sklearn.cluster import KMeans
from bokeh.palettes import Category20



def send_email(name, role, requestor_email, request_focus):
    sg = sendgrid.SendGridAPIClient("TBD-ON-DEPLOYED-VM")
    from_email = Email("sundaiclub@gmail.com")
    to_email = To("sundaiclub@gmail.com")
    subject = "AI-NEWS-HOUND: New Visualization Request"
    email_text = f"""
            New Request Submitted:
            Name: {name}
            Role: {role}
            Email: {requestor_email}
            Research Focus Request: {request_focus}
            """
    content = Content("text/plain", email_text)
    mail = Mail(from_email, to_email, subject, content)
    mail.add_to(To("nader_k@mit.edu"))
    sg.client.mail.send.post(request_body=mail.get())


# Function to read the JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return json.loads(content)


st.markdown("""
<h2 style='text-align: center; color: black;'>
<span style='color: green; '> AI </span> to uncover the next <span style='color: green; '> big thing in AI </span>  
</h2>
<p style='text-align: justify; font-size: 19px;'>
Uncover underrepresented research topics with potential for impactful news stories. <br> we help journalists and researchers identify areas of significant scientific interest that lack media coverage. <br> Navigate the visualization below: <strong style='color: green;'>green areas</strong> highlight research topics that are currently hot in the academic world but have not yet been extensively covered in the news.
</p>
""", unsafe_allow_html=True)


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

    alpha_value = st.slider("Pick the hexbin opacity", 0.0, 1.0, 0.5)
    size_value = st.slider("Pick the hexbin gridsize", 0.1, 2.0, 0.25)
    
    # Slider to select the number of clusters
    num_clusters = st.slider("Select the number of clusters", 2, 10, 3)
    colors = Category20[num_clusters]


    st.markdown("""
    <h2 style='text-align: center; color: black;'>We need your Feedback!</h2>
    <p style='text-align: center;'>Let us know what you think and how can we improve.</p>
    <p style="text-align: center;">Check the open source code: <a href="https://github.com/sundai-club/news-coverage" target="_blank">https://github.com/sundai-club/news-coverage</a></p>
    """, unsafe_allow_html=True)

    # Form creation
    with st.form(key='request_form'):
        name = st.text_input('Name')
        role = st.text_input('Role')
        email = st.text_input('Email')
        request_focus = st.text_area('Feedback and suggested use cases',
                                     help='what Research Area would you like us to navigate using our visualization')

        # Form submission button
        submit_button = st.form_submit_button(label='Submit Request')

    if submit_button:
        if name and role and email and request_focus:
            st.success("Thank you! Your request has been submitted, we will contact you shortly.")
            send_email(name, role, email, request_focus)
        else:
            st.error("Please fill out all fields to submit your request.")


def get_embeddings_from_file():
    all_titles = []
    all_arxivid = []
    all_links = []
    embeddings_all = []

    options = ["AI Agents", "AI Assisted Healthcare", "AI Driven Portfolio Management", "AI Public Policies", "View_all_topics_combined"]
    selection = st.selectbox("Select a research category to view:", options)

    input_files = {
        "AI Agents": "embeddings_AIAgents.json",
        "AI Assisted Healthcare": "embeddings_AIAssistedHealthcare.json",
        "AI Driven Portfolio Management": "embeddings_AIDrivenPortfolioManagement.json",
        "AI Public Policies": "embeddings_AIPublicPolicies.json"
    }

    if selection == "View_all_topics_combined":
        inputs = ["embeddings_AIAgents.json", "embeddings_AIAssistedHealthcare.json", "embeddings_AIDrivenPortfolioManagement.json", "embeddings_AIPublicPolicies.json"]
    else:
        inputs = [input_files[selection]]

    for input in inputs:
        embeddings_json = read_json(input)
        for i in range(len(embeddings_json['embeddings'])):
            title = embeddings_json['embeddings'][i]['title']
            source = embeddings_json['embeddings'][i]['type']
            link = embeddings_json['embeddings'][i]['link']
            embedding_i = embeddings_json['embeddings'][i]['embedding']

            all_titles.append(title)
            all_arxivid.append(source)
            all_links.append(link)
            embeddings_all.append(embedding_i)

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

# Use the number of clusters from the slider
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
sources_df['cluster'] = kmeans.fit_predict(sources_df[['x', 'y']])
centroids = sources_df.groupby('cluster')[['x', 'y']].mean()

# Calculate radius for each cluster
def calculate_radius(cluster_df, centroid):
    return np.max(np.sqrt((cluster_df['x'] - centroid['x'])**2 + (cluster_df['y'] - centroid['y'])**2))

cluster_radii = {}
for cluster, centroid in centroids.iterrows():
    cluster_df = sources_df[sources_df['cluster'] == cluster]
    radius = calculate_radius(cluster_df, centroid)
    cluster_radii[cluster] = radius

TOOLTIPS = """
<div style="width:300px;">
($x, $y) <br>
@title <br>
Click to open @link <br> <br>
</div>
"""

p = figure(width=700, height=583, x_range=(0, 15), y_range=(2.5, 15),
           title="Map of embeddings for resources on AI Agents")

# Add TapTool to enable clicking on dots
taptool = p.select(type=TapTool)

# Add JavaScript callback to open link on click
p.add_tools(TapTool())

circle_renderers = []

type_to_color = {'paper': 'green', 'article': 'red', 'reddit': 'blue'}
cluster_to_color = {i: colors[i] for i in range(num_clusters)}


# Draw cluster circles
for cluster, centroid in centroids.iterrows():
    radius = cluster_radii[cluster]
    color = cluster_to_color.get(cluster, 'black')  # Use black as default if cluster exceeds colors
    p.circle(x=centroid['x'], y=centroid['y'], radius=radius, fill_alpha=0.1, line_color=color, legend_label=f'Cluster {cluster}')


for source_type, color in type_to_color.items():
    curr_source = ColumnDataSource(sources_df[sources_df["data_source"] == source_type])
    circle_renderer = p.circle('x', 'y', size=5, source=curr_source, alpha=0.3, color=color, legend_label=source_type)
    p.js_on_event('tap', CustomJS(args=dict(source=curr_source), code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var link = source.data['link'][indices[0]];
            window.open(link);
        }
    """))
    circle_renderers.append(circle_renderer)

    if source_type == 'paper':
        p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=size_value,
                 palette=np.flip(Greens[9]), alpha=alpha_value)
    if source_type == 'article':
        p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=size_value,
                 palette=np.flip(Reds[9]), alpha=alpha_value)

hover_tool = HoverTool(tooltips=TOOLTIPS, renderers=circle_renderers)
p.add_tools(hover_tool)

st.bokeh_chart(p)
