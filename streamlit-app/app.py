import streamlit as st
import numpy as np
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import TapTool, CustomJS, HoverTool
from bokeh.palettes import OrRd, Greens, Reds


from sklearn.cluster import KMeans
from bokeh.palettes import Category20

from utils.send_email import send_email
from utils.read_embeddings import get_embeddings_from_file
from utils.score_calc import add_scores


draw_hexbins = False

st.set_page_config(page_title="AI News Hound", page_icon="🐶", layout="wide")
st.markdown("""
<h2 style='text-align: center; color: black;'>
<span style='color: green; '> AI </span> to uncover the next <span style='color: grn; '> big thing in AI </span>  
</h2>
<p style='text-align: justify; font-size: 19px;'>
Uncover underrepresented research topics with potential for impactful news stories. <br> we help journalists and researchers identify areas of significant scientific interest that lack media coverage. <br> Navigate the visualization below: <strong style='color: green;'>green areas</strong> highlight research topics that are currently hot in the academic world but have not yet been extensively covered in the news.
</p>
""", unsafe_allow_html=True)


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

    options = ["AI Agents", "AI Assisted Healthcare", "AI Driven Portfolio Management", "AI Public Policies",
               "View_all_topics_combined"]

    selection = st.selectbox("Select a research category to view:", options)

    alpha_value = 0.2
    # st.slider("Pick the hexbin opacity", 0.0, 1.0, 0.25)
    if draw_hexbins:
        size_value = st.slider("Pick the hexbin gridsize", 0.1, 2.0, 0.25)
    
    # Slider to select the number of clusters
    num_clusters = st.slider("Select the number of clusters", 2, 10, 5)
    colors = Category20[num_clusters]
    
    
    show_clusters = st.sidebar.checkbox("Show Cluster Circles", value=True)



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

sources_df, all_titles, final_2d_embeddings = get_embeddings_from_file(selection)
# source = ColumnDataSource(sources_df)

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

cluster_counts = sources_df['cluster'].value_counts().sort_index()
centroids['count'] = cluster_counts.values
centroids['radius'] = [cluster_radii[i] for i in centroids.index]
centroid_source = ColumnDataSource(data=dict(
    x=centroids['x'],
    y=centroids['y'],
    count=centroids['count'],
    radius=centroids['radius']
))



TOOLTIPS_DOTS = """

<div style="width:300px;">
($x, $y) <br>
@title <br>
Click to open @link <br> <br>
</div>
"""

# Calculate the range of your points - so that figure can be centered
x_min, x_max = sources_df['x'].min(), sources_df['x'].max()
y_min, y_max = sources_df['y'].min(), sources_df['y'].max()
# Add some padding to the ranges
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1
x_range = (x_min - x_padding, x_max + x_padding)
y_range = (y_min - y_padding, y_max + y_padding)

# p = figure(width=700, height=583, x_range=x_range, y_range=y_range, title="Map of embeddings for resources on AI Agents")

p = figure(width=700, height=583, x_range=(0, 15), y_range=(2.5, 15),
           title="Map of embeddings for resources on AI Agents")
p.axis.visible = False
p.grid.visible = True

# Add TapTool to enable clicking on dots
taptool = p.select(type=TapTool)

# Add JavaScript callback to open link on click
p.add_tools(TapTool())

circle_renderers = []

type_to_color = {'paper': 'green', 'article': 'red', 'reddit': 'blue'}
cluster_to_color = {i: colors[i] for i in range(num_clusters)}


for source_type, color in type_to_color.items():
    curr_source = ColumnDataSource(sources_df[sources_df["data_source"] == source_type])

    if color == 'green':
        circle_renderer = p.square('x', 'y', size=5, source=curr_source, alpha=0.3, color=color, legend_label=source_type)
    elif color == 'red':
        circle_renderer = p.circle('x', 'y', size=5, source=curr_source, alpha=0.3, color=color, legend_label=source_type)
    else:
        circle_renderer = p.triangle('x', 'y', size=5, source=curr_source, alpha=0.3, color=color, legend_label=source_type)

    p.js_on_event('tap', CustomJS(args=dict(source=curr_source), code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var link = source.data['link'][indices[0]];
            window.open(link);
        }
    """))

    circle_renderers.append(circle_renderer)

    if draw_hexbins:
        if source_type == 'paper':
            p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=size_value,
                    palette=np.flip(Greens[9]), alpha=alpha_value)
        if source_type == 'article':
            p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=size_value,
                    palette=np.flip(Reds[9]), alpha=alpha_value)

    # (r, bins) = p.hexbin(sources_df['x'], sources_df['y'], size=size_value, palette=np.flip(Greens[9]), alpha=alpha_value, hover_color="pink", hover_alpha=0.4)
    #
    # p.js_on_event('tap', CustomJS(args=dict(source=curr_source), code="""
    #         var indices = source.selected.indices;
    #         if (indices.length > 0) {
    #             var link = source.data['link'][indices[0]];
    #             window.open(link);
    #         }
    #     """))
    #
    # hex_titles = []
    # for b in bins.q:
    #     # todo: color from gray to gold according to score...
    #     # todo: fucking dynamic title for hex
    #     r.glyph.fill_color = Greens[9][8]
    #     hex_titles.append("nice title...")
    #
    # r.data_source.data['title'] = hex_titles

    # hex_info_source = ColumnDataSource(data=dict(q=bins.q, r=bins.r, c=bins.counts, title=titles))
    #
    # fill_color = linear_cmap('c', np.flip(Greens[9]), 0, max(bins.counts))
    #
    # r = p.hex_tile(q="q", r="r", size=size_value, orientation="pointytop", aspect_scale=1,
    #                   source=hex_info_source, line_color=None, fill_color=fill_color)

    # bins = hexbin(sources_df['x'], sources_df['y'], size_value, "pointytop", aspect_scale=1)
    #
    # if fill_color is None:
    #     fill_color = linear_cmap('c', palette, 0, max(bins.counts))

    #
    # fill_color = linear_cmap('c', np.flip(Greens[9]), 0, max(bins.counts))
    #
    # source = ColumnDataSource(data=dict(q=bins.q, r=bins.r, c=bins.counts))
    #
    # p.hex_tile(q="q", r="r", size=alpha_value, orientation="pointytop", aspect_scale=1, source=source)

    # if source_type == 'paper':
    #     p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=size_value,
    #              palette=np.flip(Greens[9]), alpha=alpha_value)
    # if source_type == 'article':
    #     p.hexbin(sources_df[sources_df["data_source"] == source_type]['x'], sources_df[sources_df["data_source"] == source_type]['y'], size=size_value,
    #              palette=np.flip(Reds[9]), alpha=alpha_value)
    #



# Draw cluster circles
if show_clusters:

    for cluster, centroid in centroids.iterrows():
        radius = cluster_radii[cluster]
        color = cluster_to_color.get(cluster, 'black')  # Use black as default if cluster exceeds colors
        p.circle(x=centroid['x'], y=centroid['y'], radius=radius, fill_alpha=0.1, line_color=color)
    # hover = HoverTool(tooltips=[("article count", "@count"), ("(x,y)", "(@x, @y)"), ("radius", "@radius")])
    hover = HoverTool(tooltips=[("article count", "@count"), ("circle radius", "@radius")])
    p.circle(x='x', y='y', radius='radius', source=centroid_source, fill_alpha=0.1, line_color='black')
    p.add_tools(hover)
    
hover_tool = HoverTool(tooltips=TOOLTIPS_DOTS, renderers=circle_renderers)
if draw_hexbins:
    hover = HoverTool(tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)"), ("title", "@title")])

p.add_tools(hover_tool)
p.legend.location = "top_left"

st.bokeh_chart(p)
