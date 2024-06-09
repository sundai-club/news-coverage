import pandas as pd
import umap
import json


# Function to read the JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return json.loads(content)


def get_embeddings_from_file(selection):

    all_titles = []
    all_arxivid = []
    all_links = []
    embeddings_all = []

    # Mapping selection to corresponding JSON file
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
