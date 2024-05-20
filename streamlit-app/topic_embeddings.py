import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from reddit_api import generate_reddit_json
from mediastack import generate_article_json
from arxiv_api import generate_arxiv_json


def read_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                articles = json.load(file)
                data.extend(articles)
    return data

def process_articles(articles):
    texts = []
    metadata = []
    for article in articles:
        title = article.get('title', '')
        source = article.get('source', '')
        text = article.get('text', '')
        link = article.get('link', '')

        # Combine title, source, and text for embedding
        combined_text = f"Title: {title}\nSource: {source}\nText: {text}"
        texts.append(combined_text)
        metadata.append({
            'title': title,
            'source': source,
            'text': text,
            'link': link,
        })
    return texts, metadata

def generate_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()

def save_embeddings(metadata, embeddings, feature_names, output_file):
    result = []
    for i, meta in enumerate(metadata):
        result.append({
            'title': meta['title'],
            'source': meta['source'],
            'link': meta['link'],
            'embedding': embeddings[i].tolist()
        })
    
    with open(output_file, 'w') as file:
        json.dump({'embeddings': result, 'feature_names': feature_names.tolist()}, file)

def create_embeddings(query):
    directory = './documents/'
    
    #generate json data
    generate_reddit_json(query)
    generate_article_json(query)
    generate_arxiv_json(query)

    
    #create embeddings
    output_file = './embeddings.json'
    articles = read_json_files(directory)
    texts, metadata = process_articles(articles)
    embeddings, feature_names = generate_tfidf_embeddings(texts)
    save_embeddings(metadata, embeddings, feature_names, output_file)

if __name__ == "__main__":
    create_embeddings('LLM agents')
