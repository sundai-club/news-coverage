import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from reddit_api import generate_reddit_json
from mediastack import generate_article_json
from arxiv_api import generate_arxiv_json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer


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
        text = preprocess_text(article.get('text', ''))
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
    return tfidf_matrix.toarray()

def generate_sentence_embeddings(texts):
    # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)
    return embeddings

def save_embeddings(metadata, embeddings, output_file):
    result = []
    for i, meta in enumerate(metadata):
        result.append({
            'title': meta['title'],
            'source': meta['source'],
            'link': meta['link'],
            'embedding': embeddings[i].tolist()
        })
    
    with open(output_file, 'w') as file:
        json.dump({'embeddings': result}, file)

# Preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def main():
    #query = 'how will generative artificial intelligence AI change future of work jobs the economy'
    query = "Economy Technology Jobs AI"
    directory = './documents/'
    
    #generate json data
    # generate_reddit_json(query)
    # generate_article_json(query)
    generate_arxiv_json(query)

    #create embeddings
    '''
    nltk.download('punkt')
    nltk.download('stopwords')
    output_file = directory + 'embeddings_mpnet_2.json'
    articles = read_json_files(directory)
    texts, metadata = process_articles(articles)
    # embeddings = generate_tfidf_embeddings(texts)
    embeddings = generate_sentence_embeddings(texts)
    save_embeddings(metadata, embeddings, output_file)
    '''

if __name__ == "__main__":
    main()
