import requests
from xml.etree import ElementTree as ET
import random
import time

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
    # Add more user agents if needed
]

def get_arxiv_articles(query, max_results=30, user_agents=USER_AGENTS):
    base_url = "http://export.arxiv.org/api/query?"
    search_url = f"{base_url}search_query={query}&start=0&max_results={max_results}"
    headers = {
        "User-Agent": random.choice(user_agents)
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        articles = parse_arxiv_response(response.content)
        return articles
    else:
        print(f"Failed to fetch articles. Status code: {response.status_code}")
        return []

def parse_arxiv_response(response_content):
    root = ET.fromstring(response_content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    articles = []
    for entry in root.findall('atom:entry', ns):
        article = {
            'title': entry.find('atom:title', ns).text,
            'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
            'summary': entry.find('atom:summary', ns).text.strip(),
            'published': entry.find('atom:published', ns).text,
            'url': entry.find('atom:id', ns).text
        }
        articles.append(article)
    return articles

# Replace 'your topic' with the topic you are interested in
topic = "ai agents"
articles = get_arxiv_articles(topic)

# Print the articles
if articles:
    for idx, article in enumerate(articles, 1):
        print(f"Article {idx}:")
        print(f"Title: {article['title']}")
        print(f"Authors: {', '.join(article['authors'])}")
        print(f"Published: {article['published']}")
        print(f"URL: {article['url']}")
        print(f"Summary: {article['summary']}\n")
else:
    print("No articles found or failed to retrieve articles.")
