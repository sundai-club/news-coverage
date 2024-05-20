import requests
import json

def get_articles(api_key, keywords, languages='en', sort='published_desc', limit=25):
    url = 'http://api.mediastack.com/v1/news?access_key=' + api_key + "&keywords=ai agents&sort=published_desc"
    params = {
        'keywords': keywords,
        'languages': languages,
        'sort': sort,
        'limit': limit
    }
    
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        data = response.json()
        return data['data']
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def format_articles(articles):
    formatted_articles = []
    for article in articles:
        formatted_article = {
            'title': article['title'],
            'text': article['title'] + " " + article['description'],
            'source': article['source'],
            'link': article['url'],
            'published_at': article['published_at'],
            'type': "article"
        }
        formatted_articles.append(formatted_article)
    return formatted_articles

def print_articles(articles):
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"Published At: {article['published_at']}")
        print('-' * 80)
        
def generate_article_json(query):
    api_key = ''  # Replace with your actual API key
    keywords = query  # Example keywords
    articles = get_articles(api_key, keywords)
    formatted_articles = format_articles(articles)
    
    # Output the formatted articles as JSON
    json_output = json.dumps(formatted_articles, indent=4)

    # Save the JSON output to a file
    with open('./documents/articles.json', 'w') as json_file:
        json_file.write(json_output)

if __name__ == "__main__":
    generate_article_json('segment everything')