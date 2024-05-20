import arxiv
import json

def get_arxiv_articles(query, max_results=30):
    print("called")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    articles = []
    for result in search.results():
        article = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'text': result.summary,
            #'published': result.published,
            'link': result.entry_id,
            'type': "paper"
        }
        articles.append(article)
    print(articles)
        
    return articles


# Replace 'your topic' with the topic you are interested in
topic = "ai agents"
articles = get_arxiv_articles(topic)

# Output the formatted articles as JSON
json_output = json.dumps(articles, indent=4)
print(json_output)

    # Save the JSON output to a file
with open('papers.json', 'w') as json_file:
    json_file.write(json_output)

# Print the articles
for idx, article in enumerate(articles, 1):
    print(f"Article {idx}:")
    print(f"Title: {article['title']}")
    print(f"Authors: {', '.join(article['authors'])}")
    print(f"Published: {article['published']}")
    print(f"URL: {article['url']}")
    print(f"Summary: {article['summary']}\n")
