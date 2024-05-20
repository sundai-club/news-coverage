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
        
    return articles

def generate_arxiv_json(query):
    articles = get_arxiv_articles(query)
    
    # Output the formatted articles as JSON
    json_output = json.dumps(articles, indent=4)
    
    # Save the JSON output to a file
    with open('./documents/papers.json', 'w') as json_file:
        json_file.write(json_output)
