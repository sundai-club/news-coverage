import praw
import json
import os
from dotenv import load_dotenv

load_dotenv()


# Replace these with your own Reddit app credentials
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
user_agent = 'journalresearch/0.1 by Economy-Jellyfish-87'

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="" #username
)

def search_reddit_posts(query, limit=10):
    search_results = reddit.subreddit('all').search(query, limit=limit)

    posts = []
    for post in search_results:
        submission = reddit.submission(id=post.id)
        submission.comments.replace_more(limit=0)  # Load all comments

        comments = [post.selftext]
        for top_level_comment in submission.comments.list():
            comments.append(top_level_comment.body)

        post_info = {
            'title': post.title,
            # 'author': str(post.author),
            # 'created_utc': post.created_utc,
            # 'score': post.score,
            'text': '\n Comment: '.join(comments),
            'type': 'reddit',
            'link': post.url,
            # 'num_comments': post.num_comments,
            # 'subreddit': str(post.subreddit),
            # 'selftext': post.selftext,
            # 'comments': comments
        }
        posts.append(post_info)
    
    return posts

def save_posts_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generate_reddit_json(query, limit=25):
    posts = search_reddit_posts(query, limit)
    save_posts_to_json(posts, '../streamlit-app/documents/fow/reddit.json')

if __name__ == "__main__":
    # subreddit_name = 'learnpython'  # Example subreddit
    query = 'AI future of work'  # Example query
    # limit = 5  # Number of posts to retrieve
    generate_reddit_json(query)