/**
 * Hacker News API Aggregator
 * 
 * This API allows you to search for Hacker News articles and return the full discussion,
 * including nested comments, in a nicely formatted JSON response.
 * 
 * Usage:
 * 
 * 1. Install dependencies:
 *    npm install express axios
 * 
 * 2. Start the server:
 *    node index.js
 * 
 * 3. Search for articles:
 *    Use the /search endpoint with a query parameter to search for articles.
 *    Example: http://localhost:3000/search?q=agents
 * 
 *    This will return 10 results, each containing the title of the post, the full body of 
 *    comments as text, source, link, publication date, and type, all formatted as a nicely 
 *    formatted JSON response.
 */

const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 3000;
const HN_API_BASE_URL = 'https://hacker-news.firebaseio.com/v0';
const ALGOLIA_SEARCH_API_URL = 'https://hn.algolia.com/api/v1';

// Function to fetch a comment and its nested replies recursively
const fetchComment = async (commentId) => {
    try {
        const commentResponse = await axios.get(`${HN_API_BASE_URL}/item/${commentId}.json`);
        const commentData = commentResponse.data;

        if (commentData.kids && commentData.kids.length > 0) {
            const childCommentPromises = commentData.kids.map(childId => fetchComment(childId));
            const childComments = await Promise.all(childCommentPromises);
            commentData.children = childComments;
        } else {
            commentData.children = [];
        }

        return commentData;
    } catch (error) {
        console.error('Error fetching comment:', error);
        return null;
    }
};

// Function to fetch the full discussion for a given Hacker News item ID and aggregate comments into a single text
const fetchFullDiscussion = async (itemId) => {
    try {
        const itemResponse = await axios.get(`${HN_API_BASE_URL}/item/${itemId}.json`);
        const itemData = itemResponse.data;

        if (!itemData.kids || itemData.kids.length === 0) {
            return { item: itemData, comments: '' };
        }

        const commentPromises = itemData.kids.map(commentId => fetchComment(commentId));
        const comments = await Promise.all(commentPromises);

        // Aggregate all comments into a single text string
        const aggregateComments = (comments) => {
            let text = '';
            comments.forEach(comment => {
                if (comment.text) {
                    text += comment.text + ' ';
                }
                if (comment.children && comment.children.length > 0) {
                    text += aggregateComments(comment.children);
                }
            });
            return text;
        };

        const fullCommentsText = aggregateComments(comments);

        return { item: itemData, comments: fullCommentsText.trim() };
    } catch (error) {
        console.error('Error fetching full discussion:', error);
        return { item: null, comments: '' };
    }
};

// Root route
app.get('/', (req, res) => {
    res.send('Welcome to the Hacker News API! Use /search to search for articles and return full discussions.');
});

// Search route
app.get('/search', async (req, res) => {
    const query = req.query.q;
    if (!query) {
        return res.status(400).json({ error: 'Query parameter q is required' });
    }

    try {
        // Search for articles using the Algolia API
        const searchResponse = await axios.get(`${ALGOLIA_SEARCH_API_URL}/search`, {
            params: {
                query: query,
                tags: 'story', // Search only stories (not comments, jobs, etc.)
                hitsPerPage: 10 // Return 10 results
            }
        });

        const searchResults = searchResponse.data.hits;
        const discussionPromises = searchResults.map(hit => fetchFullDiscussion(hit.objectID));
        const discussions = await Promise.all(discussionPromises);

        // Format the results
        const results = discussions.map(discussion => ({
            title: discussion.item.title,
            text: discussion.comments,
            source: 'HackerNews',
            link: discussion.item.url || `https://news.ycombinator.com/item?id=${discussion.item.id}`,
            published_at: new Date(discussion.item.time * 1000).toISOString(),
            type: 'social'
        }));

        res.setHeader('Content-Type', 'application/json');
        res.send(JSON.stringify(results, null, 2)); // Pretty-print JSON with 2-space indentation
    } catch (error) {
        console.error('Error searching articles:', error);
        res.status(500).json({ error: 'An error occurred while searching for articles' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});