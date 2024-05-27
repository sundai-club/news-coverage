**A [Sundai Club](https://sundai.club/) Project**

_This project is a work in progress_

## Demo
[http://ai-news-hound.sundai.club/](http://ai-news-hound.sundai.club/)

## Overview
From a "Journalism Hack" week. This project aims to visualize and compare topic data from various sources, including Reddit, Arxiv, Hacker News, and news articles. By scraping and processing data from these sources, the tool generates text embeddings using the SentenceTransformers library. These embeddings are then visualized using  UMAP (Uniform Manifold Approximation and Projection - a way to make large embeddings just two-dimensions so they can be compared) to identify topic clusters and information 'deserts'—areas with sparse or minimal discussion.

The visualization shows color-coded dots to indicate the source of each topic and green hexagons to highlight topics that are not being actively reported on. This tool is designed to help users easily explore and identify gaps in information coverage across different media sources.

![visualization showing areas of ](https://raw.githubusercontent.com/sundai-club/news-coverage/main/readme-assets/visualization.png)

## Visualization Details
When looking at the visualization:
*   Dot colors show the source (e.g., Reddit, Arxiv, news articles).
*   Hexagon color indicates if something is being reported on (green means it's not).


## How It Works
We use [SentenceTransformers](https://huggingface.co/sentence-transformers) ([pip library documentation](https://pypi.org/project/sentence-transformers/)) to generate embeddings, specifically the `all-MiniLM-L6-v2` model. These embeddings are created out of text for specific topics obtained from various APIs. The embeddings are then visualized in a UMAP using Bokeh and Streamlit.

## Usage

### Running the Visualization
To see the visualization, navigate to the `streamlit` folder and run:

```
streamlit run app.py
```

Make sure to install the necessary requirements first. It is recommended to use a Python virtual environment for this.

#### Create a virtual environment:
 macOS/Linux:
```
python -m venv env
source env/bin/activate
```

Windows:
```
python -m venv env
.\env\Scripts\activate
```

#### Install the requirements
```
pip install -r requirements.txt
```
The interface will load at http://localhost:8501/.

### Loading New Topics
To load your own data, look under the **data\_source\_apis** folder. You can run **topic\_embeddings.py** to use the Reddit, Mediastack, and Arxiv APIs (each in their own files). 

Set up developer keys for:
*   Reddit: [Reddit Apps](https://old.reddit.com/prefs/apps)
*   Mediastack: [Mediastack Documentation](https://mediastack.com/documentation)

In addition to the above, there's a **hacker-news.js** file you'll need to run separately. Follow these steps:

#### Install dependencies:
```
npm install express axios
```

#### Start the server:
```
node index.js
```

#### Load articles:
Use the **/search** endpoint with a query parameter to search for articles. For example:
```
http://localhost:3000/search?q=agents
```
This will return 10 results, each containing the title of the post, the full body of comments as text, source, link, publication date, and type, formatted as a JSON response.

#### Updating Embeddings in Visualization
After generating new JSON files with the APIs, update the file names loaded into the visualization by replacing the **inputs** array in the **get\_embeddings\_from\_file** function in **app.py**:
```
inputs = ["embeddings_AIAgents.json", "embeddings_AIAssistedHealthcare.json", "embeddings_AIDrivenPortfolioManagement.json", "embeddings_AIPublicPolicies.json"]
```
Replace with the appropriate file names.

### Experimental Notebooks
In the `notebooks` folder, you will find our explorations in Jupyter notebooks for different models and UMAP settings.