# built atop of  https://huggingface.co/spaces/kiyer/pathfinder/blob/main/pages/3_answering_questions.py


# require embedder for queries
from sentence_transformers import SentenceTransformer
model_embed = SentenceTransformer('distilbert-base-nli-mean-tokens') # this needs to be placed near the top of imports or else it seg faults 
import arxiv

import os
import datetime
import faiss
import streamlit as st
import feedparser
import urllib
import cloudpickle as cp
import pickle
from urllib.request import urlopen
from summa import summarizer
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

from dotenv import load_dotenv
load_dotenv()

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain.llms import OpenAI

# from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


import shutil 



######################## Setting up LLM - OLD
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["AZURE_ENDPOINT"] = st.secrets["endpoint1"]
# os.environ["OPENAI_API_KEY"] = st.secrets["key1"]
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"


# embeddings = AzureOpenAIEmbeddings(
#     deployment="embedding",
#     model="text-embedding-ada-002",
#     azure_endpoint=st.secrets["endpoint1"],
# )

# llm = AzureChatOpenAI(
#         deployment_name="gpt4_small",
#         openai_api_version="2023-12-01-preview",
#         azure_endpoint=st.secrets["endpoint2"],
#         openai_api_key=st.secrets["key2"],
#         openai_api_type="azure",
#         temperature=0.
#     )

######################## Setting up LLM - NEW 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') 
# get embeddings model 
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.0
)





######################## Get embeddings - old 
# @st.cache_data
# def get_feeds_data(url):
#     # data = cp.load(urlopen(url))
#     with open(url, "rb") as fp:
#         data = pickle.load(fp)
#     st.sidebar.success("Loaded data")
#     return data



# feeds_link = "https://drive.google.com/uc?export=download&id=1-IPk1voyUM9VqnghwyVrM1dY6rFnn1S_"
# embed_link = "https://dl.dropboxusercontent.com/s/ob2betm29qrtb8v/astro_ph_ga_feeds_ada_embedding_18-Apr-2023.pkl?dl=0"
# dateval = "27-Jun-2023"
# dateval = "18-Apr-2023"

# feeds_link = "local_files/astro_ph_ga_feeds_upto_"+dateval+".pkl"
# embed_link = "local_files/astro_ph_ga_feeds_ada_embedding_"+dateval+".pkl"
# gal_feeds = get_feeds_data(feeds_link) # -> len() -> 1000 
# arxiv_ada_embeddings = get_feeds_data(embed_link)
# arxiv_ada_embeddings.shape -> (22747, 1536) -> 22747 articles with 1536 embedding dictionary size? 



# @st.cache_data
# def get_embedding_data(url):
#     # data = cp.load(urlopen(url))
#     with open(url, "rb") as fp:
#         data = pickle.load(fp)
#     st.sidebar.success("Fetched data from API!")
#     return data

# # url = "https://drive.google.com/uc?export=download&id=1133tynMwsfdR1wxbkFLhbES3FwDWTPjP"
# url = "local_files/astro_ph_ga_embedding_"+dateval+".pkl"
# e2d = get_embedding_data(url)
# # e2d, _, _, _, _ = get_embedding_data(url)


######################## Get embeddings - new 
# TBC
from utils import *
sources_df, all_titles, final_2d_embeddings,embeddings_all = get_embeddings_from_file()
e2d=final_2d_embeddings
# extract additional information 
all_links = list(sources_df.link)
embeddings_all = np.array(embeddings_all) # np.array(embeddings_all).shape -> (80, 768) -> 80 articles * 768 embedding size 





######################## process embeddings and metadata - old 
# # the information that is extracted from the old embeddings is: 
#     # title, arxivid, link, authors. 
# ctr = -1
# num_chunks = len(gal_feeds)
# all_text, all_titles, all_arxivid, all_links, all_authors = [], [], [], [], []

# for nc in range(num_chunks):

#     for i in range(len(gal_feeds[nc].entries)):
#         text = gal_feeds[nc].entries[i].summary
#         text = text.replace('\n', ' ')
#         text = text.replace('\\', '')
#         all_text.append(text)
#         all_titles.append(gal_feeds[nc].entries[i].title)
#         all_arxivid.append(gal_feeds[nc].entries[i].id.split('/')[-1][0:-2])
#         all_links.append(gal_feeds[nc].entries[i].links[1].href)
#         all_authors.append(gal_feeds[nc].entries[i].authors)

# d = arxiv_ada_embeddings.shape[1]                           # dimension
# nb = arxiv_ada_embeddings.shape[0]                      # database size
# xb = arxiv_ada_embeddings.astype('float32')
# index = faiss.IndexFlatL2(d)
# index.add(xb)


# model_data = [arxiv_ada_embeddings, embeddings, all_titles, all_text, all_authors]

######################## process embeddings and metadata - new 


d = embeddings_all.shape[1]                           # dimension
nb = embeddings_all.shape[0]                      # database size
xb = embeddings_all.astype('float32')
index = faiss.IndexFlatL2(d)
index.add(xb)

model_data = [embeddings_all, model_embed, all_titles,all_links]



def run_simple_query(search_query = 'all:sed+fitting', max_results = 10, start = 0, sort_by = 'lastUpdatedDate', sort_order = 'descending'):
    """
        Query ArXiv to return search results for a particular query
        Parameters
        ----------
        query: str
            query term. use prefixes ti, au, abs, co, jr, cat, m, id, all as applicable.
        max_results: int, default = 10
            number of results to return. numbers > 1000 generally lead to timeouts
        start: int, default = 0
            start index for results reported. use this if you're interested in running chunks.
        Returns
        -------
        feed: dict
            object containing requested results parsed with feedparser
        Notes
        -----
            add functionality for chunk parsing, as well as storage and retreival
        """

    base_url = 'http://export.arxiv.org/api/query?';
    query = 'search_query=%s&start=%i&max_results=%i&sortBy=%s&sortOrder=%s' % (search_query,
                                                     start,
                                                     max_results,sort_by,sort_order)

    response = urllib.request.urlopen(base_url+query).read()
    feed = feedparser.parse(response)
    return feed

def find_papers_by_author(auth_name):

    doc_ids = []
    for doc_id in range(len(all_authors)):
        for auth_id in range(len(all_authors[doc_id])):
            if auth_name.lower() in all_authors[doc_id][auth_id]['name'].lower():
                print('Doc ID: ',doc_id, ' | arXiv: ', all_arxivid[doc_id], '| ', all_titles[doc_id],' | Author entry: ', all_authors[doc_id][auth_id]['name'])
                doc_ids.append(doc_id)

    return doc_ids

def faiss_based_indices(input_vector, nindex=10):
    xq = input_vector.reshape(-1,1).T.astype('float32')
    D, I = index.search(xq, nindex)
    return I[0], D[0]

def list_similar_papers_v2(model_data,
                        doc_id = [], input_type = 'doc_id',
                        show_authors = False, show_summary = False,
                        return_n = 10):

    # arxiv_ada_embeddings, embeddings, all_titles, all_abstracts, all_authors = model_data
    arxiv_ada_embeddings, embeddings, all_titles, all_links = model_data
    

    if input_type == 'doc_id':
        print('Doc ID: ',doc_id,', title: ',all_titles[doc_id])
#         inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        inferred_vector = arxiv_ada_embeddings[doc_id,0:]
        start_range = 1
    elif input_type == 'arxiv_id':
        print('ArXiv id: ',doc_id)
        arxiv_query_feed = run_simple_query(search_query='id:'+str(doc_id))
        if len(arxiv_query_feed.entries) == 0:
            print('error: arxiv id not found.')
            return
        else:
            print('Title: '+arxiv_query_feed.entries[0].title)
            if isinstance(embeddings, SentenceTransformer):
                inferred_vector = np.array(embeddings.encode(arxiv_query_feed.entries[0].summary))
            else:
                # default behaviour (open ai embeddings)
                inferred_vector = np.array(embeddings.embed_query(arxiv_query_feed.entries[0].summary))
        start_range = 0
    elif input_type == 'keywords':
        if isinstance(embeddings, SentenceTransformer):
            inferred_vector = np.array(embeddings.encode(doc_id))
        else:
            # default behaviour (open ai embeddings)
            inferred_vector = np.array(embeddings.embed_query(doc_id))
        start_range = 0
    else:
        print('unrecognized input type.')
        return
    
    sims, dists = faiss_based_indices(inferred_vector, return_n+2)
    textstr = ''
    abstracts_relevant = []
    fhdrs = []
    ids = []

    for i in range(start_range,start_range+return_n):

        ####### Add information about found queries - old 
        # abstracts_relevant.append(all_text[sims[i]])
        # fhdr = str(sims[i])+'_'+all_authors[sims[i]][0]['name'].split()[-1] + all_arxivid[sims[i]][0:2] +'_'+ all_arxivid[sims[i]]
        # fhdrs.append(fhdr)
        # textstr = textstr + str(i+1)+'. **'+ all_titles[sims[i]] +'** (Distance: %.2f' %dists[i]+')   \n'
        # textstr = textstr + '**ArXiv:** ['+all_arxivid[sims[i]]+'](https://arxiv.org/abs/'+all_arxivid[sims[i]]+')  \n'
        
        ####### Add information about found queries - new 
        
        # fhdr = str(sims[i])+'_'+all_links[sims[i]]
        fhdrs.append(all_links[sims[i]])
        textstr = textstr + str(i+1)+'. **'+ all_titles[sims[i]] +'** (Distance: %.2f' %dists[i]+')   \n'
        textstr = textstr + '(' + all_links[sims[i]] + ')  \n'
        
        
        
        
        if show_authors == True:
            textstr = textstr + '**Authors:**  '
            temp = all_authors[sims[i]]
            for ak in range(len(temp)):
                if ak < len(temp)-1:
                    textstr = textstr + temp[ak].name + ', '
                else:
                    textstr = textstr + temp[ak].name + '   \n'
        if show_summary == True:
            textstr = textstr + '**Summary:**  '
            text = all_text[sims[i]]
            text = text.replace('\n', ' ')
            textstr = textstr + summarizer.summarize(text) + '  \n'
        if show_authors == True or show_summary == True:
            textstr = textstr + ' '
        textstr = textstr + '  \n'
        ids.append(sims[i])
    #return textstr, abstracts_relevant, fhdrs, sims
    return textstr, fhdrs, sims, ids


def generate_chat_completion(messages, model="gpt-4", temperature=1, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_textstr(i, show_authors=False, show_summary=False):
    textstr = ''
    textstr = '**'+ all_titles[i] +'**   \n'
    textstr = textstr + '**ArXiv:** ['+all_arxivid[i]+'](https://arxiv.org/abs/'+all_arxivid[i]+')  \n'
    if show_authors == True:
        textstr = textstr + '**Authors:**  '
        temp = all_authors[i]
        for ak in range(len(temp)):
            if ak < len(temp)-1:
                textstr = textstr + temp[ak].name + ', '
            else:
                textstr = textstr + temp[ak].name + '   \n'
    if show_summary == True:
        textstr = textstr + '**Summary:**  '
        text = all_text[i]
        text = text.replace('\n', ' ')
        textstr = textstr + summarizer.summarize(text) + '  \n'
    if show_authors == True or show_summary == True:
        textstr = textstr + ' '
    textstr = textstr + '  \n'

    return textstr


def run_rag(query, return_n = 10, show_authors = False, show_summary = False):


    #sims, absts, fhdrs, simids = list_similar_papers_v2(model_data,
    sims, fhdrs, simids,ids = list_similar_papers_v2(model_data,
                                  doc_id = query,
                                  input_type='keywords',
                                  show_authors = show_authors, show_summary = show_summary,
                                  return_n = return_n)
    
    
    
    
    # ########################## Create chrome DB to read through selected abstracts - old 
    # temp_abst = ''
    # loaders = []
    # for i in range(len(absts)):
    #     temp_abst = absts[i]

    #     try:
    #         text_file = open("absts/"+fhdrs[i]+".txt", "w")
    #     except:
    #         os.mkdir('absts')
    #         text_file = open("absts/"+fhdrs[i]+".txt", "w")
    #     n = text_file.write(temp_abst)
    #     text_file.close()
    #     loader = TextLoader("absts/"+fhdrs[i]+".txt")
    #     loaders.append(loader)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    # splits = text_splitter.split_documents([loader.load()[0] for loader in loaders])
    # vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # retriever = vectorstore.as_retriever()
    
    
    ########################## [temp debugging] Create .txt files with all abstracts in separate folder - NEW 

    # if os.path.exists('all_arxiv_absts'):
    #     shutil.rmtree('all_arxiv_absts') # remove the previous folder results  (to avoid reading wrong files)
    # os.makedirs('all_arxiv_absts')
    # temp_links= model_data[-1]
    # for i, paper_url_temp in enumerate(temp_links):
        
    #     # paper_url = fhdrs[i] 
    #     if not paper_url_temp.startswith('http://arxiv.org'): 
    #         continue
        
    #     # Extract the paper ID from the URL
    #     paper_id = paper_url_temp.split('/')[-1]

    #     # Create a search for the specific paper ID
    #     search = arxiv.Search(id_list=[paper_id])
        
    #     # Fetch the result using the default client
    #     client = arxiv.Client()
    #     papers = []
    #     for paper in client.results(search):
    #         papers.append(paper)
    #     if not papers:
    #         continue
        
    #     # paper.download_pdf(dirpath="./absts", filename=f"{i}.pdf")
    #     text = paper.summary
    #     text = text.replace('\n', ' ')
    #     text = text.replace('\\', '')
        
    #     # save into folder (required for llangchain chunking)
    #     fname = "all_arxiv_absts/"+str(i)+".txt"
    #     text_file = open(fname, "w")
    #     n = text_file.write(text)
    #     text_file.close()        
        
                
        
        
    
    ########################## Create chrome DB to read through selected abstracts - NEW 
    # download all relevant documents 
    
    if os.path.exists('absts'):
        shutil.rmtree('absts') # remove the previous folder results  (to avoid reading wrong files)   
    os.makedirs('absts') 

        
    loaders = []
    matched_papers = {}
    for paper_url,i in zip(fhdrs,ids): 
        # paper_url = fhdrs[i] 
        if not paper_url.startswith('http://arxiv.org'): 
            continue
        
        
        # Extract the paper ID from the URL
        paper_id = paper_url.split('/')[-1]

        # Create a search for the specific paper ID
        search = arxiv.Search(id_list=[paper_id])
        
        # Fetch the result using the default client
        client = arxiv.Client()
        papers = []
        for paper in client.results(search):
            papers.append(paper)
        if not papers:
            continue
        
        
                
        # paper.download_pdf(dirpath="./absts", filename=f"{i}.pdf")
        text = paper.summary
        text = text.replace('\n', ' ')
        text = text.replace('\\', '')
        
        # save into folder (required for llangchain chunking)
        fname = "absts/"+str(i)+".txt"
        text_file = open(fname, "w")
        n = text_file.write(text)
        text_file.close()
        loader = TextLoader(fname)
        loaders.append(loader)     
        
        # get all the info about the paper and save for later 
        matched_papers[str(i)] = paper 
        
            
           
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_documents([loader.load()[0] for loader in loaders])
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
        
    

    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context from the literature to answer the question.
    If you don't know the answer, just say that you don't know.
    Use six sentences maximum and keep the answer concise.
    {context}
    Question: {question}
    Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    rag_answer = rag_chain_with_source.invoke(query)

    st.markdown('### User query: '+query)

    st.markdown(rag_answer['answer'])
    
    
    
    # extract title and names from matched sources 
    ids_matched=[]
    from IPython import embed; embed()
    for i in range(len(rag_answer['context'])):
        # find id of document 
        fullfilename=rag_answer['context'][i].metadata['source']
        id_match = os.path.basename(fullfilename).replace('.txt','')
        ids_matched.append(id_match)
        
        # get title and link 
        matched_paper = matched_papers[id_match] 
        
        # here we extract some info to 
        rag_answer['context'][i].metadata['arxiv'] = matched_paper
    
    # opstr = '#### Primary sources: \n'
    # srcnames = []
    # for i in range(len(rag_answer['context'])):
    #     print(rag_answer['context'][i].metadata['source'])
    #     srcnames.append(rag_answer['context'][i].metadata['source'])



    # srcnames = np.unique(srcnames)
    # srcindices = []
    # for i in range(len(srcnames)):
    #     temp = srcnames[i].split('_')[1]
    #     srcindices.append(int(srcnames[i].split('_')[0].split('/')[1]))
    #     if int(temp[-2:]) < 40:
    #         temp = temp[0:-2] + ' et al. 20' + temp[-2:]
    #     else:
    #         temp = temp[0:-2] + ' et al. 19' + temp[-2:]
    #     temp = '['+temp+']('+all_links[int(srcnames[i].split('_')[0].split('/')[1])]+')'
    #     st.markdown(temp)
    # abs_indices = np.array(srcindices)
    
    
    
    abs_indices = list(np.unique([int(i) for i in ids_matched]))

    fig = plt.figure(figsize=(9,9))
    plt.scatter(e2d[0:,0], e2d[0:,1],s=2)
    plt.scatter(e2d[ids,0], e2d[ids,1],s=30)
    plt.scatter(e2d[abs_indices,0], e2d[abs_indices,1],s=100,color='k',marker='d')
    plt.title('localization for question: '+query)
    st.pyplot(fig)

    st.markdown('\n #### List of relevant papers:')
    st.markdown(sims)
    
    return rag_answer

dateval = "26-May-2024?"
st.title('ArXiv-based question answering')
st.markdown('[Includes papers up to: `'+dateval+'`]')
st.markdown('Concise answers for questions using arxiv abstracts + GPT-4. You might need to wait for a few seconds for the GPT-4 query to return an answer (check top right corner to see if it is still running).')
st.markdown('The answers are followed by relevant source(s) used in the answer, a graph showing which part of the embedding manifold it drew the answer from (tightly clustered points generally indicate high quality/consensus answers) followed by a bunch of relevant papers used by the RAG to compose the answer.')
st.markdown('If this does not satisfactorily answer your question or rambles too much, you can also try ... <TBD>')

query = st.text_input('Your question here:',
value="What can you tell me about LoFAs?")
return_n = st.slider('How many papers should I show?', 1, 30, 3)

sims = run_rag(query, return_n = return_n)