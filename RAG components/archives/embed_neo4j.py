from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
import requests
import pandas as pd
import urllib.parse  # To handle URL joining
import fitz
import tqdm

import numpy as np
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup, SoupStrainer

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader, CSVLoader, UnstructuredTSVLoader, TextLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document

from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

from utils.datahandler import DataHandler

ollama_emb = OllamaEmbeddings(
    model="llama3.1",
)

llm = ChatNVIDIA(base_url="http://10.149.8.40:8000/v1", model="meta/llama-3.1-8b-instruct")
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://10.149.8.40:8001/v1", truncate="END")

        
websites = [
    "https://www.iras.gov.sg",
    "https://www.mom.gov.sg",
    "https://www.acra.gov.sg",
    "https://singaporelegaladvice.com",
    "https://www.ipos.gov.sg",
    "https://www.enterprisesg.gov.sg",
    "https://www.skillsfuture.gov.sg",
    "https://www.hsa.gov.sg",
    "https://www.sfa.gov.sg"
]

# everything after this is specifically for neo4j. Please remember to eventually merge this DataHandler class with the notebook's. Or make it into a package?

datahandler = DataHandler(embedder=embedder)
datahandler.from_cached_websites("/raw_web_data/html")

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
AUTH = ("neo4j", "cringemfpassword") # (cringe admin username and pw)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity() #vett connection, if shit dont work gg buddy
    driver.close()
    print("closed driver!!")
    
graph = Neo4jGraph(
    url="neo4j://10.149.8.40:7687",
    username="neo4j",
    password="cringemfpassword",
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)


docs = []
for text, sauce in tqdm(zip(datahandler.textual_data, datahandler.textual_metadata)):
    
    with open("temp.txt", 'w') as f:
        f.write(text)
    loader = TextLoader("temp.txt")

    documents = loader.load()
    for document in documents:
        document.metadata = sauce
    docs = text_splitter.split_documents(documents)
    
    llm = OllamaFunctions(model="llama3.1", temperature=0, format="json")

    llm_transformer = LLMGraphTransformer(llm=llm)

    graph_documents = llm_transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )