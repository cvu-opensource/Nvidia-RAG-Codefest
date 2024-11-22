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

# datahandler.scrape_pdfs('/raw_data/pdf')
# datahandler.scrape_csvs('project/raw_data/csvs')


# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
AUTH = ("neo4j", "cringemfpassword") # (cringe admin username and pw)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity() #vett connection, if shit dont work gg buddy
    driver.close()
    print("closed driver!!")
    
# llm_transformer = LLMGraphTransformer(llm=llm)
    
graph = Neo4jGraph(
    url="neo4j://10.149.8.40:7687",
    username="neo4j",
    password="cringemfpassword",
)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)

# def __enter__(self):
#         return self

# def __exit__(self, exc_type, exc_val, exc_tb):
#         self._driver.close()

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
    # print(graph_documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    
#     embeddings = OllamaEmbeddings(
#         model="mxbai-embed-large",
#     )

#     vector_index = Neo4jVector.from_existing_graph(
#         embeddings,
#         search_type="hybrid",
#         node_label="Document",
#         text_node_properties=["text"],
#         embedding_node_property="embedding",
#         url="neo4j://10.149.8.40:7687",
#         username="neo4j",
#         password="cringemfpassword",
#     )
#     vector_retriever = vector_index.as_retriever()
    
#     driver = GraphDatabase.driver(
#         uri="neo4j://10.149.8.40:7687",
#         auth = ("neo4j",
#                "cringemfpassword"))

#     def create_fulltext_index(tx):
#         query = '''
#         CREATE FULLTEXT INDEX `fulltext_entity_id` 
#         FOR (n:__Entity__) 
#         ON EACH [n.id];
#         '''
#         tx.run(query)

#     # Function to execute the query
#     def create_index():
#         with driver.session() as session:
#             session.execute_write(create_fulltext_index)
#             print("Fulltext index created successfully.")

#     # Call the function to create the index
#     try:
#         create_index()
#     except:
#         pass

#     # Close the driver connection
#     driver.close()
    
#     class Entities(BaseModel):
#         """Identifying information about entities."""

#         names: list[str] = Field(
#             ...,
#             description="All the person, organization, or business entities that "
#             "appear in the text",
#         )

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are extracting all the person, organizations, and business entities that appear in the text.",
#             ),
#             (
#                 "human",
#                 "Use the given format to extract information from the following "
#                 "input: {user_input}",
#             ),
#         ]
#     )
    
#     def placeholder():
#         """
#         Never call this function.
#         """
#         return "Get fucked kiddo"
    
#     llm_with_tools = llm.bind_tools([placeholder, placeholder, placeholder])

#     entity_chain = llm.with_structured_output(Entities)
#     # print(chain.invoke("What is IRAS and what are its relations?"))
#     try:
#         print(entity_chain.invoke("nigger?"))
#     except:
#         print("unable to interpret prompt given to graph RAG, returning empty string as context")
    
#     def generate_full_text_query(input: str) -> str:
#         words = [el for el in remove_lucene_chars(input).split() if el]
#         if not words:
#             return ""
#         full_text_query = " AND ".join([f"{word}~2" for word in words])
#         # print(f"Generated Query: {full_text_query}")
#         return full_text_query.strip()


#     # Fulltext index query
#     def graph_retriever(question: str) -> str:
#         """
#         Collects the neighborhood of entities mentioned
#         in the question
#         """
#         result = ""
#         entities = entity_chain.invoke(question)
#         for entity in entities.names:
#             response = graph.query(
#                 """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
#                 YIELD node,score
#                 CALL {
#                   WITH node
#                   MATCH (node)-[r:!MENTIONS]->(neighbor)
#                   RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
#                   UNION ALL
#                   WITH node
#                   MATCH (node)<-[r:!MENTIONS]-(neighbor)
#                   RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
#                 }
#                 RETURN output LIMIT 50
#                 """,
#                 {"query": entity},
#             )
#             result += "\n".join([el['output'] for el in response])
#         return result
    
#     print(graph_retriever("How are are Smses and Scamalert.Sg related?"))

    
#     db = Neo4jVector.from_documents(
#     docs, ollama_emb, url="neo4j://10.149.8.40:7687", username="neo4j", password="cringemfpassword"
# )
    # print(type(sauce))
    # doc = Document(page_content=text, metadata=sauce)
    # # print(type(doc))
    # print(doc)
    # split_doc = text_splitter.split_documents([doc])
    # print(split_doc)
    # # documents = text_splitter.split_documents([text])
    # # for document in documents:
    # #     document.metadata = sauce
    # docs.extend(split_doc)
    # # doc = Document(page_content=text, metadata=sauce)
    
