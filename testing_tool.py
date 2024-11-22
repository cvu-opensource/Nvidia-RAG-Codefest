from IPython.display import Image, display
import getpass, os, base64
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from collections.abc import Iterable
from random import randint
from langgraph.prebuilt import InjectedState
from langchain_core.messages.tool import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import requests

# tvly_api_key = getpass.getpass("Enter your tvly API key: ")
# assert tvly_api_key.startswith("tvly-"), f"{tvly_api_key[:5]}... is not a valid key"
os.environ["TAVILY_API_KEY"] = "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"

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", base_url="http://10.149.8.40:8000/v1")

def RAG_FROM_DATABASE(query: str) -> str:
    """
    A Retrieval-Augmented Generation (RAG) tool to answer a query using:
    1. nv-embedqa-e5-v5 embeddings to embed the query.
    2. Milvus vectorstore for similarity-based retrieval.
    3. nv-rerankqa-mistral-4b-v3 reranker for improved result relevance.
    4. Outputs the best response from the reranker.
    
    Parameters:
        query (str): The user's input query.
        
    Returns:
        str: The most relevant information retrieved from the database.
        
    """
    
    # Step 1: Specifiying embedding client
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://localhost:8001/v1", truncate="NONE")
    
    # Step 2: Embed the query
    embeded_query =  embedder.embed_query(query)
    #print(embeded_query)
    
    # Step 3: Specifying vectorstore and sending
    DATA_API = "http://10.149.8.40:9998"
    
    # Step 4: Crafting request to milvus to get k(x) result. Output(list of results) = ["result1", "result2", ...]
    milvus_response = requests.post(                    
        DATA_API + "/search",
        json={"question": query, "embedded_question": embeded_query, "top_k":3},
        #files={"image": uploaded_image.getvalue() if uploaded_image else None}
        files = None
        ).json()
    
    retrieved_results = milvus_response["results"]
    
    # Step 5: Specifying reranking model
    rerank_client = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3", base_url="http://localhost:8002")
    
    if milvus_response:
        # prepare documents from vectorstore response
        '''
        documents = [
            {
                "content": passage.page_content,
                "metadata": passage.metadata
            }
            for passage in milvus_response
        ]
        '''
        
        documents = [Document(page_content=key)
            for key, value in retrieved_results.items()
        ]
        
        # Perform reranking and sorting the results with the scores in decending order
        rerank_response = sorted(rerank_client.compress_documents(query=query, documents=documents)["documents"], key=lambda x: x["score"], reverse=True)
        # print(rerank_response)
    
    # Step 6: Select top result (returns to model top result and cite)
        if rerank_response:
            top_result = rerank_response[0]
            output = f"Top Result: {top_result.content}\n\nCite: {top_result.metadata}"
            return output
    else:
        return "No relevant results found for the query"
    
test=RAG_FROM_DATABASE("query")
print(test)