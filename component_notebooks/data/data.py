# import os
# import requests
# import pandas as pd
# import urllib.parse 
# import fitz

# from tqdm import tqdm
# from io import StringIO
# from bs4 import BeautifulSoup, SoupStrainer
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader, CSVLoader, UnstructuredTSVLoader

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import hashlib

from db import MilvusDB

app = FastAPI()

# Initialize MilvusDB instance
database = MilvusDB("http://localhost:19530")  
database.create_collection("Documents", dimensions=1024)  


class InsertRequest(BaseModel):
    documents: List[str]
    embeddings: List[List[float]]
    metadata: Optional[List[dict]] = None

class SearchRequest(BaseModel):
    question: str
    embedded_question: List[float]
    top_k: Optional[int] = 5

class FeedbackRequest(BaseModel):
    doc_id: str
    feedback: str

@app.post("/insert")
async def insert_data(request: InsertRequest):
    try:
        database.insert_data(
            original=request.documents,
            embedded=request.embeddings,
            metadata=request.metadata,
        )
        return {"status": "success", "message": "Data inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_data(request: SearchRequest):
    try:
        results = database.retrieve_data(
            embedded_question=request.embedded_question,
            question=request.question,
            top_k=request.top_k,
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        database.update_document_scores(doc_id=request.doc_id, feedback=request.feedback)
        return {"status": "success", "message": "Feedback processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_records")
async def get_all_records(limit: int = 1000):
    try:
        records = database.get_all_records(limit=limit)
        return {"status": "success", "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))