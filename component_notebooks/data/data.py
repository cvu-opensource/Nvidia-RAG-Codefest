# API packages
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File

# Nvidia packages
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# Other scripts
from database import MilvusDB
from datahandler import DataHandler

app = FastAPI()

# Initialise models used
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://localhost:8001/v1", truncate="END")
text_splitter = CharacterTextSplitter(chunk_size=2048, separator=" ", chunk_overlap=64)

# Initialise DataHandler instance 
datahandler = DataHandler(embedder, text_splitter)

# Initialize MilvusDB instance
database = MilvusDB("http://10.149.8.40:19530")  
database.collection_name = "Documents"
if not database.client.has_collection(database.collection_name):
    database.create_collection(database.collection_name, dimensions=1024) 

class ProcessRequest(BaseModel):
    urls: Optional[List[str]] = None
    pdfs: Optional[List[str]] = None
    csvs: Optional[List[str]] = None
    
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

@app.post("/process")
async def process_data(request: ProcessRequest):
    try:
        data = datahandler.process_data(
            urls=request.urls,
            pdfs=request.pdfs,
            csvs=request.csvs,
        )
        return {"status": "success", "message": "Data processed successfully", "data":data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            embedded_question=embedder.embed_query(request.data['question']),
            question=request.data['question'],
            top_k=request.data['top_k'],
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