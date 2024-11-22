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
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", base_url="http://10.149.8.40:8001/v1", truncate="END")
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", base_url="http://10.149.8.40:8000/v1")
text_splitter = CharacterTextSplitter(chunk_size=2048, separator=" ", chunk_overlap=64)

# Initialise DataHandler instance 
datahandler = DataHandler(embedder, text_splitter)

# Initialize MilvusDB instance
database = MilvusDB("http://10.149.8.40:19530", llm)  
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
    top_k: Optional[int] = 50

class FeedbackRequest(BaseModel):
    doc_id: str
    feedback: str

@app.post("/process")
async def process_data(request: ProcessRequest):
    json = request.dict()
    try:
        processed_data = datahandler.process_data(json)
        return {"status": "success", "message": "Data processed successfully", "processed_data": processed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert")
async def insert_data(request: InsertRequest):
    json = request.dict()
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
    json = request.dict()
    try:
        results = database.retrieve_data(
            embedded_question=json["embedded_question"],
            question=json["question"],
            top_k=json["top_k"],
        )
        results = database.rechunk_data(
            data=results,
            new_chunk_size=512
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    json = request.dict()
    try:
        database.update_document_scores(query=json['query'], reponse=json['response'], feedback=json['feedback'])  # Finalise how to use user feedback
        return {"status": "success", "message": "Feedback processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_records")
async def get_all_records(limit: int = 1000):
    json = request.dict()
    try:
        records = database.get_all_records(limit=json['limit'])
        return {"status": "success", "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))