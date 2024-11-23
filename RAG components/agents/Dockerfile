FROM python:3.9-slim-buster

# hastily copied all pip installs from all relevant dockerfiles. Some probably not used but just for safety.

RUN pip install faiss-gpu # replace with faiss-gpu if you are using GPU
RUN pip install faiss-cpu
RUN pip install requests pdfplumber spacy camelot-py 
RUN pip install pandas==2 numpy==1.26.4 
RUN pip install beautifulsoup4 
RUN pip install pymupdf
RUN pip install lxml
RUN pip install unstructured
RUN pip install langchain langchain_community langchain-nvidia-ai-endpoints langchain-experimental neo4j tiktoken yfiles_jupyter_graphs python-dotenv json-repair langchain-openai langchain_core

COPY multi-agent-requirements.txt /tmp/multi-agent-requirements.txt

RUN pip install -r /tmp/multi-agent-requirements.txt 
RUN pip install langgraph
RUN pip install langchain-nvidia-ai-endpoints==0.3.5
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install uvicorn fastapi

EXPOSE 9997
WORKDIR /agents

CMD ["uvicorn", "lj-multi-agent:app", "--host", "0.0.0.0", "--port", "9997", "--reload"]