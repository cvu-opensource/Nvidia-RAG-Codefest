FROM python:3.9-slim
RUN pip install --no-cache-dir streamlit requests 
WORKDIR /app
COPY app.py /app/
EXPOSE 9999
CMD ["streamlit", "run", "app.py", "--server.port=9999", "--server.address=0.0.0.0"]