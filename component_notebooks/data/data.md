## To run the DB backend

1) Build the docker image using the Dockerfile

- `docker build -t milvus-api .`

2) Run the container with the built image

- `docker run -p 9998:9998 --name milvusdb milvus-api`