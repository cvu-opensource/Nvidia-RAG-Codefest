## To run the DB backend

1) Build the docker image using the Dockerfile

- `docker build -t data-api .`

2) Run the container with the built image

- `docker run --rm -p 9998:9998 --name data-api data-api`