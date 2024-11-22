## To run the VLM

1) Build the docker image using the Dockerfile

- `docker build -t qwen-vlm .`

2) Run the container with the built image

- `docker run -p 8003:8003 -v /local/g05/qwen_cp:/qwen_cp qwen-vlm`