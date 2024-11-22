# Goal is to build a simple, interactive and customisable user interface

## Streamlit

Instead of copying online repositories, we will be using **Streamlit**;

Streamlit is an open-source Python library that makes it easy to create and share **custom web apps for machine learning and data science**. It can be used to quickly build and deploy powerful data applications.

It acts as a simple HTML-JS application, needing to send requests to a running backend server like Tower!

## To run the UI

1) Build the docker image using the Dockerfile

- `docker build -t streamlit-ui .`

2) Run the container with the built image

- `docker run -p 9999:9999 --rm streamlit-ui`

3) Open localhost:8501