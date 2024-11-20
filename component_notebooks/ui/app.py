import streamlit as st
import requests
import os

DATA_API = "http://localhost:9998"

if "nim_api_endpoint" not in st.session_state:
    st.session_state["nim_api_endpoint"] = os.getenv("NIM_API_ENDPOINT", "")
if "nim_api_key" not in st.session_state:
    st.session_state["nim_api_key"] = os.getenv("NIM_API_KEY", "")

st.set_page_config(page_title="SG Corporate Buddy", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["User Assistant", "Admin Panel"])

if page == "User Assistant":
    st.title("Your SG Corporate Buddy")
    st.sidebar.title("Instructions")
    st.sidebar.write("""
    - Enter your query in the text box.
    - Click **Submit** to get an answer.
    - Provide feedback using the radio buttons.
    - Feedback is optional but helps improve the system.
    - Enter more queries.
    """)

    st.subheader("Need help? Ask away!")
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
    user_query = st.text_area("Type your query here:", height=100)

    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Processing..."):
                files = {"image": uploaded_image.getvalue()} if uploaded_image else None
                data = {"question": user_query, "top_k":5}
                
                response = requests.post(
                    DATA_API + "\search",  # TODO: TO INFERENCE PIPELINE
                    data=data
                ).json()
                
                st.subheader("Response:")
                st.write(response.get("answer", "No answer received."))

                st.subheader("Rate the response:")
                feedback = st.radio("How relevant was this response?", ["Very Relevant", "Somewhat Relevant", "Not Relevant"])
                
                if st.button("Submit Feedback"):
                    feedback_response = requests.post(
                        DATA_API + "\feedback",  # TODO: TO INGESTION PIPELINE
                        json={"query": user_query, "response": response.get("answer", ""), "feedback": feedback},
                    )
                    if feedback_response.status_code == 200:
                        st.success("Arigatou for your feedback!!")
                    else:
                        st.error("L bro. Something's dead.")
        else:
            st.warning("Your query is empty you troller!")

elif page == "Admin Panel":
    st.title("Admin Panel")
    st.subheader("Feed new data:")

    uploaded_files = st.file_uploader("Upload files (PDFs, docs, etc.):", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    images = st.file_uploader("Upload images (png, jpg, jpeg):", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    link = st.text_input("Upload links:")

    if st.button("Ingest Files"):
        if not uploaded_files and not link.strip() and not images: st.error("Nothing to upload.")
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Ingesting {file.name}..."):
                    response = requests.post(
                        DATA_API + "/insert",  # TODO: TO INGESTION PIPELINE
                        files={"file": file.getvalue()},
                        data={"filename": file.name},
                    )
                    if response.status_code == 200:
                        st.success(f"Successfully ingested {file.name}")
                    else:
                        st.error(f"Failed to ingest {file.name}")

        if link.strip():
            with st.spinner("Ingesting link..."):
                response = requests.post(
                    DATA_API + "/insert",  # TODO: TO INGESTION PIPELINE
                    json={"link": link},
                )
                if response.status_code == 200:
                    st.success("Successfully ingested the link.")
                else:
                    st.error("Failed to ingest the link.")

        if images:
            for image in images:
                with st.spinner(f"Ingesting {image.name}..."):
                    response = requests.post(
                        DATA_API + "/insert",  # TODO: TO INGESTION PIPELINE
                        files={"image": image.getvalue()},
                        data={"filename": image.name},
                    )
                    if response.status_code == 200:
                        st.success(f"Successfully ingested {image.name}")
                    else:
                        st.error(f"Failed to ingest {image.name}")

    st.subheader("Configure APIs:")

    st.text_input(
        "Current API Endpoint:",
        value=st.session_state["nim_api_endpoint"],
        key="nim_api_endpoint_input",
        placeholder="Enter NIM API endpoint",
    )
    st.text_input(
        "API Key:",
        value=st.session_state["nim_api_key"],
        key="nim_api_key_input",
        placeholder="Enter NIM API key",
        type="password",
    )
    
    if st.button("Save API Configuration"):
        st.session_state["nim_api_endpoint"] = st.session_state["nim_api_endpoint_input"]
        st.session_state["nim_api_key"] = st.session_state["nim_api_key_input"]
        st.success("API details updated successfully.")