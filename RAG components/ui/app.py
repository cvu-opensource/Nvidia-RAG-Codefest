import os
from PIL import Image
import requests
import streamlit as st

# Initialise endpoints
DATA_API = "http://10.149.8.40:9998"
AGENT_API = "http://10.149.8.40:9997"
VLM_API = "http://10.149.8.40:8003"

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state["history"] = []
    
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
                
#                 if uploaded_image:
#                     vlm_response = requests.post(
#                         VLM_API + "/invoke",
#                         json={'user_query': user_query, 'image':uploaded_image.getvalue()}
#                     ).json()
                    
#                     if vlm_response.status_code != 200:
#                         st.warning(f"VLM API call failed: {vlm_response.status_code}, {vlm_response.text}")
#                 else:
#                     vlm_response = None

                files = {}
                data = {'prompt': user_query}

                if uploaded_image:
                    files['image'] = ('image.jpg', uploaded_image.getvalue(), 'image/jpeg')

                    # Send the POST request to the API
                    vlm_response = requests.post(
                        VLM_API + "/invoke",
                        data=data,
                        files=files  # Send the image as a file
                    ).json()
                    st.success("VLM API call successful")
                else:
                    vlm_response = None
                
                if 'history' in st.session_state:
                    history = '.'.join([f"human's question is {interaction['query']} and model's response is {interaction['response']}" for interaction in st.session_state['history']])
                else:
                    history = ''
                    
                llm_response = requests.post(
                    AGENT_API + "/invoke",
                    json={'query':user_query, 'vlm_context':vlm_response['vlm_response'] if vlm_response else '', 'history':history}
                ).json()
                
                st.subheader("Response:")
                st.write(llm_response['content'])
                
                st.session_state["history"].append({"query": user_query, "response": llm_response['content']}) 

                st.subheader("Rate the response:")
                feedback = st.radio("How relevant was this response?", ["Very Relevant", "Somewhat Relevant", "Not Relevant"])
                
                if st.button("Submit Feedback"):
                    st.success("Arigatou for your feedback!!")  # Temporary till we figure out how to use it 
                    # feedback_response = requests.post(
                    #     DATA_API + "/feedback",  # TODO: TO INGESTION PIPELINE AND GRAPH RAG
                    #     json={"query": user_query, "response": response.get("answer", ""), "feedback": feedback},
                    # ).json()
                    # if feedback_response.status_code == 200:
                    #     st.success("Arigatou for your feedback!!")
                    # else:
                    #     st.error("L bro. Something's dead.")
        else:
            st.warning("Your query is empty you troller!")
            
    # Display conversation history
    if st.session_state["history"]:
        st.subheader("Conversation History")
        for entry in st.session_state["history"]:
            st.write(f"**You:** {entry['query']}")
            st.write(f"**Buddy:** {entry['response']}")

elif page == "Admin Panel":
    st.title("Admin Panel")
    st.subheader("Feed new data:")

    uploaded_files = st.file_uploader("Upload files (PDFs, docs, etc.):", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    images = st.file_uploader("Upload images (png, jpg, jpeg):", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    links = st.text_area("Upload links (one per line):")


    if st.button("Ingest Files"):
        if not uploaded_files and not link.strip() and not images: 
            st.error("Nothing to upload.")
        else:
            with st.spinner("Ingesting files..."):
                try:
                    response = requests.post(
                        DATA_API + "/process",  # TODO: TO INGESTION PIPELINE - DATAHANDLER
                        json={
                            "urls": [link.strip() for link in links_input.strip().split("\n") if link.strip()] if links else None,  
                            "pdfs": images if images else None, 
                            "csvs":uploaded_files if uploaded_files else None
                        }
                    ).json()
                    if response.status_code == 200:
                        st.success(f"Successfully ingested documents.")
                    else:
                        st.error(f"Failed to ingest documents.")
                        raise Exception("Error saving files")
                        
                    processed_data = requests.get("processed_data")
                    response = requests.post(
                        DATA_API + "/insert",  # TODO: TO INGESTION PIPELINE - DATABASE
                        json=processed_data
                    ).json()
                    if response.status_code == 200:
                        st.success(f"Successfully ingested documents.")
                    else:
                        st.error(f"Failed to ingest documents.")
                        raise Exception("Error saving files")
                    
                        
                except:
                    st.error(f"Failed to ingest documents.")