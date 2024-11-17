import streamlit as st
import requests

st.set_page_config(page_title="SG Corporate Buddy", layout="centered")

st.sidebar.title("Instructions")
st.sidebar.write("""
- Enter your query in the text box.
- Click **Submit** to get an answer.
- Provide feedback using the radio buttons.
- Feedback is optional but helps improve the system.
- Enter more queries.                 
""")

st.title("Your SG Corporate Buddy")

st.subheader("Need help? Ask away!")
uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
user_query = st.text_area("Type your query here:", height=100)

if st.button("Submit"):
    if user_query.strip():
        with st.spinner("Processing..."):
            
            files = {"image": uploaded_image.getvalue()} if uploaded_image else None
            data = {"query": user_query}
            
            # Send request to the backend
            response = requests.post(
                "http://localhost:5000/inference",
                data=data,
                files=files,
            ).json()
            
            # Display Response
            st.subheader("Response:")
            st.write(response.get("answer", "No answer received."))
            
            # Feedback Section
            st.subheader("Rate the response:")
            feedback = st.radio("How relevant was this response?", ["Very Relevant", "Somewhat Relevant", "Not Relevant"])
            
            if st.button("Submit Feedback"):
                feedback_response = requests.post(
                    "http://localhost:5000/feedback",
                    json={"query": user_query, "response": response.get("answer", ""), "feedback": feedback},
                )
                if feedback_response.status_code == 200:
                    st.success("Arigatou for your feedback!!")
                else:
                    st.error("L bro. Something's dead.")
    else:
        st.warning("Your query is empty you troller!")
