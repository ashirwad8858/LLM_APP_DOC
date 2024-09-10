import streamlit as st
import openai
from PyPDF2 import PdfReader

# Configure OpenAI API key
openai.api_key = 'open-ai-key'

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_answer(query, document_text):
    # Generate an answer using OpenAI API with the newer ChatCompletion method
    messages = [
        {"role": "system", "content": "You are an assistant that answers questions based on the provided document."},
        {"role": "user", "content": f"Document: {document_text}"},
        {"role": "user", "content": f"Question: {query}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can also use 'gpt-4' if you have access
        messages=messages,
        max_tokens=200,
        temperature=0.2,
    )
    return response['choices'][0]['message']['content'].strip()

def main():
    st.title("Document-based Question Answering with OpenAI")

    # Step 1: Upload document
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        # Step 2: Extract text from uploaded document
        document_text = extract_text_from_pdf(uploaded_file)
        st.write("Document uploaded successfully! You can now ask questions based on the document.")

        # Step 3: Provide input for the user to ask a question
        user_question = st.text_input("Ask a question based on the document")

        # Step 4: Generate and display answer
        if st.button("Get Answer"):
            if user_question:
                answer = generate_answer(user_question, document_text)
                st.write("Answer:", answer)
            else:
                st.write("Please ask a question!")

if __name__ == "__main__":
    main()
