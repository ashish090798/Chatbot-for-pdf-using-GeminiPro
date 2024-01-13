import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

st.title("Q&A Chatbot for FAQs")
st.text("(Upload your PDF -- Ask a question)")
target_dir = "pdfs"
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
                    # Set a really small chunk size, just to show.
                    chunk_size = 1024,
                    chunk_overlap  = 128,            
                )

        chunks = text_splitter.create_documents([text])

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        location = "pdfs"

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

        def create_chain(prompt):
            model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain

        chain = create_chain(prompt)

        text = st.text_input('Enter your question here ')
        if text:
            st.write("Response : ")
            with st.spinner("Searching for answers ..... "):
                response = chain(
                    {"input_documents":chunks, "question": text},
                    return_only_outputs=True)
                st.write(response['output_text'])
            st.write("")

    except Exception as e:
        st.write(e)
    



