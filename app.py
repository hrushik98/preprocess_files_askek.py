import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import boto3
import uuid
import csv
session = boto3.Session(
    aws_access_key_id=st.secrets['aws_access_key_id'],
    aws_secret_access_key=st.secrets['aws_secret_access_key']
)

s3 = session.resource('s3')

bucket_name = st.secrets['bucket_name']
bucket = s3.Bucket(bucket_name)

bucket.download_file('library.csv', 'library.csv')



st.title("Upload embeddings to S3")

api_key = st.text_input('Enter openai api key',type = "password")

file = st.file_uploader("Upload your pdf here", type=["pdf"])
book_name = st.text_input("Enter the name of the book")

if st.button("Upload"):
    book_name = book_name.replace(" ", "_")
    uuid_name = uuid.uuid4()
    with open("library.csv", "a") as f:
        datai = csv.writer(f)
        datai.writerow([book_name, uuid_name])

    if file is not None:
        file_content = file.read()
        with open(f"{book_name}.pdf", "wb") as f:
            f.write(file_content)
        st.success(f"{book_name}.pdf has been saved in the current directory.")
    else:
        st.warning("Please upload a PDF file before clicking the 'Upload' button.")

    loader = PyPDFLoader(f"{book_name}.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    db = FAISS.from_documents(docs, embeddings)
    book_name = uuid_name
    db.save_local(f"{book_name}_index")
    st.write('Book successfully indexed')
    st.write('Uploading to S3...')
    
    for file in os.listdir(f"{book_name}_index"):
        bucket.upload_file(f"{book_name}_index/{file}", f"{book_name}_index/{file}")

    
    st.success('Uploaded to S3')
    bucket.upload_file("library.csv", "library.csv")
    
    
    
    
    st.success('Uploaded to S3')
    
    
    
