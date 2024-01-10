import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
#from langchain.callbacks import get_openai_callback
from secret_key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key


def main():
    st.set_page_config(page_title="Ask your PDF", page_icon=":books:")
    
    st.header("Query with PDF :books:")
    
    #upload the file
    pdf = st.file_uploader("Upload your PDFs here and click on 'Process'", type="pdf")
    
    # extract the file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)
        chunks = text_splitter.split_text(text)
        
        #st.write(chunks)
        
        # create embeddings i.e., vector representations of the chunks
        embeddings = OpenAIEmbeddings()
        knowledge_base = Chroma.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            #with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            #    print(cb)
            
            st.write(response)
 
if __name__ == '__main__':
    main()