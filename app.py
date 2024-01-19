import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI
import translation
import mail
from secret_key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key


def main():
    st.set_page_config(page_title="Ask your PDF", page_icon=":books:")
    
    st.header("Query with PDF :books:")
    
    #upload the file
    pdf = st.file_uploader("Upload your PDF here", type="pdf")
    
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
        
        
        # create embeddings i.e., vector representations of the chunks
        embeddings = OpenAIEmbeddings()
        knowledge_base = Chroma.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            st.write(response)
            
            want_translation = st.selectbox("Do you want to translate to any language?", ("Yes", "No"), index=None, placeholder="Choose an option")
            
            if want_translation == "Yes":
                st.header("Text Translator")
                from_language = "English"
                to_language = st.selectbox("Translate to", ("English", "French", "German", "Spanish", "Tamil", "Telugu", "Kannada", "Hindi"))
                
                if from_language == to_language:
                    st.warning("Please select different languages for translation.")
                    st.stop()

                if from_language and to_language and text:
                    translated = translation.generate_translation(to_language, response)
                    translated_response = translated['translatedText'].strip()
                    st.write(translated_response)
            
            want_email = st.selectbox("Do you want a copy of this to be emailed?", ("Yes", "No"), index=None, placeholder="Choose an option")
            
            if want_email == "Yes":
                email_input = st.text_input('Enter Email Address').strip()
                if email_input:
                    if want_translation == "Yes":
                        mail.send_email(email=email_input ,subject="PDF Doc Searcher Response", body=translated_response)
                    else:
                        mail.send_email(email=email_input ,subject="PDF Doc Searcher Response", body=response)

 
if __name__ == '__main__':
    main()