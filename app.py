import streamlit as st
import doc_finder


st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Query PDF Source")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(doc_finder.get_llm_response(form_input))


# run using "streamlit run app.py"