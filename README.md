# Multi-PDF Chatbot

This project is a comprehensive Python application that combines text extraction, document analysis, language translation, and email automation. The system is designed to facilitate efficient handling of PDF documents, enabling users to upload files, extract text using Chroma DB, pose targeted questions, and receive precise language translations along with email capabilities therefore streamlining communication.

## Features

- Multi-PDF option for users to handle and analyze multiple PDF documents
- PDF text extraction and vector representations via Chroma DB
- Targeted question-based chatbot for document information
- Language translation capabilities for multilingual document understanding
- Automated email process using GMail API


## Technologies Used

- **Frontend:** Streamlit
- **Backend:** Python
- **Vector Databse:** [Chroma DB](https://link-to-chroma-db-repo)
- **PDF Handling:** [PyPDF2](https://github.com/mstamy2/PyPDF2)
- **AI Libraries:**
  - [OpenAI](https://github.com/openai/llms): `langchain.llms`
  - [Langchain](https://github.com/langchain/langchain): 
    - `langchain.prompts`
    - `langchain.chains`
    - `CharacterTextSplitter`
    - `OpenAIEmbeddings`
  - **Authentication and Authorization:**
    - [Google Auth](https://github.com/googleapis/google-auth-library-python): `google-auth` 
    - [Google Auth OAuthLib](https://github.com/googleapis/google-auth-library-oauthlib-python): `google-auth-oauthlib` 
    - [Google Auth HTTPLib2](https://github.com/googleapis/google-auth-library-python-httplib2): `google-auth-httplib2` 
- **Google API Integration:** [Google API Python Client](https://github.com/googleapis/google-api-python-client): `google-api-python-client`


## Usage

### Prerequisites

Before running the application, ensure you have the necessary API keys for OpenAI and GMail.

### Getting Started

1. Clone this repository to your local machine.
2. Create a `secret_key.py` file in the project root directory.
3. Add your OpenAI API key to the `secret_key.py` file:
   
   ```python
   # secret_key.py
   openapi_key="YOUR_API_KEY_HERE"
   
4. Download the json file from GCP and name the file as `credentials.json`. Save this file in the project root directory.
5. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   
7. Run the application using `streamlit run app.py`.

## How it Works

Upon uploading multiple PDFs, the application initiates a systematic data processing workflow. The uploaded data is meticulously segmented into manageable chunks, and each segment undergoes embedding for storage in the Chroma vector database. Users are prompted to inquire about the content through targeted questions. Leveraging OpenAI's Language Model (LLM), the system conducts a semantic search, providing users with insightful answers. For enhanced accessibility, users can opt to translate the results into another language using OpenAI's language translation capabilities. The final touch involves seamless email communication facilitated by the GMail API, allowing users to effortlessly share or archive the analyzed and translated content. This streamlined process caters to the needs of professionals, researchers, and students, offering a comprehensive solution for efficient document handling, information retrieval, and multilingual communication.
