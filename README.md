# Financial report Chat App


## Introduction
------------
Financial report Chat App is a Python application that allows you to chat with financial reports. You can ask questions about the company financial status using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded Financial reports.

This service includes the backoffice feature where operator uploads the Financial report PDF files to perform text embeding and save in the vector store, and the client chat service for testing. 

This service uses ChatOpenAI LLM, therefore OpenAI API key is required. 

The text embedding model is the open source HuggingFaceInstructEmbeddings.

## How It Works
------------

![Financial report Chat App Diagram](./docs/financial-report-chat-diagram.jpg) 

The application follows these steps to provide responses to your questions:

1. financial report HTML Loading: The app reads financial report documents (confluence web page) and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the financial reports.

## Dependencies and Installation
----------------------------
To install the financial report Chat App, please follow these steps:
 
=======

1. Clone the repository to your local machine.

2. Activate Python virtual environment for OpenAI where you have installed, for example
  
  On Windows:
  ```
  # Create new virtual environment at the same location as the root project folder
    Windows:
        python -m venv venv

    Mac/Linux
        python3 -m venv venv

  # Activate on Windows
    .\venv\Scripts\activate

  # Or activate on Linux/Mac
    source venv/bin/activate
  
  ```

Create a .env file at the root level and include this:
Obtain an API key from OpenAI and add it to the `.env` file in the project directory.

financial_report_FAISS_INDEX_DIR="../financial_report_faiss_index"
financial_report_FAISS_INDEX_BAAI_MODEL_DIR="../financial_report_faiss_BAAI_index" 
OPENAI_API_KEY=<your key here>
GFACEHUB_API_TOKEN=<your key here> 

Your openAI API key can be found on https://platform.openai.com/api-keys. 

3. Install the required dependencies by running the following command:
   ```
   $ pip install -r requirements2.txt
   ```

5. To use PineCone vector database, we need to import the pinecone api key.
   https://docs.pinecone.io/guides/get-started/quickstart
   ```
   export PINECONE_API_KEY=your_pinecone_api_key
   ```
   
## Usage
-----
To use the financial report Chat App, follow these steps:

6. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

7. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   
   For the back office app, run with backoffice url
   ```
   $ streamlit run app.py --server.baseUrlPath=/backoffice --server.port 8501
   ```

8. The application will launch in your default web browser, displaying the user interface.

9. Load multiple financial report documents into the app by following the provided instructions.

10. Ask questions in natural language about the loaded financial reports using the chat interface.

11. To run with the fast and lean embedding model that has good quality but very fast, the BAAI small embedding:

   ```
   $ streamlit run financial report_search_BAAI_small.py
   ```


   # To clean vector store first. The separate -- is to indicate to streamlit that the options after that are for  # the python program
   ```
   $ streamlit run financial report_search_BAAI_small.py --server.baseUrlPath=/backoffice --server.port 8501 -- --clean
   ```


   NOTE:This script uses a separate folder for the vector store
   financial_report_FAISS_INDEX_BAAI_MODEL_DIR="../financial_report_faiss_BAAI_index" 
  



