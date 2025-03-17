import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import shutil
import pickle
from datetime import datetime
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI


@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def initialize_session_state():
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "current_files" not in st.session_state:
        st.session_state.current_files = set()
    if "stored_documents" not in st.session_state:
        st.session_state.stored_documents = {}

def get_stored_documents(index_db_path):
    try:
        docstore_path = os.path.join(index_db_path, "index.pkl")
        print(f"DEBUG: Attempting to read docstore from: {docstore_path}")
        
        if os.path.exists(docstore_path):
            print(f"DEBUG: Found docstore file")
            with open(docstore_path, 'rb') as f:
                docstore_data = pickle.load(f)
                
                if not isinstance(docstore_data, tuple) or len(docstore_data) != 2:
                    print(f"DEBUG: Invalid docstore format")
                    return {}
                
                stored_docs = {}
                inmemory_docstore = docstore_data[0]
                doc_ids = docstore_data[1]
                docstore_dict = inmemory_docstore._dict
                
                for doc_id in doc_ids.values():
                    doc = docstore_dict.get(doc_id)
                    if doc and hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        doc_name = doc.metadata['source']
                        if doc_name not in stored_docs:
                            stored_docs[doc_name] = []
                        stored_docs[doc_name].append(doc_id)
                
                return stored_docs
        else:
            print(f"DEBUG: Docstore file not found at {docstore_path}")
            return {}
    except Exception as e:
        print(f"DEBUG: Error reading stored documents: {str(e)}")
        return {}

def display_stored_documents(index_db_path):
    docs = get_stored_documents(index_db_path)
    print("Current stored documents:", docs)
    
    if docs:
        st.markdown('### Currently Indexed Documents')
        
        for doc_name in docs.keys():
            with st.container():
                cols = st.columns([6, 1])
                with cols[0]:
                    st.markdown(f'<div class="processed-file">✓ Successfully uploaded: <span class="success-text">{doc_name}</span></div>', unsafe_allow_html=True)
                with cols[1]:
                    if st.button("DEL", key=f"remove_{doc_name}", type="primary"):
                        with st.spinner(f"Removing {doc_name}..."):
                            if remove_documents(index_db_path, [doc_name]):
                                st.success(f"Successfully removed {doc_name}")
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                st.error(f"Failed to remove {doc_name}")

def remove_documents(index_db_path, docs_to_remove):
    print(f"Attempting to remove documents: {docs_to_remove}")
    try:
        embeddings = get_embeddings_model()
        vectorstore = FAISS.load_local(index_db_path, embeddings)
        
        backup_path = f"{index_db_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.path.exists(index_db_path):
            shutil.copytree(index_db_path, backup_path)
            print(f"Created backup at: {backup_path}")
        
        try:
            docstore_path = os.path.join(index_db_path, "index.pkl")
            with open(docstore_path, 'rb') as f:
                docstore_data = pickle.load(f)
            
            remaining_chunks = []
            removed_ids = []
            docstore_dict = docstore_data[0]._dict
            doc_ids = docstore_data[1]
            
            id_to_index = {doc_id: idx for idx, doc_id in doc_ids.items()}
            
            for doc_id in doc_ids.values():
                doc = docstore_dict.get(doc_id)
                if doc and hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    if doc.metadata['source'] not in docs_to_remove:
                        remaining_chunks.append(doc)
                    else:
                        removed_ids.append(id_to_index[doc_id])
                        print(f"Removing chunk from {doc.metadata['source']}")
            
            print(f"Found {len(removed_ids)} chunks to remove")
            
            if not remaining_chunks:
                shutil.rmtree(index_db_path)
                st.session_state.stored_documents = {}
                print("Removed all documents from store")
                st.cache_resource.clear()
                if 'conversation' in st.session_state:
                    del st.session_state['conversation']
                return True
            
            new_vectorstore = FAISS.from_documents(remaining_chunks, embeddings)
            
            if os.path.exists(index_db_path):
                shutil.rmtree(index_db_path)
            new_vectorstore.save_local(index_db_path)
            print("Saved updated vector store")
            
            st.cache_resource.clear()
            if 'conversation' in st.session_state:
                del st.session_state['conversation']
            print("Cleared cache after document removal")
            
            st.session_state.stored_documents = get_stored_documents(index_db_path)
            return True
            
        except Exception as e:
            print(f"Error during document removal: {str(e)}")
            if os.path.exists(backup_path):
                if os.path.exists(index_db_path):
                    shutil.rmtree(index_db_path)
                shutil.copytree(backup_path, index_db_path)
                print("Restored from backup after error")
            return False
            
    except Exception as e:
        print(f"Error accessing vector store: {str(e)}")
        return False

def get_text_chunks(text, source_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source_name}]
    )
    return chunks

def process_documents(pdf_docs, index_db_path):
    if not pdf_docs:
        return None
    
    try:
        print("Starting document processing...")
        os.makedirs(index_db_path, exist_ok=True)
        
        all_chunks = []
        progress_bar = st.empty()
        
        for idx, pdf in enumerate(pdf_docs):
            try:
                print(f"Processing file {idx+1}/{len(pdf_docs)}: {pdf.name}")
                progress_text = f"Processing {pdf.name} ({idx + 1}/{len(pdf_docs)})"
                file_progress = progress_bar.progress(0, text=progress_text)
                
                pdf_reader = PdfReader(pdf)
                print(f"PDF loaded with {len(pdf_reader.pages)} pages")
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + "\n"
                    print(f"Extracted text from page {page_num+1}")
                    file_progress.progress(
                        (page_num + 1) / len(pdf_reader.pages),
                        text=f"{progress_text} - Page {page_num + 1}/{len(pdf_reader.pages)}"
                    )
                
                print(f"Creating chunks for {pdf.name}")
                chunks = get_text_chunks(text, pdf.name)
                print(f"Created {len(chunks)} chunks with source: {pdf.name}")
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing file {pdf.name}: {str(e)}")
                st.error(f"Error processing {pdf.name}: {str(e)}")
                continue
        
        print(f"Total chunks created: {len(all_chunks)}")
        with st.spinner('Creating embeddings...'):
            print("Initializing embeddings model")
            embeddings = get_embeddings_model()
            print("Creating new vectorstore")
            new_vectorstore = FAISS.from_documents(all_chunks, embeddings)
            
            if os.path.exists(index_db_path):
                try:
                    print("Loading existing vectorstore for merging")
                    exist_vectorstore = FAISS.load_local(index_db_path, embeddings)
                    print("Merging with existing vectorstore")
                    new_vectorstore.merge_from(exist_vectorstore)
                    print("Successfully merged with existing index")
                except Exception as merge_error:
                    print(f"Merge error: {str(merge_error)}")
                    st.warning(f"Starting fresh with new index due to: {str(merge_error)}")
            
            print(f"Saving vectorstore to {index_db_path}")
            new_vectorstore.save_local(index_db_path)
            
            processed_file_names = {doc.name for doc in pdf_docs}
            st.session_state.processed_files.update(processed_file_names)
            st.session_state.stored_documents = get_stored_documents(index_db_path)
            
            # Clear the file uploader after successful processing
            #st.session_state['file_uploader'] = None
            # removed due to the error msg: st.session_state.file_uploader cannot be modified after the widget with key file_uploader is instantiated.
            
            print("Document processing completed successfully")
            return new_vectorstore
            
    except Exception as e:
        print(f"Critical error in process_documents: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return None
    
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1500
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        max_token_limit=2000
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar">👤</div>
                    <div class="message">{message.content}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">🤖</div>
                    <div class="message">{message.content}</div>
                </div>
            """, unsafe_allow_html=True)


def main():
   st.set_page_config(
       page_title="Chat with financial reports",
       page_icon="::",
       layout="wide"
   )

   load_dotenv()
   index_db_path = os.environ.get("FINANCIAL_REPORT_FAISS_INDEX_BAAI_REMOVE_FILES_DIR")
   if not index_db_path:
       st.error("FINANCAIL_REPORT_FAISS_INDEX_BAAI_REMOVE_FILES_DIR environment variable not set")
       return

   if not os.path.exists(index_db_path):
       st.cache_resource.clear()
       if 'conversation' in st.session_state:
           del st.session_state['conversation']
       print("Cleared cache due to missing index directory")

   print(f"Using index path: {index_db_path}")
   
   initialize_session_state()

   with open('./custom_baai.css') as f:
       custom_css = f.read()
   st.markdown(f'{custom_css}', unsafe_allow_html=True)

   if os.path.exists(index_db_path):
       print(f"Found existing index directory at: {index_db_path}")
       existing_docs = get_stored_documents(index_db_path)
       if existing_docs:
           st.session_state.stored_documents = existing_docs
           print(f"Loaded {len(existing_docs)} documents into session state")
       else:
           print("No documents found in existing index")
   else:
       print(f"No existing index directory at: {index_db_path}")

   st.header("Chat with financial reports ::")
   
   with st.sidebar:
       st.subheader("Document Management")
       
       display_stored_documents(index_db_path)
       
       st.markdown("---")
       
       pdf_docs = st.file_uploader(
           "Upload financial reports here and click on 'Process'", 
           accept_multiple_files=True,
           key='file_uploader'
       )

       if pdf_docs and not any(doc.name in st.session_state.processed_files for doc in pdf_docs):
           st.markdown("Files ready to process:")
           for doc in pdf_docs:
               if doc.name not in st.session_state.processed_files:
                   st.markdown(f'<div class="upload-status">⚪ Waiting to process: {doc.name}</div>', unsafe_allow_html=True)

       if st.button("Process Documents"):
           if not pdf_docs:
               st.warning("Please upload PDF documents first.")
               return
           
           new_vectorstore = process_documents(pdf_docs, index_db_path)
           if new_vectorstore:
               st.session_state.conversation = get_conversation_chain(new_vectorstore)
               st.success("Documents processed and indexed successfully!")
               st.experimental_rerun()
   
   user_question = st.text_input("Ask a question about company or stock:")
   
   if user_question:
       try:
           if st.session_state.conversation is None:
               print("Initializing new conversation")
               with st.spinner("Loading knowledge base..."):
                   vectorstore = FAISS.load_local(index_db_path, get_embeddings_model())
                   st.session_state.conversation = get_conversation_chain(vectorstore)
           handle_userinput(user_question)
       except Exception as e:
           print(f"Error in conversation handling: {str(e)}")
           st.error(f"Error processing question: {str(e)}")

if __name__ == '__main__':
   main()