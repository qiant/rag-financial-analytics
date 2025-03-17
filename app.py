import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import shutil
import pickle
from datetime import datetime

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer


def initialize_session_state():
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "current_files" not in st.session_state:
        st.session_state.current_files = set()
    if "rerun_requested" not in st.session_state:
        st.session_state.rerun_requested = False

def display_file_status(uploaded_files):
    if uploaded_files:
        st.markdown("""
        <div class="file-status-header">
            Status Legend: 
            <span class="status-indicator status-processed"></span>Processed 
            <span class="status-indicator status-unprocessed"></span>Unprocessed
        </div>
        """, unsafe_allow_html=True)
        
        current_file_names = {file.name for file in uploaded_files}
        st.session_state.current_files = current_file_names
        st.session_state.processed_files = st.session_state.processed_files.intersection(current_file_names)
        
        for file in uploaded_files:
            if file.name in st.session_state.processed_files:
                st.markdown(
                    f'<div class="processed-file">✓ {file.name}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="unprocessed-file">○ {file.name}</div>',
                    unsafe_allow_html=True
                )

def update_processed_files(processed_files):
    st.session_state.processed_files.update(processed_files)
    st.session_state.rerun_requested = True

def create_backup(index_path):
    if os.path.exists(index_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{index_path}_backup_{timestamp}"
        shutil.copytree(index_path, backup_path, dirs_exist_ok=True)
        print(f"Backup created at: {backup_path}")
        return backup_path
    return None

def restore_from_backup(backup_path, index_path):
    if os.path.exists(backup_path):
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        shutil.copytree(backup_path, index_path)
        print(f"Index restored from backup: {backup_path}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        print("process file: ", pdf.name)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_existing_vectorstore(index_db_name):
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        try:
            return FAISS.load_local(index_db_name, embeddings)
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            backup_path = create_backup(index_db_name)
            try:
                if os.path.exists(index_db_name):
                    docstore_path = os.path.join(index_db_name, "docstore.pkl")
                    
                    with open(docstore_path, 'rb') as f:
                        docstore_data = pickle.load(f)
                    
                    texts = []
                    for doc_id in docstore_data[1]:
                        doc = docstore_data[0].get(doc_id)
                        if doc and hasattr(doc, 'page_content'):
                            texts.append(doc.page_content)
                    
                    if texts:
                        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
                        if os.path.exists(index_db_name):
                            shutil.rmtree(index_db_name)
                        vectorstore.save_local(index_db_name)
                        return vectorstore
                    else:
                        raise ValueError("No valid texts found in the existing index")
                else:
                    raise FileNotFoundError(f"Index directory not found: {index_db_name}")
            except Exception as rebuild_error:
                print(f"Error rebuilding index: {str(rebuild_error)}")
                if backup_path:
                    restore_from_backup(backup_path, index_db_name)
                raise rebuild_error
            finally:
                if backup_path and os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        raise e

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    print("in get conversation chain: ")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print(conversation_chain)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"""
                <div class="chat-message user">
                    <img class="avatar" src="https://api.dicebear.com/7.x/avataaars/svg?seed=user" />
                    <div class="message">{message.content}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot">
                    <img class="avatar" src="https://api.dicebear.com/7.x/bottts/svg?seed=bot" />
                    <div class="message">{message.content}</div>
                </div>
            """, unsafe_allow_html=True)

def process_documents(pdf_docs, index_db_path):
    if not pdf_docs:
        return None
    
    try:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        new_vectorstore = get_vectorstore(text_chunks)
        
        if os.path.exists(index_db_path):
            try:
                exist_vectorstore = get_existing_vectorstore(index_db_path)
                new_vectorstore.merge_from(exist_vectorstore)
                print("Successfully merged with existing index")
            except Exception as merge_error:
                st.warning(f"Starting fresh with new index due to: {str(merge_error)}")
        
        new_vectorstore.save_local(index_db_path)
        
        processed_file_names = {doc.name for doc in pdf_docs}
        update_processed_files(processed_file_names)
        
        return new_vectorstore
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None

def main():
    load_dotenv()
    index_db_path = os.environ.get("Financial_report_FAISS_INDEX_DIR")
    print("vector store index saved in db: ", index_db_path)

    st.set_page_config(page_title="Chat with Financial Reports",
                      page_icon="::")
    
    initialize_session_state()
    with open('./custom.css') as f:
        custom_css = f.read()
    st.markdown(f'{custom_css}', unsafe_allow_html=True)

    st.header("Chat with Financial Reports ::")
    user_question = st.text_input("Ask a question about company or stock:")
    
    if user_question:
        try:
            if st.session_state.conversation is None:
                with st.spinner("Loading knowledge base..."):
                    vectorstore = get_existing_vectorstore(index_db_path)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            handle_userinput(user_question)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    with st.sidebar:
        st.subheader("Financial Report Chat")
        pdf_docs = st.file_uploader(
            "Upload Financial Reports links here and click on 'Process'", accept_multiple_files=True)
        
        display_file_status(pdf_docs)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload PDF documents first.")
                return
            
            with st.spinner("Processing documents..."):
                new_vectorstore = process_documents(pdf_docs, index_db_path)
                if new_vectorstore:
                    st.session_state.conversation = get_conversation_chain(new_vectorstore)
                    st.success("Documents processed and indexed successfully!")
                    if st.session_state.rerun_requested:
                        st.session_state.rerun_requested = False
                        st.experimental_rerun()

if __name__ == '__main__':
    main()