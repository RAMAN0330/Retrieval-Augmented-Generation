# Import all necessary libraries
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import groq_llm, huggingface_instruct_embedding

st.set_page_config(layout='wide', page_title="Objectbox and Langchain")

st.title('Objectbox VectorstoreDB with LLAMA3')

# Define the chat prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function for vector embedding and ObjectBox VectorstoreDB
def vector_embedding():
    if 'vectors' not in st.session_state:
        # Initialize embeddings, document loader, and text splitter
        st.session_state.embeddings = huggingface_instruct_embedding()
        st.session_state.loader = PyPDFDirectoryLoader('us-census-data')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Split the loaded documents into chunks
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:200])
        # Create the ObjectBox vector store from documents
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768,
            db_directory='objectbox'
        )
        st.write('ObjectBox Database is ready. You can now enter your question')

# Button to trigger embedding
if st.button('Embed Documents'):
    vector_embedding()

# Input field for user question
user_input = st.text_input('Enter your question from documents')

# Check if user input exists and `vectors` is initialized before proceeding
if user_input and 'vectors' in st.session_state:
    # Create the document retrieval chain
    document_chain = create_stuff_documents_chain(groq_llm(), prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time taken for the response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_input})
    st.write(response['answer'])
    st.write(f'Response time: {(time.process_time() - start):.2f} secs')

    # Display the document similarity search results
    with st.expander("Document Similarity Search"):
        # Iterate through and display relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
else:
    if not user_input:
        st.write('Please enter a question to retrieve relevant documents.')
    elif 'vectors' not in st.session_state:
        st.write('Click on "Embed Documents" to initialize the ObjectBox Database.')
