# End to End RAG Project using ObjectBox and LangChain
 - In this incredibly groundbreaking project, I’ve built a RAG (Retrieval-Augmented Generation) app using ObjectBox Vector Database and LangChain. Why, you ask? Well, because RAG lets us make language models actually useful by feeding them real data on the go! With ObjectBox, you can do all that locally—no cloud drama, no “sending your data to mysterious places.” 🚀

## 📝 Description
- This project proudly showcases the super complex implementation of an advanced RAG system. It uses the fancy ObjectBox vector database and Groq’s LLAM3 model as the brain (aka LLM) to extract info from a bunch of PDF documents. You’re welcome.

Steps  followed:
1. I have used the `PyPdfDirectoryLoader` from the `langchain_community` document loader to load the PDF documents from the `us-census-data` directory.
2. Converted the boring text into exciting chunks of  `1000` using the `RecursiveCharacterTextSplitter` imported from the `langchain.text_splitter` (because who doesn’t love tiny pieces of text?).
3. stored the vector embeddings(yes, those) which were made using the `HuggingFaceBgeEmbeddings` using the `ObjectBox` vector store.
4. setup the llm `ChatGroq` with the model name `Llama3-8b-8192` (sounds cool, right?).
5. Setup `ChatPromptTemplate` for when the AI gets tired of you asking the same thing over and over.
6. Setup `vector_embedding` function to enbedd the documents and store them in the `ObjectBox` vectorstore like a hoarder.
7. finally created the `document_chain` and `retrieval_chain` for chaining llm to prompt and `retriever` to `document_chain` respectively to make the LLM chat with the prompt and link retriever to document_chain. Easy peasy. 😎


## 🔧Libraries Used
 - langchain==0.1.20
 - langchain-community==0.0.38
 - langchain-core==0.1.52
 - langchain-groq==0.1.3
 - langchain-objectbox
 - python-dotenv==1.0.1
 - pypdf==4.2.0

## Installation
 1. Prerequisites
    - Git (obviously 🙄)
    - Command line familiarity (you got this!)

 3. Create and Activate Virtual Environment (Recommended)
    - `python -m venv venv`
    - `source venv/bin/activate`
 4. Navigate to the projects directory.
 5. Install Libraries: `pip install -r requirements.txt`
 6. Navigate to the app directory `cd ./app` using your terminal 
 7. run `streamlit run app.py`
 8. open the link displayed in the terminal on your preferred browser 🌐
 9. Don’t worry, I’ve already `embedded` the documents for you. Just sit back and relax! But if for some reason it doesn’t work (who knows), just hit the `Embed Documents` button and go make yourself a coffee while it processes. ☕

There you go—an end-to-end RAG project that’s as fun to read as it is to run! 😏
