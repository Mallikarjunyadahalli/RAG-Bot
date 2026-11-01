from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from time import sleep
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


def docuemnt_process(path: str):

    # Load the PDF
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Split the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents=docs)

    # Embedding and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_db = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

### Uploading document section.
st.set_page_config(
    page_title="Doc Reader",
    page_icon="📃"
)

st.header("📃 Upload document - Ask Anything")
st.caption("Upload your own data than ask anything about document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if not st.session_state.document_uploaded:
    file = st.file_uploader("Upload your document", type=['pdf'])
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing Document..."):
            docuemnt_process("./uploaded_document.pdf")

        st.markdown("✅ Document uploaded successfully...")
        sleep(1)
        st.rerun()


if st.session_state.document_uploaded and st.session_state.vector_db:

    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anything...")
    if query:
        st.session_state.messages.append({"role" : "user", "content" : query})
        st.chat_message("user").markdown(query)
        with st.spinner("Thinking..."):
            documents = st.session_state.vector_db.similarity_search(query, k=3)

            context = ""
            for doc in documents:
                context += doc.page_content + "\n\n"

            prompt = f"""You are a helpful document assistant. Answer the user's question using  ONLY the context extracted from their uploaded document.

                Rules:
                    1. Base your answer strictly on the context below. Do not use outside knowledge to add facts.
                    2. If the answer is not in the context, reply exactly: "I couldn't find that in the uploaded document."
                    3. If the context only partly answers the question, give what is available and say what is missing.
                    4. Be clear and concise. Use bullet points or short steps when the answer has multiple parts.
                    5. Do not mention "the context" or "the chunks" in your answer. Speak as if you read the document ("According to the document...").
                    6. If the question is a greeting or small talk, respond briefly and ask what they'd like to know about the document.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    
            """

            result = llm.invoke(prompt)
            st.session_state.messages.append({"role" : "ai", "content" : result.content})
            st.chat_message("ai").markdown(result.content)