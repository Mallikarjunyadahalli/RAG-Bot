from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load the PDF
loader = PyPDFLoader(r"C:\Users\kam\OneDrive\Desktop\RAG-project\Mallikarjun_Medical_Report.pdf")
docs = loader.load()
print(f"Loader docs: {len(docs)}")
print("\n\n")
print("\n\n")


# Split the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents=docs)
print(f"Chunks: {len(chunks)}")
print("\n\n")
print("\n\n")


# Embedding and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vecto_db = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model
)

query = "What is the name of Patient ?"
context = vecto_db.similarity_search(query=query, k=4)
print(context)