from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# Load the PDF
loader = PyPDFLoader(r"C:\Users\kam\OneDrive\Desktop\RAG-project\Mallikarjun_Medical_Report.pdf")
docs = loader.load()
print(f"Loader docs: {len(docs)}")


# Split the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents=docs)
print(f"Chunks: {len(chunks)}")


# Embedding and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vecto_db = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model
)

query = "What is the name of Doctor and his age. ?"
documents = vecto_db.similarity_search(query=query, k=4)

# context 
context = ""

for doc in documents:
    context += doc.page_content + "\n\n"

# Prompt
prompt = f""" 
        You are a helpfull assistent and provide answer based on the provides context, Add your own emoji's based on the user query.
        
        context: {context},
        question: {query}
 """


# llm model
llm = ChatGroq(model="openai/gpt-oss-120b")
answer = llm.invoke(prompt)
print(answer.content)