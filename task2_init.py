import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

chroma_client = chromadb.PersistentClient(path = "./db/task2/chroma", settings = chromadb.Settings(allow_reset = True))
chroma_client.reset()
embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
file_paths = ["knowledge_base/task2/team_info.txt", "knowledge_base/task2/cv_ray.pdf", "knowledge_base/task2/cv_mohamed.pdf"]
# file_paths = ["the-fellowship-of-the-ring.pdf", "google-terms-of-service.pdf"]

for file_path in file_paths:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        document = loader.load()
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunked_documents = text_splitter.split_documents(document)

    Chroma.from_documents(
        collection_name = "task2",
        persist_directory = "./db/task2/chroma",
        documents = chunked_documents,
        embedding = embedding_function,
        client = chroma_client,
    )
