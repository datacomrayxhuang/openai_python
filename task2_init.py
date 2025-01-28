import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

chroma_client = chromadb.PersistentClient(settings = chromadb.Settings(allow_reset=True))
chroma_client.reset()
embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
file_paths = ["the-fellowship-of-the-ring.pdf", "google-terms-of-service.pdf"]

for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunked_documents = text_splitter.split_documents(document)

    Chroma.from_documents(
        collection_name = "task2",
        persist_directory = "./chroma",
        documents = chunked_documents,
        embedding = embedding_function,
        client = chroma_client,
    )

    print(f"Added {len(chunked_documents)} chunks to chroma db")