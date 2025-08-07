from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_db(text_data, embedding_model, persist_path="chroma_store"):
    docs = [Document(page_content=text_data)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vector_db = Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_path)
    return vector_db.as_retriever(search_kwargs={"k": 3})
