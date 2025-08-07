import os
import pandas as pd
from utils.file_handler import load_and_split_data
from utils.embedder import get_embedding_model
from utils.retriever import create_vector_db
from utils.chatbot import create_qa_chain

def main():
    file_path = input("📂 Enter path to your Excel or CSV file: ")
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    print("✅ Loading and preparing data...")
    text_data = load_and_split_data(file_path)
    embedding_model = get_embedding_model()
    vector_db = create_vector_db(text_data, embedding_model)
    qa_chain = create_qa_chain(vector_db)

    print("\n🤖 Chatbot is ready! Type 'exit' to quit.\n")
    while True:
        question = input("Ask: ")
        if question.lower() in ['exit', 'quit']:
            break
        answer = qa_chain.run(question)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
