from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

def create_qa_chain(retriever):
    hf_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    local_llm = HuggingFacePipeline(pipeline=hf_pipe)
    return RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever, chain_type="stuff")
