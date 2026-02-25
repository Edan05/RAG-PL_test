# RAG-PL_test

The LLM runs on CPU, tried to run on AMD card using DirectML to no avail
#
I recommend creating a .venv (virtual environment) before installing requirements and running the code
#
download_dataset.py is for retrieving .txt files and saving them to documents (to simulate a database)

create_knowledge_base is for embedding the txt files in documents and saving them in knowledge_base.pkl and document_index.faiss

rag_pipeline.py is the file that activates the LLM and gives it the info it needs, it also activates the LLM and asks for a query input.
