# RAG-PL_test

The LLM now runs on AMD gpu (if compatible with directml).

I recommend creating a .venv (virtual environment) before installing requirements and running the code
#
download_dataset.py is for retrieving .txt files and saving them to documents

create_knowledge_base is for embedding the txt files in documents and saving them in qdrant

rag_pipeline.py is the file that activates the LLM and gives it the info it needs, it also activates the LLM and asks for a query input.
