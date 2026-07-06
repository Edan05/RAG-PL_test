# RAG-PL_test
the pipeline uses qwen3-4b-instruct-2507 as the LLM, qwen3-0.6b as router and BAAI/bge-m3 as the encoder.

the pipeline also understands if the question is a general topic one, or if it needs to retrieve documents to answer.
#
The LLM now checks available devices (Cuda, DirectML, CPU).

I recommend creating a .venv (virtual environment) before installing requirements and running the code

You need to install the onnx versions of the generator and router, being careful to install either the directml or cuda version of them depending on the machine's gpu distributor (CUDA for Nvidia, dml for AMD)
#
download_dataset.py is for retrieving .txt files and saving them to documents

create_knowledge_base is for embedding the txt files in documents and saving them in Qdrant

pipeline_with_router.py is the pipeline itself.
