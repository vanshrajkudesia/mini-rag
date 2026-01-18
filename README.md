# mini-rag
A simple end-to-end RAG system built using FastAPI, Hugging Face models, Pinecone vector database, and Cohere reranker. The application allows users to upload text, ask questions, and receive answers grounded in retrieved context with visible citations.

chunking Parameters chunk size = 800 overlap = 80

Vector Database Provide: Pinecone Index Dimension : 384

Top-k retrieval k = 10 for matching cosine similarity is used

Reranking Provider : Cohere Top-N retrieval after reranking = 5

LLM Provider : Hugging Face (HF) Model: google/flan-t5-small

User Interface Built using HTML inside FastAPI

title: Mini Rag App sdk: gradio sdk_version: 6.3.0 app_file: app.py

Remark: Initially, OpenAI models were used as the LLM for answer generation. However, due to free-tier credit exhaustion and API rate limits, OpenAI models were discontinued. The system was migrated to a free Hugging Face LLM (google/flan-t5-base). Tradeoff observed: Reduction in answer fluency and coherence Occasional shorter or less precise responses
<img width="1168" height="477" alt="arch" src="https://github.com/user-attachments/assets/29e67546-7b23-4adb-b4a0-7b8b97a5a82f" />
