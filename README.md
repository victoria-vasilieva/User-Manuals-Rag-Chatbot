text
# User Manuals RAG Assistant

A production-style **RAG-based chatbot** that answers questions from technical user manuals (PDFs), built as both an AI engineering system and an evaluation framework.  
The system uses **Roborock robot vacuum manuals** as a realistic dataset to test how Retrieval-Augmented Generation performs on long, structured, technical documentation.  
This project goes beyond "it works" and focuses on **measuring**, **understanding**, and **systematically improving** RAG systems.

# Try Chatbot – Online Demo

[![Run on Streamlit](https://static.streamlit.io/badges/streamlit-badge-primary.svg)](https://share.streamlit.io/yourusername/user-manuals-rag-chatbot/main/app.py)

Click the badge to launch the chatbot in your browser without installing anything locally.

## Features

- **Chatbot**: PDF parsing → embedding retrieval → LLM generation → Streamlit UI
- **Retrieval**: Cross-encoder reranking + HyDE query expansion
- **Evaluation**: RAGAS metrics (Faithfulness, Recall, Precision, Relevance) on a separate evaluation vector store

## Architecture

User Query → [HyDE] → Embedding Retriever → Reranker → LLM → Answer
                               
                               
RAGAS Evaluation

## Quick Start Locally

_# Copy .env.example to .env and add your API key_
pip install -r requirements.txt
_# Add PDFs to data/manuals/_
python ingest.py
streamlit run app.py
python evaluation/evaluation_engine.py

text

## Tech Skills Demonstrated

| Area            | Skills                                      |
|-----------------|---------------------------------------------|
| **AI Engineering** | RAG pipelines, reranking, prompt engineering, LLM orchestration |
| **Data/Analytics** | Controlled experiments, RAGAS evaluation, metric optimization |
| **Software**     | Modular config-driven design, reproducible pipelines |

## Why This Project?

Answers key RAG engineering questions:
- Does reranking consistently improve answer quality?
- How sensitive is retrieval to chunk size?
- When does HyDE help—and when does it fail?
- How reliable are standard RAG metrics vs actual answer quality?

## Results & Experiments

### Experiments Conducted

| Experiment | Description |
|------------|-------------|
| Chunking | Multiple chunk sizes tested |
| Retriever Depth | Varying top-k values |
| Baseline | Dense retrieval without reranking or HyDE |
| Reranking | Cross-encoder applied to improve passage ordering |
| HyDE | Query replaced with synthetic document embedding |


### Key Findings
- **Faithfulness** was the main focus—traditional metrics sometimes mislead
- **Reranking** consistently improved faithfulness
- **HyDE** didn't always boost faithfulness
- **Chunk size/retriever depth** strongly impacted faithfulness

### Example Results

| Configuration | Faithfulness |
|---------------|--------------|
| Baseline      | 0.87         |
| + Reranker    | 0.95         | 
| + HyDE        | 0.95         |

**Shows ability to**: Design experiments, interpret metrics critically, optimize pipelines, ensure non-hallucinated outputs.

## Future Work
- Automated experiment tracking
- Embedding model comparison
- Error clustering and analysis
- Regression testing for RAG pipelines
- Context usage visualization

## Disclaimer
Educational/portfolio use only. Roborock manuals used for experimentation, not redistributed.
