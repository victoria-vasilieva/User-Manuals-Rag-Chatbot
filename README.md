<p align="left">
  <!-- App link -->
  <a href="https://victoria-vasilieva-user-manuals-rag-chatbot-app-qy7sxe.streamlit.app/">
    <img src="https://img.shields.io/badge/Run_on-Streamlit-FF4B4B?logo=streamlit&style=for-the-badge" alt="Run on Streamlit">
  </a>
  <!-- Core tech -->
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LLM-Llama_Index-00BFFF?style=for-the-badge" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/Eval-ragas-7E57C2?style=for-the-badge" alt="ragas">
</p>


# User Manuals RAG Assistant

https://github.com/user-attachments/assets/b147dd4c-a5ca-4d0b-b407-1ef7e089160d

A production-style **RAG-based chatbot** that answers questions from technical user manuals (PDFs), built as both an AI engineering system and an evaluation framework.  

The system uses **Roborock robot vacuum manuals** as a realistic dataset to test how Retrieval-Augmented Generation performs on long, structured, technical documentation.  

This project goes beyond "it works" and focuses on **measuring**, **understanding**, and **systematically improving** RAG systems.

# Try Chatbot – Online Demo

<p align="left">
  <a href="https://victoria-vasilieva-user-manuals-rag-chatbot-app-qy7sxe.streamlit.app/">
    <img src="https://img.shields.io/badge/Run%20on-Streamlit-FF4B4B?logo=streamlit" alt="Run on Streamlit" width="220">
  </a>
</p>

<p><sub><em>Click the badge to launch the chatbot in your browser without installing anything locally.</em></sub></p>

## Features

- **Chatbot**: PDF parsing → embedding retrieval → LLM generation → Streamlit UI
- **Retrieval**: Cross-encoder reranking + HyDE query expansion
- **Evaluation**: RAGAS metrics (Faithfulness, Recall, Precision, Relevance) on a separate evaluation vector store

## Architecture

<img width="777" height="218" alt="Bildschirmfoto 2025-12-08 um 11 48 44" src="https://github.com/user-attachments/assets/7243133a-4a68-4b7f-a212-e66dabf58302" />


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

| Experiment | Description | Results |
|------------|-------------|-------------|
| Chunking | Multiple chunk sizes tested | strongly improved faithfulness |
| Retriever Depth | Varying top-k values | strongly improved faithfulness |
| Reranking | Cross-encoder applied to improve passage ordering | consistently improved faithfulness |
| HyDE | Query replaced with synthetic document embedding | didn't impacted faithfulness |


### Key Findings
- **Faithfulness** was the main focus — traditional metrics sometimes mislead


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


## Quick Start Locally

_# Copy .env.example to .env and add your API key_

pip install -r requirements.txt

_# Add PDFs to data/manuals/_

python main.py

streamlit run app.py

python evaluate.py

## Disclaimer
Educational/portfolio use only. Roborock manuals used for experimentation, not redistributed.
