# GRACE-IRFusion

An enhanced version of GRACE for software vulnerability detection using hybrid retrieval fusion.

---

## Overview

This project improves the original GRACE framework by enhancing the retrieval stage.

Instead of relying on a single similarity strategy, this version integrates multiple retrieval signals:

- Semantic similarity (CodeT5 embeddings)
- Lexical similarity (BM25)
- Structural similarity (AST-based tokens)
- Reciprocal Rank Fusion (RRF)
- Optional embedding whitening

The objective is to improve in-context example selection and increase vulnerability detection accuracy.

---

## Motivation

The original GRACE framework relies primarily on semantic similarity for retrieval.

However, semantic-only retrieval may:

- Miss lexical overlap patterns
- Ignore structural similarity in code
- Fail on rare vulnerability patterns

This project introduces hybrid retrieval fusion to address these limitations.

---

## Key Improvements

- Hybrid retrieval (Semantic + BM25 + AST)
- Reciprocal Rank Fusion (RRF)
- Optional embedding whitening
- Java dataset support (SARD / Juliet)
- Improved cross-dataset generalization

---

## Pipeline

1. Parse source code using Joern  
2. Extract graph-based representations  
3. Generate embeddings using CodeT5  
4. Apply hybrid retrieval fusion  
5. Select top-k in-context examples  
6. Prompt LLM for vulnerability prediction  

---

## Project Structure


.
├── data/ # Dataset files
├── retrieval/ # BM25, RRF, fusion modules
├── embeddings/ # CodeT5 embedding generation
├── scripts/ # Execution scripts
├── genexample.py # Embedding generation
├── llmpre.py # LLM inference
└── README.md


---

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- Joern
- scikit-learn
- rank-bm25

Install dependencies:

```bash
pip install -r requirements.txt