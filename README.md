# Hybrid Search with Reranking and Filtering

Advanced hybrid search combining **dense**, **sparse**, and **late interaction** embeddings with Qdrant vector database for superior search quality and multi-tenant security.

## Features

- **Multi-Modal Search**: Dense (semantic) + Sparse (keyword) + Late interaction (precision)
- **Hybrid Retrieval**: Prefetch with dense/sparse, rerank with late interaction
- **Multi-Tenant**: Secure user filtering with indexed isolation
- **Production-Ready**: Scalable Qdrant architecture with batch processing

## Architecture

```
Query → [Dense + Sparse Prefetch] → [Late Interaction Rerank] → [User Filter] → Results
```

## Quick Start

### Installation
```bash
pip install qdrant-client fastembed tqdm polars
```

## Embedding Models

| Type | Model | Purpose | Output |
|------|-------|---------|--------|
| **Dense** | all-MiniLM-L6-v2 | Semantic similarity | 384D vectors |
| **Sparse** | Qdrant/bm25 | Keyword matching | Variable sparse |
| **Late Interaction** | colbertv2.0 | Precise reranking | Multi-vector |

## Performance

| Operation | Time (1K docs) |
|-----------|----------------|
| Dense embeddings | ~20 seconds |
| Sparse embeddings | ~1 second |
| Late interaction | ~3-4 minutes |

## Dataset

Uses ArXiv paper abstracts from [Hugging Face](https://huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary) for scientific text retrieval demonstration.

## Use Cases

- Enterprise document search
- E-commerce product discovery
- Scientific literature retrieval
- Customer support knowledge bases

## References

- [Qdrant Hybrid Search](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
- [FastEmbed Library](https://github.com/qdrant/fastembed)
- [ColBERT Paper](https://arxiv.org/abs/2004.12832)