# Hybrid Search with Reranking and Filtering

A comprehensive implementation of advanced hybrid search combining multiple embedding approaches for superior text retrieval performance.

## Overview

This notebook demonstrates a production-ready hybrid search system that combines three embedding techniques:

- **Dense Embeddings** - Semantic search using neural networks for conceptual similarity
- **Sparse Embeddings** - Keyword-based search via BM25 for exact term matching  
- **Late Interaction Embeddings** - Token-level precision using ColBERT for reranking

The system uses Qdrant vector database for scalable storage and retrieval, with multi-tenant filtering capabilities.

## Features

- **Multi-Modal Search**: Combines semantic and keyword-based retrieval
- **Advanced Reranking**: Uses ColBERT for fine-grained relevance scoring
- **Multi-Tenant Security**: User-based filtering with indexed tenant fields
- **GPU Acceleration**: Optimized batch processing with CUDA support
- **Production Ready**: Scalable architecture with Qdrant Cloud integration

## Dataset

Uses 1 million arXiv paper abstracts from the [Hugging Face dataset](https://huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary) for demonstration.

## Requirements

```bash
pip install qdrant-client fastembed fastembed-gpu tqdm polars torch rerankers
```

## Setup

1. **Qdrant Cloud**: Create a cluster and obtain endpoint/API key
2. **Environment**: Configure CUDA for GPU acceleration (optional)
3. **Credentials**: Set `QDRANT_ENDPOINT` and `QDRANT_API_KEY` environment variables

## Architecture

```
Query → [Dense + Sparse Embeddings] → Hybrid Search → ColBERT Reranking → Results
```

### Workflow

1. **Embedding Generation**: Create dense and sparse vectors for documents
2. **Index Creation**: Store multi-vector points in Qdrant with user metadata
3. **Hybrid Retrieval**: Combine dense and sparse search for candidate documents
4. **Filtering**: Apply user-based security filters
5. **Reranking**: Use ColBERT for final relevance scoring

## Performance

- **Dense Embeddings**: ~20 seconds per 1000 docs (CPU)
- **Sparse Embeddings**: ~1 second per 1000 docs (CPU)
- **GPU Acceleration**: Significantly faster with CUDA support

## Key Components

- **FastEmbed**: Lightweight embedding generation
- **Qdrant**: High-performance vector database
- **ColBERT**: Token-level reranking model
- **Binary Quantization**: Memory-efficient storage

## Usage Example

```python
# Query with user filtering
query = "What are the most interesting galaxies in the universe?"
target_user_id = "user_3"

# Hybrid search with reranking returns top 50 most relevant results
```

## Future Enhancements

- Query-time filtering optimization for better performance
- Embedding caching layer for frequently accessed content
- LLM integration for RAG capabilities for better user experience
- Parallel processing improvements for better performance

## References

- [Qdrant Hybrid Search Documentation](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
- [FastEmbed Library](https://github.com/qdrant/fastembed)
- [ColBERT Reranking](https://github.com/stanford-futuredata/ColBERT)

---

*Last updated: June 1, 2025*