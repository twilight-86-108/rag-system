# RAG System - Design Document

## Architecture Decision Records (ADRs)

### ADR-001: LLM Model Selection

**Context**:
This project's RAG requires a balance between security, compliance, and performance.

**Considered Options**:

1. Qwen3:8b (Alibaba, highest Japanese performance)
2. Gemma2:9b (Google, strong reasoning)
3. Llama3.1:8b (Meta, balanced)

**Decision**: Llama3.1:8b

**Rationale**:

- Security: Llama3.1 is open source and has a permissive license.
- Context Window: 128K (16x larger than Gemma2:9b)
- Performance: MMLU 66.7% (sufficient)
- License: Apache 2.0

**Trade-offs Accepted**:

- Slightly lower reasoning than Gemma2 (68.6% vs 66.7%)
- Accepted for security and context window advantage

### ADR-002: Vector Store Selection

**Decision**: ChromaDB

**Rationale**:

- a

### ADR-003: Dependency Version Strategy

**Decision**: 最新安定版へ変更

**Retional**:

- LangChain 1.0:安定性
- Ollama 0.6.1:最新API対応
- ChromaDB 1.4.0: 安定版、後方互換性あり

## Technical Stack

- LLM: Ollama (llama3.1:8b)
- Vector Store: ChromaDB
- Framework: LangChain
- Embeddings: all-MiniLM-L6-v2
- UI: Streamlit
- Language: Python 3.11.8

## Implementation Plan
