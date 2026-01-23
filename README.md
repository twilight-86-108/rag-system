# RAG System

LangChain 1.0+ を用いたRAG（Retrieval-Augmented Generation）システム

---

## 技術スタック

| レイヤー | 技術 | 選定理由 |
|---------|------|---------|
| **LLM** | Llama 3.1 8B | 128Kコンテキスト、Meta製、日本語対応 |
| **LLM Runtime** | Ollama | ローカル実行による機密性確保、API依存排除 |
| **Embeddings** | all-MiniLM-L6-v2 | 軽量（384次元）、日英両対応 |
| **Vector DB** | ChromaDB | ローカル永続化、Python native |
| **Framework** | LangChain 1.0+ | LCEL準拠、拡張性 |
| **UI** | Streamlit | 高速プロトタイピング、Python統合 |

---

## アーキテクチャ

### システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
│                       (app.py)                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    RAG Pipeline                              │
│                   (retrieval.py)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Retriever  │→ │   Prompt    │→ │   LLM (Llama 3.1)   │  │
│  │  (top-k=3)  │  │  Template   │  │   + Streaming       │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
│         │                                                    │
│  ┌──────▼──────┐                    ┌─────────────────────┐  │
│  │ Vector Store │                   │  Message History    │  │
│  │  (ChromaDB)  │                   │  (per session)      │  │
│  └─────────────┘                    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### データフロー

1. **Document Ingestion**: PDF/TXT → Loader → Chunker → Embeddings → ChromaDB
2. **Query Processing**: Question → Retriever → Context + History → LLM → Streaming Response

---

## 主要機能

### 実装済み機能

| 機能 | 説明 | 技術的ハイライト |
|-----|------|-----------------|
| **ドキュメント処理** | PDF/TXT読み込み、チャンク分割 | Factory Pattern、日本語対応セパレータ |
| **ベクトル検索** | セマンティック類似度検索 | ChromaDB永続化、Cosine類似度 |
| **質問応答** | RAGベース回答生成 | LCEL準拠パイプライン |
| **ストリーミング** | リアルタイム応答表示 | `chain.stream()` + `st.write_stream()` |
| **会話履歴** | マルチターン対話 | RunnableWithMessageHistory |
