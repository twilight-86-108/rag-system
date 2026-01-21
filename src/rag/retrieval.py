from typing import List, Dict, Any, Optional
import logging

# LangChain
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreManager
from rag.llm_integration import LLMManager, create_rag_prompt, format_context_from_docs

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverConfig:
    """
    Retrieverの設定を管理

    Attributes:
        search_type: 検索タイプ（'similarity', 'mmr'）
        k: 取得する文書数
        score_threshold: スコア閾値
    """

    def __init__(
        self,
        search_type: str = "similarity",
        k: int = 3,
        score_threshold: Optional[float] = None,
    ):
        """
        Arguments:
            search_type: 'similarity' or 'mmr'
            k: 取得文書数
            score_threshold: スコア閾値（0.0~1.0, 低いほど類似）
        """
        self.search_type = search_type
        self.k = k
        self.score_threshold = score_threshold

        logger.info(
            f"RetrieverConfig initialized: type={search_type},"
            f"k={k}, threshold={score_threshold}"
        )


class RAGPipeline:
    """
    RAGの完全なパイプライン

    Components:
    - Vector Store (ChromaDB)
    - Retriever (VectorStoreRetriever)
    - Prompt Template (RAG用)
    - LLM (ChatOllama + Llama 3.1)
    """

    def __init__(
        self,
        vectorstore,
        llm_manager: LLMManager,
        retriever_config: Optional[RetrieverConfig] = None,
    ):
        """
        Arguments:
            vectorstore: Chroma Vector Store
            llm_manager: LLMManager instance
            retriever_config: Retriever設定
        """
        self.vectorstore = vectorstore
        self.llm_manager = llm_manager
        self.retriever_config = retriever_config or RetrieverConfig()

        self.retriever = self._create_retriever()
        self.prompt = create_rag_prompt()
        self.llm = llm_manager.get_llm()
        self.chain = self._create_rag_chain()

        logger.info("RAGPipeline initialized successfully.")

    def _create_retriever(self):
        """
        Vector StoreからRetrieverを作成

        Returns:
            VectorStoreRetriever: 設定済みRetriever
        """
        search_kwargs = {"k": self.retriever_config.k}

        if self.retriever_config.score_threshold is not None:
            search_kwargs["score_threshold"] = self.retriever_config.score_threshold

        retriever = self.vectorstore.as_retriever(
            search_type=self.retriever_config.search_type,
            search_kwargs=search_kwargs,
        )

        logger.info(
            f"Created retriever: type={self.retriever_config.search_type},"
            f"k={self.retriever_config.k}"
        )

        return retriever

    def _create_rag_chain(self):
        """
        RAGチェーンを作成

        Returns:
            Runnable: 実行可能なチェーン
        """

        # ドキュメントフォーマット
        def format_docs(docs):
            return format_context_from_docs(docs)

        # RAGチェーン構築
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        # ログ
        logger.info("RAG chain created")

        return rag_chain

    def _retrieve_documents(self, question: str) -> List[Document]:
        """
        Retriever から関連ドキュメントを取得（LangChainのAPI差異を吸収）

        - 古いAPI: get_relevant_documents(query)
        - 新しいAPI: invoke(query) / ainvoke(query)
        """
        # LangChain のバージョン差異吸収
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(question)  # type: ignore[attr-defined]

        # langchain-core の Runnable としての retriever
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(question)
            # 念のため型を正規化（異常系は空にする）
            return docs if isinstance(docs, list) else []

        raise AttributeError(
            "Retriever does not support 'get_relevant_documents' or 'invoke'."
        )

    def query(self, question: str) -> str:
        """
        質問に対してRAG応答を生成

        Arguments:
            question: ユーザーの質問

        Returns:
            str: LLMからの応答
        """
        logger.info(f"Processing query: '{question}'")

        try:
            response = self.chain.invoke(question)

            logger.info(f"Response generated: {len(response)} chars")

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """
        質問に対して応答とソース文書を返す

        Arguments:
            question: ユーザーの質問

        Returns:
            dict: {
                'answer': 応答文字列,
                'sources': List[Document]
            }
        """
        logger.info(f"Processing query with sources: '{question}'")

        # 文書取得
        source_docs = self._retrieve_documents(question)
        # 回答生成
        answer = self.query(question)
        # 結果
        result = {
            "answer": answer,
            "sources": source_docs,
            "num_sources": len(source_docs),
        }
        # ログ
        logger.info(f"Query completed with {len(source_docs)} sources")
        # return
        return result

    @classmethod
    def from_existing_vectorstore(
        cls,
        collection_name: Optional[str] = None,
        retriever_config: Optional[RetrieverConfig] = None,
    ):
        """
        既存のVector StoreからRAGPipelineを作成

        Arguments:
            collection_name: 読み込むコレクション名
            retriever_config: Retriever設定

        Returns:
            RAGPipeline: 初期化済みパイプライン
        """
        logger.info("Creating RAGPipeline from existing vector store...")

        vectorstore_manager = VectorStoreManager()
        vectorstore = vectorstore_manager.load_vectorstore(collection_name)

        llm_manager = LLMManager()

        pipeline = cls(
            vectorstore=vectorstore,
            llm_manager=llm_manager,
            retriever_config=retriever_config,
        )

        logger.info("RAGPipeline creted successfully")

        return pipeline

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        collection_name: Optional[str] = None,
        retriever_config: Optional[RetrieverConfig] = None,
    ):
        """
        ドキュメントから新規にRAGPipelineを作成

        Arguments:
            documents: チャンク済みDocumentリスト
            collection_name: 新規コレクション名
            retriever_config: Retriever設定

        Returns:
            RAGPipeline: 初期化済みパイプライン
        """
        logger.info(f"Creating RAGPipeline from {len(documents)} documents")
        vectorstore_manager = VectorStoreManager()
        vectorstore = vectorstore_manager.create_vectorstore(
            documents=documents, collection_name=collection_name
        )

        llm_manager = LLMManager()

        pipeline = cls(
            vectorstore=vectorstore,
            llm_manager=llm_manager,
            retriever_config=retriever_config,
        )

        logger.info("RAGPipeline created from documents successfully")

        return pipeline


if __name__ == "__main__":
    # 既存Vector StoreからRAGPipeline作成
    print("Test 1")
    try:
        pipeline = RAGPipeline.from_existing_vectorstore()
        print("RAG Pipeline created successfuly")

    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # 簡単なやり取り（日本語）
    print("Test 2")
    try:
        question_jp = "RAGシステムとは何？簡潔に説明して。"
        print(f"\nQuestion: {question_jp}")

        answer = pipeline.query(question_jp)
        print("\nAnswer:")
        print(f"{answer}\n")

    except Exception as e:
        print(f"\nTest 2 failed: {e}")
        import traceback

        traceback.print_exc()

    # 英語で質問
    print("Test 3")
    try:
        question_en = "What are the security requirements for AI system?"
        print(f"\nQuestion: {question_en}")

        answer = pipeline.query(question_en)

        print("\nAnswer:")
        print(f"{answer}\n")

    except Exception as e:
        print(f"Test 3 failed: {e}\n")
        import traceback

        traceback.print_exc()

    # ソース付き回答
    print("Test 4")

    try:
        question = "Vector Databaseについて教えて"
        print(f"\nQuestion: {question}")

        result = pipeline.query_with_sources(question)

        print("\nAnswer:")
        print(result["answer"])

        print(f"\nSources ({result['num_sources']} documents):")
        for i, doc in enumerate(result["sources"], 1):
            print(f"\nSource {i}:")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Metadata: {doc.metadata}")

        print()
    except Exception as e:
        print(f"Test 4 failed: {e}\n")
        import traceback

        traceback.print_exc()

    print("All Testing Completed")
