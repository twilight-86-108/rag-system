from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

# LangChain
from chromadb.types import VectorEmbeddingRecord
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.vdms import DEFAULT_COLLECTION_NAME
from langchain_core import vectorstores
from langchain_core.documents import Document

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingConfig:
    """
    Embedding生成の設定を管理

    Attributes:
        model_name: HuggingFace上のモデル名
        model_kwargs: モデル初期化のパラメータ
        encode_kwargs: エンコード時のパラメータ
    """
    # デフォルトモデル
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        normalize_embeddings: bool = True
    ):
        """
        Arguments:
            model_name: 使用するEmmbeddingモデル名
            device: 'cpu' or 'cuda'
            normalize_embeddings: ベクトル正規化するか
        """
        self.model_name = model_name
        self.model_kwargs = {'device': device}
        self.encode_kwargs = {'normalize_embeddings': normalize_embeddings}

        logger.info(
            f"EmbeddingConfig initialized: model={model_name},"
            f"device={device}, normalize={normalize_embeddings}"
        )
    

def create_embeddings(config: Optional[EmbeddingConfig] = None) -> HuggingFaceEmbeddings:
    """
    HuggingFaceEmbeddingsインスタンスを作成
    
    Arguments:
        config: Embedding設定
    Returns:
        HuggingFaceEmbeddings: 設定済みEmbeddingモデル
    """
    # configで設定されてなければデフォルト設定を使用
    if config is None:
        config = EmbeddingConfig()
    # HuggingFaceEmbeddingsをインスタンス化
    embeddings = HuggingFaceEmbeddings(
        model_name=config.model_name,
        model_kwargs=config.model_kwargs,
        encode_kwargs=config.encode_kwargs,
    )
    # ロギング出力
    logger.info(f"Created HuggingFaceEmbeddings with model: {config.model_name}")

    return embeddings


class VectorStoreConfig:
    """
    ChromaDB Vector Storeの設定を変更

    Attributes:
        collection_name: ChromaDBのコレクション名
        persist_directory: 永続化ディレクトリパス
        distance_metric: 距離計算方法 
    """
    DEFAULT_COLLECTION = "rag_docs"
    DEFAULT_PERSIST_DIR = "./chroma_db"

    def __init__(
        self, 
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        distance_metric: str = "cosine"
    ):
        """
        Arguments:
            collection_name: コレクション名（DB内のテーブル相当）
            persist_directory: 永続化先ディレクトリ
            distance_metric: 'cosine', 'l2', 'ip'
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        # ディレクトリ作成
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # ロギング
        logger.info(
            f"VectorStoreConfig initialized: collection={collection_name},"
            f"persidt_dir={persist_directory}, metric=distance_metric"
        )

    
class VectorStoreManager:
    """
    ChromaDB Vector Storeの作成・管理を担当
    """
    def __init__(
        self, 
        embedding_config: Optional[EmbeddingConfig] = None,
        vectorstore_config: Optional[VectorStoreConfig] = None
    ):
        """
        Arguments:
            embedding_config: Embedding設定
            vectorstore_config: VectorStore設定
        """
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vectorstore_config = vectorstore_config or VectorStoreConfig()
        self.embeddings = create_embeddings(self.embedding_config)

        logger.info("VectorStoreManager initialized")
    
    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None
    ) -> Chroma:
        """
        ドキュメントからChromaDB Vector Storeを作成

        Arguments:
            documents: 買う脳するDocumentリスト（チャンク済み）
            collection_name: コレクション名
        
        Returns:
            Chroma: 作成されたVector Store
        """
        # documentがないとき
        if not documents:
            logger.watning("No documents to add to vector store")
            return None
        
        # collection_name決定
        collection_name = collection_name or self.vectorstore_config.collection_name

        logger.info(
            f"Creating vector store with {len(documents)} documents"
            f"in collection '{collection_name}'..."
        )
        # Chroma.from_documents()でVector Store作成
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.vectorstore_config.persist_directory,
        )
        # 統計情報ロギング
        logger.info(
            "Vector store created successfully."
            f"Collection: {collection_name},"
            f"Documents: {len(documents)},"
            f"Persist directory: {self.vectorstore_config.persist_directory}"
        )
        # Vector Storeを返却
        return vectorstore

    def load_vectorstore(
        self,
        collecrion_name: Optional[str] = None
    ) -> Chroma:
        """
        既存のVectore Storeを読み込み

        Arguments:
            collection_name: 読み込むコレクション名

        Returns:
            Chroma: 読み込まれたVector Store
        
        Raises:
            ValueError: コレクションが存在しない
        """
        # collection_name決定
        collection_name = collecrion_name or self.vectorstore_config.collection_name

        logger.info(f"Loading existing vector store: {collection_name}")
        # Chromaインスタンス作成
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_config.persist_directory,
            )
            # 存在確認
            count = vectorstore._collection.count()

            logger.info(
                f"Vector dtore loaded. Collection: {collection_name},"
                f"Documents: {count}"
            )
            # Vector Storeを返却
            return vectorstore
        
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise ValueError(
                f"Collection '{collecrion_name}' not found or invalid."
                f"Persist directiry: {self.vectordtore_config.persist_directory}"
            )
    
    def add_documents(
        self, 
        vectorstore: Chroma,
        documents: List[Document]
    ) -> None:
        """
        既存のVector Storeにドキュメントを追加

        Arguments:
            vectorstore: 追加先のVector Store        
            documents: 追加するDocumentリスト
        """
        pass


def test_vectorstore_search(
    vectorstore: Chroma,
    query: str,
    k: int = 3
) -> List[Document]:
    """
    Vector Storeの検索機能をテスト

    Arguments:
        vectorestore: 検索対象のVector Store
        query: 検索クエリ
        k: 取得するドキュメント数

    Returns:
        List[Document]: 類似度が高い上位 k ドキュメント
    """
    logger.info(f"Searching for: '{query}' (k={k})")

    results = vectorstore.similarity_search(query, k=k)

    print(f"Search Result fot: '{query}'")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("Score: (simikarity_search doesn't return scores by default)\n")
    
    return results


if __name__ == "__main__":
    from document_loader import DocumentLoderFactory
    from text_splitter import DocumentChunker

    # Vector Store
    print("Test 1")
    try:
        pdf_docs = DocumentLoderFactory.load_documents("data\sample_company_info.pdf")
        txt_docs = DocumentLoderFactory.load_documents("data\sample_tech_doc.txt")
        all_docs = pdf_docs + txt_docs
        print(f"Loaded {len(all_docs)} documents")

        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(all_docs)
        print(f"Created {{len(chunks)}} chunks")

        vectorstore_manager = VectorStoreManager()
        vectorstore = vectorstore_manager.create_vectorstore(chunks)

        print("Vector Store created successfully!")
    
    except Exception as e:
        print(f"Test 1 failed: {e}\n")
        import traceback
        traceback.print_exc()

    # 検索テスト（日本語）
    print("Test 2")
    try:
        query_jp = "RAGシステムとはなんですか？"
        results_jp = test_vectorstore_search(vectorstore, query_jp, k=3)
        print(f"Japanese search returned {len(results_jp)} results")
    
    except Exception as e:
        print(f"Test 2 failed: {e}\n")
    
    # 検索テスト（英語）
    print("Test 3")
    try:
        query_en = "What is the RAG system?"
        results_en = test_vectorstore_search(vectorstore, query_en, k=3)
        print(f"english search returned {len(results_en) } results")
    
    except Exception as e:
        print(f"Test 3 failed: {e}\n")

    # 永続化確認
    print("Test 4")
    try:
        vectorstore_manager_new = VectorStoreManager()
        vectorstore_loaded = vectorstore_manager_new.load_vectorstore()

        query_test = "Embedding"
        results_loaded = test_vectorstore_search(vectorstore_loaded, query_test, k=2)

        print(f"Loaded {len(results_loaded)} results")
    
    except Exception as e:
        print(f"Test 4 failed: {e}\n")
    
    print("All tests completed.")
