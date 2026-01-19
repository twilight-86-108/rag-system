from typing import List, Optional
from dataclasses import dataclass
import logging

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextSplitterConfig:
    """
    Text Splitterの設定を管理

    Attributes:
        chunk_size: チャンクの最大文字数
        chunk_overlap: チャンク間のオーバーラップ文字数
        length_function: 長さ計算関数
        separators: 分割に使用する区切り文字リスト
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    length_function: callable = len
    separators: Optional[List[str]] = None

    def __post_init__(self):
        """
        初期化後のバリデーション

        Raises:
            ValueError: 無効な設定値
        """
        # バリデーション
        if self.chunk_size <= 0:
            raise ValueError('chunk_sizeは正の値を指定')

        if self.chunk_overlap < 0:
            raise ValueError('chunk_overlapは非負数を指定')

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlapはchunk_sizeより小さい値を指定')

        # separatorsのデフォルト設定
        if self.separators is None:
            self.separators = [
                '\n\n',
                '\n',
                '。',
                '.',
                '　',
                ' ',
                '',
            ]
        
        logger.info(
            'TextSplitterConfigの初期化:'
            f"chunk_size={self.chunk_size},"
            f"chunk_overlap={self.chunk_overlap},"
            f"separators={len(self.separators)} levels"
        )

def create_text_splitter(
    config: Optional[TextSplitterConfig] = None
) -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitterのインスタンス作成

    Arguments:
        config: TextSplotterの設定（Noneの場合はデフォルト）
    
    Returns:
        RecursiveCharacterTextSplitter: 設定済みのSplitter
    """
    # configがNoneならデフォルト瀬底を使用
    if config is None:
        config = TextSplitterConfig()
    # RecursiveCharacterTextSplitterを設定でインスタンス化
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=config.length_function,
        separators=config.separators,
    )
    # ロギング出力
    logger.info(
        'RecursiveCharacterTextSplitterを作成:'
        f"chunk_size={config.chunk_size}, overlap={config.chunk_overlap}"
    )
    # Splitterをリターン
    return splitter


class DocumentChunker:
    """
    ドキュメントをチャンク化、統計情報を管理
    """
    def __init__(self, config: Optional[TextSplitterConfig] = None):
        """
        Arguments:
            config: TextSplitterの設定（Noneの場合はデフォルト）
        """
        self.config = config or TextSplitterConfig()
        self.splitter = create_text_splitter(self.config)
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0.0,
        }

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントリストをチャンクに分割

        Arguments:
            documents: 分割するDocumentリスト
        
        Returns:
            List[Document]: チャンク化されたDocumentリスト。
                            元のメタデータは保持され、chunk_indexが追加
        """
        # RecursiveCharacterTextSplitter.split_documents()を使用
        if not documents:
            logger.warning("No documents to chunk")
            return []
        # 各チャンクにメタデータ追加
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
        # 統計情報を更新
        self.stats['total_documents'] = len(documents)
        self.stats['total_chunks'] = len(chunks)

        if chunks:
            total_size = sum(len(chunk.page_content) for chunk in chunks)
            self.stats['avg_chunk_size'] = total_size / len(chunks)
        # ロギング出力
        logger.info(
            f"Chunked {len(documents)} documents into {len(chunks)} chunks."
            f"Avg chunks size: {self.stats['avg_chunk_size']:.1f} chars"
        )
        # リターン
        return chunks

    def get_stats(self) -> dict:
        """
        チャンク化統計情報を取得

        Returns:
            dict: 統計情報
        """
        return self.stats.copy()


def analyze_chunk_distribution(chunks: List[Document]) -> dict:
    """
    チャンクのサイズ分布を分析（デバッグ、最適化用）

    Arguments:
        chunks: 分析対象のチャンクリスト

    Returns:
        dict: {
            'total_chunks': チャンク総数,
            'min_size': 最小サイズ,
            'max_size': 最大サイズ,
            'avg_size': 平均サイズ,
            'median_size': 中央値,
        }
    """
    # チャンクがない
    if not chunks:
        return {
            'total_chunks': 0,
            'min_size': 0,
            'max_size': 0,
            'avg_size': 0.0,
            'median_size': 0,
        }
    # チャンクサイズ分布の確認
    sizes = [len(chunk.page_content) for chunk in chunks]
    sizes_sorted = sorted(sizes)

    return {
        'total_chunks': len(chunks),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'avg_size': sum(sizes) / len(sizes),
        'median_size': sizes_sorted[len(sizes) // 2],
    }


def debug_chunks_info(chunks: List[Document], show_content: bool = False) -> None:
    """
    チャンク除法をデバッグ出力

    Arguments:
        chunks: 表示するチャンクリスト
        show_content: 内容のプレビューも表示する場合True
    """
    print(f"Analyze {len(chunks)} chunks")

    stats = analyze_chunk_distribution(chunks)
    print(f"Min: {stats['min_size']} chars")
    print(f"Max: {stats['max_size']} chars")
    print(f"Avg: {stats['avg_size']:.1f} chars")
    print(f"Median: {stats['median_size']} chars\n")

    if show_content:
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"Chunk {i}:\n")
            print(f"Size: {len(chunk.page_content)} chars")
            print(f"Metadata: {chunk.metadata}")
            print(f"Preview: {chunk.page_content[:150]}...\n")


if __name__ == "__main__":
    from document_loader import DocumentLoderFactory

    # デフォルトのチャンク化テスト
    print("Test 1")
    try:
        docs = DocumentLoderFactory.load_documents("data/sample_company_info.pdf")
        print(f"Loaded {len(docs)} documents from PDF\n")

        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(docs)

        print(f"Chunked into {len(chunks)} chunks")
        print(f"Stats: {chunker.get_stats()}")

        debug_chunks_info(chunks, show_content=True)
        print("Test 1 Successed")
    except Exception as e:
        print(f"Test 1 failed: {e}\n")

    # カスタム設定
    print("Test 2")
    try:
        txt_docs = DocumentLoderFactory.load_documents("data\sample_tech_doc.txt")
        print(f"Loaded {len(txt_docs)} documents from txt")

        custom_config = TextSplitterConfig(
            chunk_size=300,
            chunk_overlap=50
        )
        custom_chunker = DocumentChunker(custom_config)

        custom_chunks = custom_chunker.chunk_documents(txt_docs)
        print(f"Chunked into {len(custom_chunks)} chunks")
        print(f"Stats: {custom_chunker.get_stats()}")

        stats = analyze_chunk_distribution(txt_docs)
        print(f"Size range: {stats['min_size']}-{stats['max_size']} chars")
        print("Test 2 Successed")

    except Exception as e:
        print(f"Test 2 failed: {e}\n")

    # エラーハンドリング
    print("Test 3")
    try: 
        invalid_config = TextSplitterConfig(chunk_size=-100)
        print("Should have raised ValueError")
    except Exception as e:
        print("ValueError correctly raised: {e}")
    
    try:
        invalid_config = TextSplitterConfig(chunk_size=100, chunk_overlap=200)
        print("Should have raised ValueError")
    except Exception as e:
        print("ValueError correctly raised: {e}")
