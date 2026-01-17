from typing import List, Union, Type
from pathlib import Path
from datetime import datetime
import logging

# LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoderFactory:
    """
    ファイル形式に応じたLoaderを生成する
    """

    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
    }

    @classmethod
    def load_documents(cls, file_path: Union[str, Path]) -> List[Document]:
        """
        Argments:
            file_path: ファイルパス

        Returns:
            List[Document]: LangChain Document形式のリスト
            - page_content: テキスト内容
            - metadata: {
                'source': ファイルパス,
                'file_type': ファイル形式,
                'file_size': ファイルサイズ,
                'loaded_at': ISO形式タイムスタンプ,
                'page': ページ番号
            }
        
        Raises:
            ValueError: サポートされていないファイル形式
            FileNotFoundError: ファイルが存在しない
            Exception: ファイル読み込みエラー
        """
        # 1. Path正規化(str -> Path)
        filepath = Path(file_path)
        # 2. ファイルの存在確認
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        # 3. 拡張子取得
        file_extension = filepath.suffix.lower()
        # 4. サポート形式チェック
        if file_extension not in cls.SUPPORTED_FORMATS:
            supported = ', '.join(cls.SUPPORTED_FORMATS.key())
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {supported}"
            )
        # 5. Loader選択
        loader_class = cls.SUPPORTED_FORMATS[file_extension]

        if file_extension not in cls.SUPPORTED_FORMATS:
             loader = cls._create_text_loader(filepath)
        else:
            loader = loader_class(str(filepath))
        # 6. ドキュメント読み込み
        try:
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
        # 7. メタデータ付加(ファイル名、サイズ、読み込み日時)
        for doc in documents:
            doc.metadata.update({
                'file_type': file_extension,
                'file_size': filepath.stat().st_size,
                'loaded_at': datetime.now().isoformat(),
            })
        # 8. 結果返却
        return documents
        
    @classmethod
    def _create_text_loader(cls, file_path: Path) -> TextLoader:
        """
        TextLoaderをエンコーディング自動検出で作成

        Argments: 
            file_path: テキストファイルのパス
        
        Returns:
            TextLoader: 適切なエンコーディングで初期化
        """
        encodings = ['utf-8', 'shift-jis']
        # エンコーディング確認
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                
                logger.info(f"Detected encoing: {encoding} for {file_path.name}")
                return TextLoader(str(file_path), encoding=encoding)
            
            except UnicodeDecodeError:
                continue
        
        # 失敗したとき
        raise ValueError(
            f"Could not decode {file_path.name} with any supported encoding:"
            f"{', '.join(encodings)}" 
        )

    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """
        サポート形式チェック
        
        Argments: 
            file_path: チェックするファイルのパス
        
        Returns:
            bool: サポートしていればTrue
        """

        file_path = Path(file_path)
        return file_path.suffix.lower() in cls.SUPPORTED_FORMATS


class DocumentProcessor:
    """
    複数ファイルの一括処理・前処理を担当
    - バッチ処理
    - 文字コード自動検出
    - ファイルサイズ制限
    """

    @staticmethod
    def load_from_directory(dir_path: Union[str, Path], recursive: bool = False) -> List[Document]:
        """
        ディレクトリ内の全サポートファイルの読み込み
        
        Arguments: 
            dir_path: 対象ディレクトリのパス
            recursive: サブディレクトリも処理するときTrue

        Returns:
            List[Document]: 全ファイルから読み込んだDocumentのリスト
        
        Raises:
            NotADirectoryError: 指定パスがディレクトリでない
        """
        # ファイルパス
        dir_path = Path(dir_path)
        # パスチェック
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        # ディレクトリのファイルを走査
        all_documents = []

        for file_path in dir_path.iterdir():
            if file_path.is_dir():
                continue

            if DocumentLoderFactory.is_supported(file_path):
                try:
                    docs = DocumentLoderFactory.load_documents(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
                    continue

        # ログ
        logger.info(
            f"Loaded {len(all_documents)} documents from {dir_path}"
            f"({len([f for f in dir_path.iterdir() if f.is_file()])} files)"
        )
        
        return all_documents


def debug_document_info(documents: List[Document]) -> None:
    """
    読み込んだドキュメント情報を表示（デバッグ）

    Arguments:
        documents: 表示するDocumentのリスト
    """
    print(f"Loaded {len(documents)} documents\n")

    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:")
        print(f"length: {len(doc.page_content)} chars")
        print(f"preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    # PDF
    try:
        pdf_docs = DocumentLoderFactory.load_documents("data/sample_company_info.pdf")
        debug_document_info(pdf_docs)
    except Exception as e:
        print(f"pdf failed: {e}\n")
    # txt
    try:
        txt_docs = DocumentLoderFactory.load_documents("data/sample_tech_doc.txt")
        debug_document_info(txt_docs)
    except Exception as e:
        print(f"txt failed: {e}\n")
    # ディレクトリの一括
    try:
        all_docs = DocumentProcessor.load_from_directory("data")
        print(f"{len(all_docs)} loaded")
    except Exception as e:
        print(f"directory failed: {e}\n")
