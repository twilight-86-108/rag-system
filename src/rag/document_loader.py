from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class DocumentLoderFactory:
    """
    ファイル形式に応じたLoaderを生成する
    """

    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
    }

    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> List[Document]:
        """
        Argments:
            file_path: ファイルパス

        Returns:
            List[Document]: LangChain Document形式のリスト
        
        Raises:
            ValueError: サポートされていないファイル形式
            FileNotFoundError: ファイルが存在しない
        """
        # 1. ファイルの存在確認

        # 2. 拡張子取得

        # 3. Loader選択

        # 4. ドキュメント読み込み

        # 5. メタデータ付加(ファイル名、サイズ、読み込み日時)

    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """サポート形式チェック"""
        pass


class DocumentProcessor:
    """
    複数ファイルの一括処理・前処理を担当
    - バッチ処理
    - 文字コード自動検出
    - ファイルサイズ制限
    """

    @staticmethod
    def load_from_directory(dir_path: Union[str, Path]) -> List[Document]:
        """ディレクトリ内の全サポートファイルの読み込み"""
        pass

       
