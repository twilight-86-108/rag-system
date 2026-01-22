from numpy import size
import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Optional
import sys

# プロジェクトルートの追加
sys.path.append(str(Path(__file__).parent))
# 作成したコンポーネントをインポート
from rag.document_loader import DocumentLoderFactory
from rag.text_splitter import DocumentChunker, TextSplitterConfig
from rag.vector_store import VectorStoreManager, VectorStoreConfig
from rag.llm_integration import LLMManager
from rag.retrieval import RAGPipeline, RetrieverConfig

st.set_page_config(
    page_title="RAG system",
    page_icon="🇷🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """
    全てのセッション状態変数を初期化
    """
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def render_header():
    """
    アプリケーションヘッダー
    """
    st.title("RAG system")
    st.markdown(
        """
    **Document Question & Answer System**
    
    ドキュメントをアップロード、システムを設定して独自の知識ベースに基づいた
    AI回答を生成。
    """
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.uploaded_files:
            st.metric(
                "読み込み済みドキュメント",
                len(st.session_state.uploaded_files),
                delta="準備完了",
            )
        else:
            st.metric("読み込み済みドキュメント", "0", delta="アップロード必要")

    with col2:
        if st.session_state.vectorstore:
            st.metric("vectorstore", "稼働中", delta="準備完了")
        else:
            st.metric("vectorstore", "未作成", delta="処理が必要")

    with col3:
        if st.session_state.rag_pipeline:
            st.metric("RAGパイプライン", "準備完了", delta="稼働中")
        else:
            st.metric("RAGパイプライン", "未準備", delta="処理が必要")

        st.divider()


def render_sidebar():
    """
    設定オプション付きサイドバー
    """
    st.sidebar.title("設定")

    st.sidebar.header("ドキュメントアップロード")
    uploaded_files = st.sidebar.file_uploader(
        "ドキュメントをアップロード(PDFまたはTXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="知識ベースを構築するために1つ以上のドキュメントをアップロードしてください",
    )

    st.sidebar.header("処理パラメータ")

    chunk_size = st.sidebar.slider(
        "チャンクサイズ",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="処理時のチャンクサイズ",
    )

    chunk_overlap = st.sidebar.slider(
        "チャンクオーバーラップ",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="連続するチャンク間のオーバーラップ",
    )

    st.sidebar.header("検索パラメータ")

    k = st.sidebar.slider(
        "検索結果数（k）",
        min_value=1,
        max_value=10,
        value=3,
        help="検索する関連チャンクの数",
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="LLMの創造性設定(0=固定、1=変動的)",
    )

    process_button = st.sidebar.button(
        "ドキュメントを管理",
        disabled=(uploaded_files is None or len(uploaded_files) == 0),
        use_container_width=True,
        type="primary",
    )

    with st.sidebar.expander("デバッグ情報"):
        st.json(
            {
                "アップロード済みファイル": (
                    len(uploaded_files) if uploaded_files else 0
                ),
                "チャンクサイズ": chunk_size,
                "チャンクオーバーラップ": chunk_overlap,
                "k": k,
                "temperature": temperature,
                "ベクトルストア稼働": st.session_state.vectorstore is not None,
                "RAGパイプライン稼働": st.session_state.rag_pipeline is not None,
            }
        )

    return uploaded_files, chunk_size, chunk_overlap, k, temperature, process_button


def process_uploaded_files(
    uploaded_files: List, chunk_size: int, chunk_overlap: int
) -> Optional[VectorStoreManager]:
    """
    アップロードされたファイルを処理し、ベクターストアを作成する

    Arguments:
        uploaded_files: Streamlitからアップロードされたファイルオブジェクトの一覧
        chunk_size: チャンクサイズ
        chunk_overlap: チャンクの重複

    Returns:
        VectorStoreManagerのインスタンスかNone(処理が失敗した場合)
    """
    try:
        all_documents = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"{uploaded_file.name}を読み込み中...")

                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                documents = DocumentLoderFactory.load_documents(str(file_path))
                all_documents.extend(documents)

                progress_bar.progress((i + 1) / (len(uploaded_files) * 3))

        if not all_documents:
            st.error("ドキュメントの読み込みに失敗しました。")
            return None

        st.success(f"{len(all_documents)}件のドキュメントを読み込みました")

        status_text.text("ドキュメントをチャンクに分割中...")
        config = TextSplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunker = DocumentChunker(config=config)
        split_docs = chunker.chunk_documents(all_documents)
        progress_bar.progress(2 / 3)
        st.success(f"{len(split_docs)}個のテキストチャンクを作成しました。")

        status_text.text("Vector Storeを作成中...")
        vectorstore_manager = VectorStoreManager()
        vectorstore = vectorstore_manager.create_vectorstore(split_docs)
        progress_bar.progress(1.0)
        st.success(f"{len(split_docs)}個の埋め込みでVector Storeを作成しました。")

        progress_bar.empty()
        status_text.empty()

        return vectorstore

    except Exception as e:
        st.error(f"ファイル処理中にエラーが発生しました: {str(e)}")
        import traceback

        with st.expander("エラー詳細"):
            st.code(traceback.format_exc())
        return None


def create_retrieval_chain(
    vectorstore, k: int, temperature: float
) -> Optional[RAGPipeline]:
    """
    RAG検索チェーンを作成

    Arguments:
        vectorstore: VectorStoreのインスタンス
        k: 取得ドキュメント数
        temperature: LLMのtemperature設定

    Returns:
        RAGPipelineインスタンス or None(失敗した場合)
    """
    try:
        with st.spinner("RAGパイプラインを初期化中..."):
            llm_manager = LLMManager(temperature=temperature)

            retriever_config = RetrieverConfig(k=k)
            rag_pipeline = RAGPipeline(
                vectorstore=vectorstore,
                llm_manager=llm_manager,
                retriever_config=retriever_config,
            )
            st.success("RAGパイプライン準備完了")
            return rag_pipeline

    except Exception as e:
        st.error(f"検索チェーンの作成中にエラーが発生しました: {str(e)}")
        import traceback

        with st.expander("エラー詳細"):
            st.code(traceback.format_exc())
        return None


def render_qa_interface():
    """
    Q&A インターフェース
    """
    if not st.session_state.rag_pipeline:
        st.info("サイドバーからドキュメントをアップロードして処理してください")
        st.markdown(
            """
        ### 開始方法:
        1. サイドバーをクリック
        2. PDF or TXT ファイルをアップロード
        3. 「ドキュメント処理」をクリック
        4. ここで質問
        """
        )
        return

    question = st.text_input(
        "質問を入録してください:",
        placeholder="例: このドキュメントの主なトピックは？",
        help="アップロードしたドキュメントについて質問",
        key="question_input",
    )

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        ask_button = st.button("質問する", use_container_width=True, type="primary")
    with col2:
        show_sources = st.checkbox("ソース表示", value=True)
    with col3:
        clear_button = st.button("履歴クリア", use_container_width=True)

    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    if ask_button and question:
        with st.spinner("考え中..."):
            try:
                """
                if show_sources:
                    result = st.session_state.rag_pipeline.query_with_sources(question)
                    answer = result["answer"]
                    sources = result["sources"]
                else:
                    answer = st.session_state.rag_pipeline.query_with_sources(question)
                    sources = []

                st.session_state.chat_history.append(
                    {"question": question, "answer": answer, "sources": sources}
                )
                """
                st.markdown("**質問**")
                st.write(question)

                st.markdown("**回答**")

                if show_sources:
                    result = st.session_state.rag_pipeline.query_stream_with_sources(
                        question
                    )
                    sources = result["sources"]
                    stream = result["stream"]

                    answer = st.write_stream(stream)
                else:
                    stream = st.session_state.rag_pipeline.query_stream(question)
                    answer = st.write_stream(stream)
                    sources = []

                st.session_state.chat_history.append(
                    {"question": question, "answer": answer, "sources": sources}
                )

            except Exception as e:
                st.error(f"質問処理中にエラーが発生しました: {str(e)}")
                import traceback

                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())
    if st.session_state.chat_history:
        st.divider()
        st.subheader("会話履歴")

        for i, exchange in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(
                f"Q{len(st.session_state.chat_history) - i}: {exchange['question'][:80]}...",
                expanded=(i == 0),
            ):
                st.markdown("**質問**")
                st.write(exchange["question"])

                st.markdown("**回答**")
                st.write(exchange["answer"])

                if exchange.get("sources"):
                    st.markdown(
                        f"**参照元ドキュメント({len(exchange['sources'])}件):**"
                    )
                    for j, doc in enumerate(exchange["sources"], 1):
                        with st.container():
                            st.markdown(f"**ソース {j}:**")
                            preview = doc.page_content[:300]
                            if len(doc.page_content) > 300:
                                preview += "..."
                            st.text(preview)

                            if doc.metadata:
                                st.caption(f"メタデータ: {doc.metadata}")

                            st.markdown("---")


def render_sample_queries():
    """
    サンプルクエリセクション
    """
    if not st.session_state.rag_pipeline:
        return

    st.header("サンプル質問")
    st.markdown("サンプル質問でお試し:")

    sample_queries = [
        "このドキュメントの主なトピックは？",
        "主要事項を要約して",
        "どのような技術的詳細が記載されている？",
        "重要なデータはある？",
    ]

    def set_query(q):
        st.session_state.question_input = q

    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        with cols[i % 2]:
            st.button(
                query,
                key=f"sample{i}",
                use_container_width=True,
                on_click=set_query,
                args=(query,),
            )


def render_footer():
    """
    アプリのフッター
    """
    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 20px:'>
        <small>
        <strong>RAG system</strong>
        </small>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """
    アプリケーションのエントリーポイント
    """
    # セッション状態を初期化
    init_session_state()
    # ヘッダー読み込み
    render_header()
    # サイドバー読み込み、設定取得
    uploaded_files, chunk_size, chunk_overlap, k, temperature, process_button = (
        render_sidebar()
    )

    if process_button and uploaded_files:
        st.session_state.uploaded_files = [f.name for f in uploaded_files]

        vectorstore = process_uploaded_files(uploaded_files, chunk_size, chunk_overlap)

        if vectorstore:
            st.session_state.vectorstore = vectorstore

            rag_pipeline = create_retrieval_chain(vectorstore, k, temperature)

            if rag_pipeline:
                st.session_state.rag_pipeline = rag_pipeline
                st.balloons()
                st.rerun()

    # タブのメインコンテンツ
    tab1, tab2, tab3 = st.tabs(["質問応答", "サンプル", "ヘルプ"])

    with tab1:
        render_qa_interface()

    with tab2:
        render_sample_queries()

    with tab3:
        st.markdown(
            """
        ## 使い方

        ### ドキュメントをアップロード
        1. サイドバーを開く（隠れている場合は☰をクリック）
        2. 「ドキュメントアップロード」でPDFまたはTXTファイルを選択
        3. 「ドキュメントを処理」ボタンをクリック
        4. 処理が完了するまで待つ

        ### ステップ2: 質問する
        1. 「質問応答」タブに移動
        2. テキストボックスに質問を入力
        3. 「質問する」ボタンをクリック
        4. AI生成の回答を確認
        
        ### ステップ3: 結果を確認
        - 質問の下に回答が表示されます
        - 「ソース表示」をチェックすると使用されたソースドキュメントを確認できます
        - 過去の質問は会話履歴に自動保存されます
        - 「履歴クリア」でリセット可能

        ---

        ## 設定オプション
        
        サイドバーで以下の設定を調整できます:
        
        **処理パラメータ:**
        - **チャンクサイズ**: テキストチャンクのサイズ（500-2000文字）
        - **チャンクオーバーラップ**: チャンク間のオーバーラップ（0-500文字）
          - 推奨: チャンクサイズの20%でコンテキスト保持を改善
        
        **検索パラメータ:**
        - **検索結果数 (k)**: 検索するチャンク数（1-10）
          - kが大きいほどコンテキストが多いが処理が遅い
        - **Temperature**: LLMの創造性レベル（0.0-1.0）
          - 0.0 = 固定的、事実ベースの回答
          - 1.0 = 変動的、多様な回答
        
        ---
        """
        )

    # フッター
    render_footer()


if __name__ == "__main__":
    main()
