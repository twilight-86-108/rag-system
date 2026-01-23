from socket import timeout
from typing import Optional, Dict, Any, List
import logging

# LangChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import BaseMessage

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMConfig:
    """
    LLMの設定を管理

    Attributes:
        model: 使用するOllamaのモデル名
        temperature: 生成のランダム性
        base_url: OllamaサーバーのURL
        timeout: タイムアウト時間
    """

    DEFAULT_MODEL = "llama3.1:8b"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """
        Arguments:
            model: 使用するOllamaのモデル名
            temperature: 生成のランダム性
            base_url: OllamaサーバーのURL
            timeout: タイムアウト時間
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout

        logger.info(
            f"LLMConfig initialized: model={self.model},"
            f"temperature={temperature}, base_url={base_url}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書形式で返す
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }


def create_llm(config: Optional[LLMConfig] = None) -> ChatOllama:
    """
    ChatOllamaインスタンスを作成

    Arguments:
        config: LLM設定（Noneならデフォルト）

    Returns:
        ChatOllama: 設定済みLLMインスタンス
    """
    # configがNoneならデフォルト設定を使用
    if config is None:
        config = LLMConfig()
    # ChatOllamaをインスタンス化
    llm = ChatOllama(
        model=config.model,
        temperature=config.temperature,
        base_url=config.base_url,
        timeout=config.timeout,
    )
    # ロギング出力
    logger.info(f"Created ChatOllama with model: {config.model}")
    # 作成したllm（ChatOllamaインスタンス）を返却
    return llm


RAG_SYSTEM_PROMPT = """
あなたは、提供された文書に基づいて質問に回答するAIアシスタントです。
以下のルールを厳守してください：

1. **文書ベース回答**: 提供されたコンテキスト（文書）の情報のみを使用して回答してください。
2. **不明な場合の対応**: 提供された文書に情報がない場合は、「提供された文書には、その情報が含まれていません」と正直に答えてください。憶測や一般知識で補完しないでください。
3. **引用の推奨**: 可能な限り、どの文書から情報を得たかを示してください（例: "提供された文書によると..."）。
4. **簡潔な回答**: 質問に対して簡潔かつ正確に答えてください。冗長な説明は避けてください。
5. **日本語対応**: 日本語の質問には日本語で、英語の質問には英語で回答してください。

提供される情報:
- コンテキスト: 質問に関連する可能性のある文書の抜粋
- 質問: ユーザーからの具体的な質問
"""

RAG_SYSTEM_PROMPT_WITH_HISTORY = """
あなたは、提供された文書に基づいて質問に回答するAIアシスタントです。

## 最重要ルール: 会話の継続性
- ユーザーが「それ」「これ」「その」「上記」などの指示語を使った場合、**必ず直前の会話履歴を参照**して何を指しているか特定してください。
- 例: 前の質問が「RAGとは？」で、次の質問が「それの利点は？」なら、「RAGの利点」について回答してください。

## 回答ルール
1. **文書ベース回答**: 提供されたコンテキスト（文書）の情報を優先して使用してください。
2. **不明な場合**: 文書に情報がない場合は正直に「提供された文書には含まれていません」と答えてください。
3. **引用推奨**: 可能な限り、情報の出典を示してください。
4. **簡潔な回答**: 質問に対して簡潔かつ正確に答えてください。
5. **言語対応**: 質問と同じ言語で回答してください。

## 会話履歴の活用
以下の会話履歴を参考に、文脈を理解して回答してください。
"""

RAG_HUMAN_TEMPLATE = """
コンテキスト：{context}

質問：{question}

回答：
"""

RAG_HUMAN_TEMPLATE_WITH_HISTORY = """
コンテキスト: {context}

質問: {question}

回答: 
"""

def create_rag_prompt() -> ChatPromptTemplate:
    """
    RAG用のChatPromptTemplateを作成

    Returns:
        ChatPromptTemplate: システムプロンプト＋RAG_HUMAN_TEMPLATE
    """
    # SystemMessagePromptTemplateを作成
    system_message = SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    # HumanMessagePromptTemplateを作成
    human_message = HumanMessagePromptTemplate.from_template(RAG_HUMAN_TEMPLATE)
    # ChatPromptTemplate.from_messages()で結合
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            human_message,
        ]
    )
    # logging
    logger.info("Created RAG prompt template with system and human messages")

    return prompt

def create_rag_prompt_with_history() -> ChatPromptTemplate:
    """
    会話履歴対応のRAG用ChatPromptTemplateを作成

    Returns:
        ChatPromptTemplate: システム + 履歴 + ユーザー入力
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT_WITH_HISTORY),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", RAG_HUMAN_TEMPLATE_WITH_HISTORY),
        ]
    )

    logger.info("Created RAG prompt template with chat history support")

    return prompt

class LLMManager:
    """
    LLMの作成と管理を担当
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        model: str = LLMConfig.DEFAULT_MODEL,
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434",
    ):
        """
        Arguments:
            config: LLM設定 (優先)
            model: (configがない場合に使用)
            temperature: (configがない場合に使用)
            base_url: (configがない場合に使用)
        """
        if config is None:
            config = LLMConfig(model=model, temperature=temperature, base_url=base_url)
        self.config = config
        self.llm = create_llm(config)

        logger.info(
            f"LLMManager initialized with model: {config.model},"
            f"temperature: {config.temperature}"
        )

    def get_llm(self) -> ChatOllama:
        """
        LLMインスタンスを取得

        Returns:
            ChatOllama: 設定済みLLM
        """
        return self.llm

    def test_llm(self, test_message: str = "Hello, how are you?") -> str:
        """
        LLMの動作確認テスト

        Arguments:
            test_message: テストメッセージ

        Returns:
            str: LLMからの回答

        Raises:
            Exception: Ollamaサーバーが未起動、モデル未ダウンロードなど
        """
        logger.info(f"Testing LLM with message: '{test_message}'")

        try:
            response = self.llm.invoke(test_message)

            response_text = response.content

            logger.info(f"LLM response received: {len(response_text)} chars")

            return response_text

        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            raise


def format_context_from_docs(documents: List) -> str:
    """
    検索結果のDocumentリストをコンテキスト文字列に変換

    Arguments:
        documents: Vector Storeからの検索結果

    Returns:
        str: プロンプトに注入するコンテキスト文字列
    """
    if not documents:
        return "関連する文書が見つかりませんでした。"

    context_parts = []

    for i, doc in enumerate(documents, 1):
        context_parts.append(f"===Document {i}===")
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"Source: {source}")
        context_parts.append("")

    return "\n".join(context_parts)


if __name__ == "__main__":
    # LLM接続テスト
    print("Test 1")
    try:
        llm_manager = LLMManager()
        response = llm_manager.test_llm("こんにちは")
        print(f"LLM Response: {response}\n")

    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback

        traceback.print_exc()

    # プロンプトテンプレート
    print("Test 2")
    try:
        prompt = create_rag_prompt()
        sample_context = """
            RAG (Retrieval-Augmented Generation):
            大規模言語モデルに外部知識ベースを組み合わせる手法。
            検索(Retrieval)と生成(Generation)を統合することで、
            ハルシネーション(幻覚)を低減し、信頼性の高い応答を実現する。
        """
        sample_question = "RAGとはなにか簡潔に説明して。"

        formatted_messeges = prompt.format_messages(
            context=sample_context, question=sample_question
        )

        print("\nFormatted Prompt:")
        for i, msg in enumerate(formatted_messeges, 1):
            print(f"\nMessage {i}({msg.__class__.__name__})")
            print(msg.content)

    except Exception as e:
        print(f"Test 2 failed: {e}\n")
        import traceback

        traceback.print_exc()

    # 簡易RAGシミュレーション
    print("Test 3")
    try:
        if llm_manager is None:
            llm_manager = LLMManager()

        llm = llm_manager.get_llm()
        prompt = create_rag_prompt()

        chain = prompt | llm

        response = chain.invoke(
            {"context": sample_context, "question": sample_question}
        )

        print(f"\nRAG Response: {response.content}")
        print()

    except Exception as e:
        print(f"Test 3 failed: {e}\n")
        import traceback

        traceback.print_exc()

    # Context Formatting
    print("Test 4")
    try:
        from langchain_core.documents import Document

        mock_docs = [
            Document(
                page_content="RAGは検索拡張生成と呼ばれる技術です。",
                metadata={"source": "data\sample_tech_doc.txt", "chunk_index": 0},
            ),
            Document(
                page_content="ベクトルデータベースを使用して類似文書を検索します。",
                metadata={"source": "data/sample_tech_doc.txt", "chunk_index": 5},
            ),
        ]

        context_formatted = format_context_from_docs(mock_docs)

        print("\nFormatted Context:")
        print(context_formatted)

    except Exception as e:
        print(f"Test 4 failed: {e}\n")
        import traceback

        traceback.print_exc()

    print("\nAll tests completed.\n")
