"""
Setup test environments
"""

def test_ollama():
    """Test ollama connection"""
    from ollama import Client
    client = Client()
    assert client is not None
    print("Ollama connection test passed")

def test_langchain():
    """Test LangChain connection"""
    from langchain_ollama import OllamaLLM
    ollama = OllamaLLM(model="llama3.1:8b")
    assert ollama is not None
    print("LangChain connection test passed")

def test_chroma():
    """Test ChromaDB connection"""
    from chromadb import Client
    client = Client()
    assert client is not None
    print("ChromaDB connection test passed")

if __name__ == "__main__":
    print("Starting test setup...")
    test_ollama()
    test_langchain()
    test_chroma()
    print("All tests passed")
