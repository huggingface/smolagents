from smolagents.jina_web_reader import JinaWebReaderTool

def test_tool():
    print("🔄 Initializing Jina Tool...")
    tool = JinaWebReaderTool()
    
    # We test on a real, documentation-heavy URL
    test_url = "https://huggingface.co/docs/smolagents/index"
    
    print(f"🌍 Fetching: {test_url}")
    result = tool.forward(test_url)
    
    print("\n✅ RESULT SNIPPET (First 500 chars):")
    print("-" * 50)
    print(result[:500])
    print("-" * 50)

    # Validation logic
    if "smolagents" in result.lower() or "documentation" in result.lower():
        print("\n🚀 SUCCESS: Content verified.")
    else:
        print("\n❌ FAILURE: Content seems empty or wrong.")

if __name__ == "__main__":
    test_tool()
