from .tools import Tool

class HubSearchTool(Tool):
    name = "HubSearchTool"
    description = "Search Hugging Face Hub for models and datasets by query. Returns top results by downloads."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search term or task (e.g., 'text-to-image', 'bert', 'finance dataset')."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """
        Initialize the Hugging Face Hub Search tool.
        """
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        """
        Searches the Hugging Face Hub for Models and Datasets.
        Useful for finding state-of-the-art models for specific tasks (e.g., 'text-classification', 'llama').
        
        Args:
            query: The search term or task (e.g., 'text-to-image', 'bert', 'finance dataset').
        """
        try:
            # Lazy Import (Standard HF library)
            from huggingface_hub import HfApi
            api = HfApi()
            
            # 1. Search Models (Top 3 by Downloads)
            models = api.list_models(
                search=query,
                sort="downloads",
                direction=-1,
                limit=3
            )
            
            # 2. Search Datasets (Top 3 by Downloads)
            datasets = api.list_datasets(
                search=query,
                sort="downloads", 
                direction=-1,
                limit=3
            )
            
            results = []
            
            # Format Models
            if models:
                results.append("🤖 **Top Models:**")
                for m in models:
                    likes = getattr(m, 'likes', 0)
                    downloads = getattr(m, 'downloads', 0)
                    results.append(
                        f"- ID: {m.modelId}\n"
                        f"  ❤️ Likes: {likes} | ⬇️ Downloads: {downloads}\n"
                        f"  🔗 https://huggingface.co/{m.modelId}"
                    )
            
            # Format Datasets
            if datasets:
                results.append("\n📊 **Top Datasets:**")
                for d in datasets:
                    likes = getattr(d, 'likes', 0)
                    downloads = getattr(d, 'downloads', 0)
                    results.append(
                        f"- ID: {d.id}\n"
                        f"  ❤️ Likes: {likes} | ⬇️ Downloads: {downloads}\n"
                        f"  🔗 https://huggingface.co/datasets/{d.id}"
                    )

            if not results:
                return f"No models or datasets found on the Hub for '{query}'."

            return "\n".join(results)

        except Exception as e:
            return f"Error searching HF Hub: {str(e)}"
