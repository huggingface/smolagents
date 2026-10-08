# Bedrock Managed Knowledge Base Support

## Overview
Adds a Bedrock Knowledge Base tool for smolagents that performs managed retrieval without requiring a local vector store.

## Usage
```python
from smolagents import CodeAgent, HfApiModel
from smolagents.tools import BedrockKnowledgeBaseTool

kb_tool = BedrockKnowledgeBaseTool(knowledge_base_id="YOUR_KB_ID")
agent = CodeAgent(tools=[kb_tool], model=HfApiModel())
agent.run("What does our documentation say about authentication?")
```

## Configuration
| Variable | Description | Default |
|---|---|---|
| KNOWLEDGE_BASE_ID | Bedrock Knowledge Base ID | None |
| AWS_REGION | AWS region for the KB | us-east-1 |
| AWS_PROFILE | AWS credentials profile | None |
| USE_AGENTIC_RETRIEVAL | Enable agentic retrieval | true |
| MAX_RESULTS | Maximum retrieval results | 5 |

## Features
- Managed search (no vector store needed)
- Agentic retrieval with query decomposition + reranking
- Automatic fallback to plain Retrieve if agentic fails
- Multi-source support (S3, Web, Confluence, SharePoint)
- Compatible with smolagents Tool interface

## SDK Requirements
- boto3 >= 1.43
- smolagents >= 1.0

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieveStream"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```

## References
- [Build a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html)
- [Retrieve API](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html)
- [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)
