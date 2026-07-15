import boto3
from smolagents import CodeAgent, AmazonBedrockModel

session = boto3.Session(
    profile_name="your-profile",
    region_name="us-west-2"
)
client = session.client("bedrock-runtime")

model = AmazonBedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    client=client
)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.run("Could you give me the 118th number in the Fibonacci sequence?")
