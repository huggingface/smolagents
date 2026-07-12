from smolagents import CodeAgent, models, tool, GradioUI, WebSearchTool

model = models.OpenAIModel()

agent = CodeAgent(
    tools=[WebSearchTool(engine="bing")],
    model=model,
    stream_outputs=True,
)

GradioUI(agent, file_upload_folder="./upload").launch(share=False)
