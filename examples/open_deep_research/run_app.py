

from smolagents.gradio_ui import GradioUI

from run import create_agent


agent = create_agent()

demo = GradioUI(agent)

if __name__ == "__main__":
    demo.launch()
