import litellm
from cachetools import TTLCache
from flask import Flask, request

from smolagents.server import ModelList, AgentCoordinator, FlaskServer, agent_builder

litellm.set_verbose = True

app = Flask(__name__)
cache = TTLCache(1024, 600)
model_list = ModelList()
coordinator = AgentCoordinator(model_list, agent_builder, cache)
server = FlaskServer(app, coordinator)


@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json

    return data


if __name__ == '__main__':
    server.run(debug=False)
