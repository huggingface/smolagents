import os
import Config
import logging
import traceback

from agents.contract_finder_agent.contract_finder_agent import ContractFinderAgent
from llm_studio_agents.utils.templates.health_check import health_bp
from llm_studio_agents.utils.utils import process_config, get_setup_details, deep_update_dict
from llm_studio_agents.utils.agent_enums import input_enum

import google.cloud.logging

from flask import Flask, jsonify, request
from flask_cors import cross_origin

# Setup logging
logging.getLogger().handlers = []  # FIXME: Comment this before running locally
client = google.cloud.logging.Client()
client.setup_logging()

app = Flask(__name__)
app.json.sort_keys = False

# logging.basicConfig(level=logging.DEBUG)

PREFIX = "/utility/contract-retrieval-rag-agent"  # NOTE: /utility/<agent-pod-name> should be kept here
AGENT_NAME = ContractFinderAgent

app.register_blueprint(health_bp, url_prefix=f'{PREFIX}/health')

@app.route(f'{PREFIX}/')
@cross_origin(supports_credentials=True)
def home():
    return f"{AGENT_NAME.__name__} is running!", 200  # NOTE: health check


def prediction(data, trace):
    """
    Orchestrator function that instantiates the agent, configures it, and executes the prediction.
    
    Args:
        data: Request payload containing agent_arguments, metadata, and optional agent_config
        trace: Execution trace dictionary
        
    Returns:
        Tuple of (response, retrieved_documents, citations, bypass_orchestrator_response)
        
    Raises:
        Exception: On configuration fetch failure or agent setup failure
    """
    rag_agent = AGENT_NAME()
    logging.info("######################### INPUT DATA #########################")
    logging.info(f"Input data: {data}")
    
    # Fetch or validate agent configuration
    if data.get('agent_config', None):
        logging.info("Got the setup config from the input data payload")
        agent_config = data.get('agent_config')
    else:
        logging.info("Calling MongoDB to fetch the setup config")
        agent_config = get_setup_details(data['concierge_id'], data['agent_id'])
    
    if agent_config is None:
        raise Exception("Could not fetch setup config from MongoDB")
    
    # Override configuration with request-provided settings
    agent_settings = data.get(input_enum.AGENT_SETTINGS.value, {})
    override_section = agent_settings.get(input_enum.OVERRIDE_AGENT_CONFIG.value) or {}
    override_config = override_section.get(rag_agent.__class__.__name__)

    if override_config:
        logging.info("Overriding the config.")
        agent_config = deep_update_dict(agent_config, override_config)

    # Process configuration and initialize tools
    agent_config = process_config(config=agent_config, sub_level="tools")

    # Setup agent with configuration and data
    logging.info("Setting up agent")
    rag_agent.setup(config=agent_config, data=data, agent_id=data['agent_id'])

    next_trace = {}
    logging.info("Setup completed")
    
    # Execute agent run method (to be implemented in later ACTs)
    response, retrieved_documents, citations, bypass_orchestrator_response = rag_agent.run(next_trace=next_trace, trace=trace, **data['agent_arguments'])
    return response, retrieved_documents, citations, bypass_orchestrator_response


@app.route(f'{PREFIX}/prediction', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict_api():
    """
    API endpoint for contract retrieval agent predictions.
    Handles request validation, agent setup, and response formatting.
    
    Returns:
        JSON response with agent output or error message
        Status 200 on success, 400 on validation error, 500 on setup/execution error
    """
    trace = {}
    try:
        # Extract and parse request payload
        data = request.get_json()
        
        # Call prediction orchestrator (includes setup)
        response, retrieved_documents, citations, bypass_orchestrator_response = prediction(data, trace)
        
        # Format documents with retrieved contract details (ACT 15: Step A6·B2)
        documents = None
        if retrieved_documents:
            documents = {
                "retrieved_documents": retrieved_documents,
            }

        # Build metadata with execution trace for observability and debugging
        metadata = {
            "retrieved_documents_count": len(retrieved_documents) if retrieved_documents else 0,
            "citations_count": len(citations) if citations else 0,
            "trace_metadata": trace,
            "execution_status": "success"
        }
        
        # Append trace root if present (from agent execution)
        if 'root' in trace:
            metadata["trace_root"] = trace.pop('root')

        # Return successful response with spec-required keys
        # Per ACT 15: response, documents, citations, metadata
        return jsonify({
            "response": response,
            "documents": documents,
            "citations": citations,
            "metadata": metadata,
            "bypass_orchestrator_response": bypass_orchestrator_response
        }), 200
        
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        # Return error response with 500 status
        return jsonify({
            "error": str(e),
            "trace": trace,
            "trace_root": trace.pop('root', None)
        }), 500


@app.route(f'{PREFIX}/get_config', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_config_api():
    try:
        response = AGENT_NAME.get_setup_config()
        if response:
            return jsonify(response), 200
        else:
            return jsonify({"message": "No configuration found"}), 204
    except Exception as e:
        logging.info(f"Error in get_config_api: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route(f'{PREFIX}/get_llm_config', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_llm_config_api():
    try:
        response = AGENT_NAME().get_llm_config()
        if response:
            return jsonify(response), 200
        else:
            return jsonify({"message": "No LLM configuration found"}), 204
    except Exception as e:
        logging.info(f"Error in get_llm_config_api: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5002, debug=True) # FIXME: Uncomment this before running locally

    gunicorn_command = (
        f"gunicorn --workers {Config.WORKER_COUNT} --worker-class gthread --bind 0.0.0.0:5002 "
        f"--timeout {Config.WORKER_TIMEOUT} --keep-alive 120 --max-requests {Config.MAX_REQUEST_TO_WORKER_RESTART} --max-requests-jitter 50 "
        f"--log-level info --threads {Config.WORKER_THREADS_COUNT} --access-logfile - --error-logfile - "
        f"--graceful-timeout {Config.WORKER_GRACEFUL_TIMEOUT} --limit-request-line 8190 run:app"
    )
    
    os.system(gunicorn_command)  # FIXME: Comment this before running locally
