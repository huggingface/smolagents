import logging
import os
from collections import Counter

from smolagents import OpenAIServerModel, ToolCallingAgent, tool


logging.basicConfig(
    filename="DEBUGGING_LOG.txt",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    force=True,
)
logger = logging.getLogger(__name__)


# python3 -m venv .venv
# source .venv/bin/activate
# pip install -e .
# pip install pyyaml, openai
# python3 examples/multi_agent_workflow.py

#
# #
# 1. model setup
# #
# #  - multiple agents can use the same model, or you can use different models for different agents
# #  - in this example, we will use the OpenAI API for all of our agents
# #

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
model = OpenAIServerModel(model_id="gpt-4.1-nano", api_key=openai_api_key)


#
# #
# 2. define a task
# #
# #  - we will have the team of agents work together to process returns data and identify the most returned product
# #  - this task will entail:
# #      - iterating through the returns data
# #      - if the return is a 'whole order', the agent will need to look up the order in ShipStation to get the details
# #      - adding each return to a dataset
# #      - analyzing the dataset to find the most returned product
# #      - returning a report of the top returned product

team_goal = "Your primary objective is to identify the most returned product from our returns data and provide a report of the most returned item. you will need to systematically go through every row of the exceptions returns data until there are no more remaining. If a row explicitly lists a product_id, the you will need to add this return directly to our returns dataset, ensuring all relevant return data (order ID, item ID, reason, and quantity) is captured. If a row indicates a whole order return, you will need to retrieve all individual SKUs associated with that order from shipstation. those skus and returns data will then need to be added to our returns dataset, ensuring each SKU is associated with the original order data from the exceptions returns. Since there are a significant number of returns, you must confirm that all rows have been processed. After processing a batch, you should actively check if there are more returns to process. Do not consider this phase complete until all returns data has been thoroughly reviewed and integrated. Once all returns data has been collected and added to the dataset you will need to Identify Top Returned Item. analyze the complete returns dataset to identify the product with the highest number of returns. provide a clear and concise report detailing the most returned product. i cannot stress this enough. if the agent says that all items have been added to the returns data, ask them to check again. we have been working on this for a long time and they are still not done."

#
# #
# 3. tools setup
# #
# #  - we will need to define tools for the agents to use to accomplish the task
# #  - in this example, were using an iterator to stream data from a fake returns dataset
# #


def _load_returns_data_stream():
    """
    stream returns data from the exceptions log, yielding one row at a time.
    """
    fake_rows = [
        {"order_id": "1001", "item_id": "WidgetA", "reason": "defective", "qty": 2},
        {"order_id": "1002", "item_id": "whole order", "reason": "damaged", "qty": 2},
        {
            "order_id": "1003",
            "item_id": "WidgetB",
            "reason": "not as described",
            "qty": 2,
        },
        {
            "order_id": "1004",
            "item_id": "WidgetC",
            "reason": "wrong item sent",
            "qty": 1,
        },
        {"order_id": "1005", "item_id": "WidgetA", "reason": "defective", "qty": 1},
        {
            "order_id": "1006",
            "item_id": "WidgetB",
            "reason": "not as described",
            "qty": 3,
        },
        {
            "order_id": "1007",
            "item_id": "whole order",
            "reason": "customer changed mind",
            "qty": 1,
        },
        {
            "order_id": "1008",
            "item_id": "WidgetC",
            "reason": "damaged in transit",
            "qty": 2,
        },
        {"order_id": "1009", "item_id": "WidgetA", "reason": "defective", "qty": 1},
        {
            "order_id": "1010",
            "item_id": "WidgetB",
            "reason": "not as described",
            "qty": 2,
        },
        {
            "order_id": "1011",
            "item_id": "WidgetC",
            "reason": "wrong item sent",
            "qty": 1,
        },
        {"order_id": "1012", "item_id": "whole order", "reason": "defective", "qty": 1},
        {
            "order_id": "1013",
            "item_id": "WidgetB",
            "reason": "not as described",
            "qty": 2,
        },
    ]
    for row in fake_rows:
        yield row


# #
# 3.1 tools setup
# #
# #  - the model will be able to get the next row from the exceptions log
# #


# Create a persistent generator instance for the lifecycle of the flow
_returns_data_stream = _load_returns_data_stream()


@tool
def load_next_exceptions_row() -> dict:
    """
    Returns the next row from the exceptions log until there are no more rows.
    """
    try:
        next_item = next(_returns_data_stream)
    except StopIteration:
        next_item = {}
    return next_item


# #
# 3.2 tools setup
# #
# #  - the model will be able to look up an order in ShipStation by order ID
# #


@tool
def lookup_order_in_shipstation(order_id: str) -> list[dict]:
    """
    look up order data in ShipStation by order id

    Args:
        order_id: The order ID to look up.
    """
    data = {
        "1001": [{"order_id": "1001", "item_id": "WidgetA", "qty": 2}],
        "1002": [
            {"order_id": "1002", "item_id": "WidgetC", "qty": 1},
            {"order_id": "1002", "item_id": "WidgetA", "qty": 1},
        ],
        "1003": [{"order_id": "1003", "item_id": "WidgetB", "qty": 2}],
        "1007": [
            {"order_id": "1007", "item_id": "WidgetB", "qty": 1},
            {"order_id": "1007", "item_id": "WidgetC", "qty": 1},
        ],
        "1012": [{"order_id": "1012", "item_id": "WidgetC", "qty": 1}],
    }

    return {"items": data.get(order_id, [])}


return_data = []

# #
# 3.3 tools setup
# #
# #  - the model will be able to add a product return to the returns dataset
# #  - this will also write the return to a file so we can see what the agent is doing


@tool
def add_product_return_to_database(order_id: str, item_id: str, reason: str, qty: int) -> bool:
    """
    add return to the returns dataset

    Args:
        order_id: The order ID of the return.
        item_id: The item_id being returned.
        reason: The reason for the return.
        qty: The qty of the item_id being returned.

    """
    global return_data
    return_data.append({"order_id": order_id, "item_id": item_id, "reason": reason, "qty": qty})
    with open("EXAMPLE_returns_data.txt", "a") as f:
        f.write(f"{order_id}, {item_id}, {reason}, {qty}\n")
    return True


# #
# 3.4 tools setup
# #
# #  - the model will be able to get the top N returned products
# #


@tool
def get_top_n_returned_products(n: int) -> list[tuple]:
    """
    Dummy tool to compile a report of the top N most returned products.
    Args:
        n: The number of top returned products to return.
    Returns:
        list of tuples of the top N returned products and their counts.
    """
    global return_data
    if not return_data:
        return []

    # Count occurrences of each item_id in the returns data
    item_counts = Counter(item["item_id"] for item in return_data)

    # Get the top N most common items
    top_items = item_counts.most_common(n)

    return top_items


#
# #
# 4. setup agents
# #
# #  - we will create a manager agent that will coordinate the work of the other agents
# #  - we will create three managed agents that will
# #


agent_1 = ToolCallingAgent(
    name="order_processing_agent",
    max_steps=100,
    description="an agent for processing orders and handling ecommerce info for shipstation",
    tools=[lookup_order_in_shipstation],
    model=model,
    verbosity_level=2,
)


agent_2 = ToolCallingAgent(
    name="returns_processing_agent",
    description="an agent for processing returns data",
    max_steps=100,
    tools=[load_next_exceptions_row, add_product_return_to_database],
    model=model,
    verbosity_level=2,
)

agent_3 = ToolCallingAgent(
    name="data_analysis_agent",
    description="an agent for getting the top returned product after the dataset has been processed",
    max_steps=100,
    tools=[get_top_n_returned_products],
    model=model,
    verbosity_level=2,
)


agent_0 = ToolCallingAgent(
    max_steps=100,
    tools=[],
    managed_agents=[agent_1, agent_2, agent_3],
    model=model,
    verbosity_level=2,
)


#
# manager agent receives the task
#
#    go through our returns data and identify the most returned product -- this will entail loading the exceptions returns data, and going through every row. if the product is listed, it will add that to the dataset, if it says 'whole order', it will need to look up the order in shipstation, identify all of the skus in the order, and add those to the dataset with the appropriate order data from the exceptions returns. then it will need to analyze the data and create some kind of report for the top returned item
#
# 2. manager agent creates a plan
# 3. manager agent delegates tasks to the managed agents
# 4. managed agents execute their tasks
# 5. manager agent assesses the results of the managed agents
# 6. manager agent continues to delegate tasks until the goal is achieved
# 7. manager agent provides a final report of the top returned product
#

output = agent_0.run(team_goal)

print(output)
