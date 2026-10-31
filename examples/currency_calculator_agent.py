"""
Currency Calculator Agent example for smolagents.
This script defines a tool to fetch exchange rates and uses a CodeAgent to perform conversions.
"""

import requests

from smolagents import ApiModel, CodeAgent, tool


@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> float:
    """Fetches the exchange rate from a base currency to a target currency.

    Args:
        base_currency: Currency code to convert from, e.g., 'USD'.
        target_currency: Currency code to convert to, e.g., 'EUR'.
    Returns:
        The exchange rate as a float.
    """
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    if data.get("result") != "success":
        raise RuntimeError(f"Failed to fetch rates for {base_currency}: {data.get('error-type')}")
    rates = data.get("rates", {})
    if target_currency not in rates:
        raise ValueError(f"Target currency {target_currency} not found in rates.")
    return float(rates[target_currency])


if __name__ == "__main__":
    # Initialize the model (using the generic ApiModel)
    model = ApiModel()
    agent = CodeAgent(tools=[get_exchange_rate], model=model, verbosity_level=2)
    # Example query: convert 150 USD to INR
    query = "Convert 150 USD to INR using the default exchange rate."
    print(f"Task: {query}\n")
    result = agent.run(query)
    print(f"\nAgent Output:\n{result}")
