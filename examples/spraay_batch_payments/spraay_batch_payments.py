"""
Spraay Batch Payments Tool for smolagents

Enables AI agents to execute batch cryptocurrency payments on Base
using the Spraay protocol. Send ETH and ERC-20 tokens to up to 200
recipients in a single transaction with ~80% gas savings.

Requirements:
    pip install smolagents web3

Environment Variables:
    SPRAAY_PRIVATE_KEY: Private key for the wallet executing payments
    SPRAAY_RPC_URL: RPC endpoint (default: https://mainnet.base.org)
"""

import os
import json
from typing import Optional

from smolagents import tool, CodeAgent, InferenceClientModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPRAAY_CONTRACT = "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC"
DEFAULT_RPC = "https://mainnet.base.org"

SPRAAY_ABI = json.loads("""[
    {"inputs":[{"internalType":"address[]","name":"recipients","type":"address[]"},{"internalType":"uint256","name":"amountEach","type":"uint256"}],"name":"batchSendETH","outputs":[],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"address[]","name":"recipients","type":"address[]"},{"internalType":"uint256","name":"amountEach","type":"uint256"}],"name":"batchSendToken","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address[]","name":"recipients","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"name":"batchSendETHVariable","outputs":[],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"address[]","name":"recipients","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"name":"batchSendTokenVariable","outputs":[],"stateMutability":"nonpayable","type":"function"}
]""")

ERC20_APPROVE_ABI = json.loads("""[{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"approve","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"}]""")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def spraay_batch_send_eth(recipients: list, amount_each_ether: str) -> str:
    """Send equal amounts of ETH to multiple recipients in a single transaction
    on Base using the Spraay batch payment protocol. Saves ~80% on gas fees.
    Maximum 200 recipients per transaction. A 0.3% protocol fee applies.

    Args:
        recipients: List of recipient wallet addresses as hex strings starting with 0x.
        amount_each_ether: Amount of ETH to send to each recipient as a string, e.g. '0.01'.

    Returns:
        Transaction result with BaseScan link.
    """
    from web3 import Web3

    if len(recipients) > 200:
        return "Error: Maximum 200 recipients per transaction."

    w3 = Web3(Web3.HTTPProvider(os.getenv("SPRAAY_RPC_URL", DEFAULT_RPC)))
    account = w3.eth.account.from_key(os.getenv("SPRAAY_PRIVATE_KEY"))
    contract = w3.eth.contract(address=w3.to_checksum_address(SPRAAY_CONTRACT), abi=SPRAAY_ABI)

    amount_wei = w3.to_wei(amount_each_ether, "ether")
    total_wei = amount_wei * len(recipients)
    fee = total_wei * 3 // 1000
    addrs = [w3.to_checksum_address(r) for r in recipients]

    tx = contract.functions.batchSendETH(addrs, amount_wei).build_transaction({
        "from": account.address,
        "value": total_wei + fee,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 21000 + (len(recipients) * 30000),
        "maxFeePerGas": w3.eth.gas_price * 2,
        "maxPriorityFeePerGas": w3.to_wei("0.001", "gwei"),
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    status = "Success" if receipt.status == 1 else "Failed"
    return f"Batch ETH sent! Recipients: {len(recipients)}, Amount each: {amount_each_ether} ETH, Status: {status}, Tx: https://basescan.org/tx/{tx_hash.hex()}"


@tool
def spraay_batch_send_token(token_address: str, recipients: list, amount_each: str, decimals: Optional[int] = 18) -> str:
    """Send equal amounts of an ERC-20 token (USDC, USDT, etc.) to multiple recipients
    in a single transaction on Base using the Spraay batch payment protocol.
    Automatically handles token approval. Maximum 200 recipients.

    Args:
        token_address: ERC-20 token contract address on Base.
        recipients: List of recipient wallet addresses as hex strings.
        amount_each: Token amount per recipient in human units, e.g. '100' for 100 USDC.
        decimals: Token decimal places. Use 6 for USDC/USDT, 18 for most other tokens.

    Returns:
        Transaction result with BaseScan link.
    """
    from web3 import Web3

    if len(recipients) > 200:
        return "Error: Maximum 200 recipients per transaction."

    w3 = Web3(Web3.HTTPProvider(os.getenv("SPRAAY_RPC_URL", DEFAULT_RPC)))
    account = w3.eth.account.from_key(os.getenv("SPRAAY_PRIVATE_KEY"))
    contract = w3.eth.contract(address=w3.to_checksum_address(SPRAAY_CONTRACT), abi=SPRAAY_ABI)

    amount_raw = int(float(amount_each) * (10 ** decimals))
    total_raw = amount_raw * len(recipients)
    token_cs = w3.to_checksum_address(token_address)
    addrs = [w3.to_checksum_address(r) for r in recipients]

    # Approve
    token_contract = w3.eth.contract(address=token_cs, abi=ERC20_APPROVE_ABI)
    approve_tx = token_contract.functions.approve(
        w3.to_checksum_address(SPRAAY_CONTRACT), total_raw
    ).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 60000,
        "maxFeePerGas": w3.eth.gas_price * 2,
        "maxPriorityFeePerGas": w3.to_wei("0.001", "gwei"),
    })
    signed_approve = account.sign_transaction(approve_tx)
    approve_hash = w3.eth.send_raw_transaction(signed_approve.raw_transaction)
    w3.eth.wait_for_transaction_receipt(approve_hash)

    # Batch send
    tx = contract.functions.batchSendToken(token_cs, addrs, amount_raw).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 21000 + (len(recipients) * 50000),
        "maxFeePerGas": w3.eth.gas_price * 2,
        "maxPriorityFeePerGas": w3.to_wei("0.001", "gwei"),
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    status = "Success" if receipt.status == 1 else "Failed"
    return f"Batch token sent! Token: {token_address}, Recipients: {len(recipients)}, Amount each: {amount_each}, Status: {status}, Tx: https://basescan.org/tx/{tx_hash.hex()}"


@tool
def spraay_batch_send_eth_variable(recipients: list, amounts_ether: list) -> str:
    """Send different amounts of ETH to multiple recipients in a single transaction
    on Base using the Spraay batch payment protocol. Each recipient receives a custom amount.
    Maximum 200 recipients.

    Args:
        recipients: List of recipient wallet addresses as hex strings.
        amounts_ether: List of ETH amounts as strings matching each recipient, e.g. ['0.01', '0.05'].

    Returns:
        Transaction result with BaseScan link.
    """
    from web3 import Web3

    if len(recipients) > 200:
        return "Error: Maximum 200 recipients per transaction."
    if len(recipients) != len(amounts_ether):
        return "Error: Number of recipients must match number of amounts."

    w3 = Web3(Web3.HTTPProvider(os.getenv("SPRAAY_RPC_URL", DEFAULT_RPC)))
    account = w3.eth.account.from_key(os.getenv("SPRAAY_PRIVATE_KEY"))
    contract = w3.eth.contract(address=w3.to_checksum_address(SPRAAY_CONTRACT), abi=SPRAAY_ABI)

    amounts_wei = [w3.to_wei(a, "ether") for a in amounts_ether]
    total_wei = sum(amounts_wei)
    fee = total_wei * 3 // 1000
    addrs = [w3.to_checksum_address(r) for r in recipients]

    tx = contract.functions.batchSendETHVariable(addrs, amounts_wei).build_transaction({
        "from": account.address,
        "value": total_wei + fee,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 21000 + (len(recipients) * 30000),
        "maxFeePerGas": w3.eth.gas_price * 2,
        "maxPriorityFeePerGas": w3.to_wei("0.001", "gwei"),
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    total_eth = sum(float(a) for a in amounts_ether)
    status = "Success" if receipt.status == 1 else "Failed"
    return f"Variable ETH batch sent! Recipients: {len(recipients)}, Total: {total_eth} ETH, Status: {status}, Tx: https://basescan.org/tx/{tx_hash.hex()}"


# ---------------------------------------------------------------------------
# Example Agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = InferenceClientModel()

    agent = CodeAgent(
        tools=[spraay_batch_send_eth, spraay_batch_send_token, spraay_batch_send_eth_variable],
        model=model,
    )

    result = agent.run(
        "Send 0.001 ETH to each of these 2 addresses on Base: "
        "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18 and "
        "0x53d284357ec70cE289D6D64134DfAc8E511c8a3D"
    )
    print(result)
