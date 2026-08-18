# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any


def prepare_mcp_server_parameters(server_parameters: Any) -> Any:
    """Validate HTTP transports and apply their default without mutating caller-owned values."""
    if isinstance(server_parameters, list):
        return [prepare_mcp_server_parameters(parameters) for parameters in server_parameters]

    if not isinstance(server_parameters, dict):
        return server_parameters

    transport = server_parameters.get("transport")
    if transport is None:
        transport = "streamable-http"
    if transport not in {"sse", "streamable-http"}:
        raise ValueError(f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'.")

    return {**server_parameters, "transport": transport}
