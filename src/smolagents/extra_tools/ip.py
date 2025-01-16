from ..tools import Tool
import requests

class GetPublicIPTool(Tool):
    name = "get_public_ip"
    description = "Gets the current public IP address using ipify.org"
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        """Gets public IP using ipify.org's free API"""
        try:
            response = requests.get('https://api.ipify.org?format=json')
            response.raise_for_status()
            return response.json()['ip']
        except requests.RequestException as e:
            return f"Error fetching IP address: {str(e)}" 