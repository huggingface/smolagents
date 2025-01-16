from ..tools import Tool
import requests

class GetIPLocationTool(Tool):
    name = "get_ip_location"
    description = "Gets geolocation data for an IP address using freeipapi.com"
    inputs = {
        "ip": {
            "type": "string",
            "description": "The IP address to geolocate",
        }
    }
    output_type = "string"

    def forward(self, ip: str) -> str:
        """Gets location data for an IP using freeipapi.com"""
        try:
            response = requests.get(f'https://freeipapi.com/api/json/{ip}')
            response.raise_for_status()
            data = response.json()
            return f"{data['latitude']}, {data['longitude']}"
        except requests.RequestException as e:
            return f"Error getting location: {str(e)}" 