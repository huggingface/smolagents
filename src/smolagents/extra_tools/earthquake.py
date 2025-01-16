from ..tools import Tool
import requests
import json
from datetime import datetime

class GetEarthquakesTool(Tool):
    name = "get_earthquakes"
    description = "Gets recent earthquakes from USGS in the past hour"
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        """Gets recent earthquakes from USGS GeoJSON feed"""
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Format the earthquake data for better readability
            formatted_quakes = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                quake_time = datetime.fromtimestamp(props['time'] / 1000.0)
                time_ago = datetime.now() - quake_time
                
                formatted_quakes.append({
                    "magnitude": props['mag'],
                    "location": props['place'],
                    "coordinates": [coords[1], coords[0]],  # lat, lon
                    "depth_km": coords[2],
                    "minutes_ago": int(time_ago.total_seconds() / 60)
                })
            
            return json.dumps(formatted_quakes, indent=2)
            
        except requests.RequestException as e:
            return f"Error fetching earthquake data: {str(e)}"
        except (KeyError, ValueError) as e:
            return f"Error parsing earthquake data: {str(e)}" 