import requests
from .tools import Tool

class OpenMeteoTool(Tool):
    name = "OpenMeteoTool"
    description = "Get real-time weather (temperature, humidity, wind) for any city using Open-Meteo. No API key required."
    inputs = {
        "location": {
            "type": "string",
            "description": "The name of the city (e.g., 'Pune', 'New York', 'Tokyo')."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """
        Initialize the OpenMeteo Weather tool.
        """
        super().__init__(**kwargs)

    def forward(self, location: str) -> str:
        """
        Retrieves current weather data for a specific location using Open-Meteo API.
        Returns Temperature, Wind Speed, and Humidity.
        
        Args:
            location: The name of the city (e.g., 'Pune', 'New York', 'Tokyo').
        """
        try:
            # 1. Geocoding (City -> Lat/Lon)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_response = requests.get(geo_url, timeout=10)
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                return f"Error: Could not find coordinates for location '{location}'."

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            city_name = geo_data["results"][0]["name"]
            country = geo_data["results"][0].get("country", "Unknown")

            # 2. Fetch Weather
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            )
            w_response = requests.get(weather_url, timeout=10)
            w_data = w_response.json()

            current = w_data.get("current", {})
            
            # 3. Format Output
            temp = current.get("temperature_2m", "N/A")
            humid = current.get("relative_humidity_2m", "N/A")
            wind = current.get("wind_speed_10m", "N/A")
            units = w_data.get("current_units", {})

            return (
                f"🌤️ **Weather Report for {city_name}, {country}**\n"
                f"🌡️ Temperature: {temp}{units.get('temperature_2m', '°C')}\n"
                f"💧 Humidity: {humid}{units.get('relative_humidity_2m', '%')}\n"
                f"💨 Wind Speed: {wind}{units.get('wind_speed_10m', 'km/h')}\n"
                f"📍 Coordinates: {lat}, {lon}"
            )

        except Exception as e:
            return f"Error fetching weather: {str(e)}"
