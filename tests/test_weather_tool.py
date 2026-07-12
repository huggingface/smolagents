import unittest
from unittest.mock import MagicMock, patch
from smolagents.weather_tool import OpenMeteoTool

class TestWeatherTool(unittest.TestCase):
    @patch('requests.get')
    def test_weather_lookup(self, mock_get):
        """
        Test weather lookup with mocked API responses.
        """
        # 1. Mock Geocoding Response (Pune)
        mock_geo_resp = MagicMock()
        mock_geo_resp.json.return_value = {
            "results": [{"name": "Pune", "country": "India", "latitude": 18.5, "longitude": 73.8}]
        }

        # 2. Mock Weather Response
        mock_weather_resp = MagicMock()
        mock_weather_resp.json.return_value = {
            "current": {
                "temperature_2m": 28.5,
                "relative_humidity_2m": 60,
                "wind_speed_10m": 12.0
            },
            "current_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h"
            }
        }

        # Configure the side_effect to return geo first, then weather
        mock_get.side_effect = [mock_geo_resp, mock_weather_resp]

        # 3. Run Tool
        tool = OpenMeteoTool()
        result = tool.forward("Pune")

        # 4. Verify
        self.assertIn("Pune, India", result)
        self.assertIn("28.5°C", result)
        self.assertIn("60%", result)

    @patch('requests.get')
    def test_city_not_found(self, mock_get):
        """Test graceful failure for unknown cities."""
        mock_get.return_value.json.return_value = {"results": []}
        
        tool = OpenMeteoTool()
        result = tool.forward("Atlantis_Fake_City")
        self.assertIn("Could not find coordinates", result)

if __name__ == "__main__":
    unittest.main()
