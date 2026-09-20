"""
MCP tools and intent detection for the Singapore Travel Assistant.
This file combines weather lookups, currency conversion, and tool selection.
"""

# Standard library imports
import re
from typing import Any, Dict

# Third-party imports
import requests


# ---------------------------------------------------------------------------
# MCP tool manager
# ---------------------------------------------------------------------------
class MCPTools:
    """Manage weather, currency conversion, and query intent detection."""

    def __init__(self):
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        self.currency_url = "https://open.er-api.com/v6/latest"
        self.singapore = {"latitude": 1.3521, "longitude": 103.8198}
        self.timeout = 10
        self.fallback_rates = {
            "INR": {"SGD": 0.013322, "USD": 0.010423, "EUR": 0.00908, "GBP": 0.00783, "JPY": 1.55},
            "SGD": {"INR": 75.07, "USD": 0.78, "EUR": 0.68, "GBP": 0.59, "JPY": 116.5},
            "USD": {"INR": 83.3, "SGD": 1.28, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5},
            "EUR": {"INR": 90.6, "SGD": 1.39, "USD": 1.09, "GBP": 0.86, "JPY": 161.9},
        }

    # ---------------------------------------------------------------------
    # Intent detection
    # ---------------------------------------------------------------------
    def detect_tool_needs(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query and decide which tools are required.
        Returns a dictionary with weather and currency requirements,
        plus the query type and any extracted parameters.
        """
        query_lower = query.lower()

        # Detect weather-related intent
        weather_keywords = {
            "weather",
            "forecast",
            "rain",
            "rainy",
            "sunny",
            "temperature",
            "temp",
            "humidity",
            "monsoon",
            "climate",
            "conditions",
            "wet",
            "dry",
            "hot",
            "cold",
            "windy",
            "thunderstorm",
            "afternoon shower",
            "rainfall",
            "umbrella",
            "indoor",
            "outdoor",
        }
        needs_weather = any(keyword in query_lower for keyword in weather_keywords)

        # Detect currency-related intent
        currency_keywords = {
            "convert",
            "currency",
            "budget",
            "cost",
            "price",
            "how much",
            "inr",
            "sgd",
            "usd",
            "eur",
            "rupee",
            "dollar",
            "exchange",
        }
        needs_currency = any(keyword in query_lower for keyword in currency_keywords)

        # Extract currency parameters when conversion is requested.
        # Use query order instead of a set so conversions like "INR to SGD"
        # do not get reversed by hash-order iteration.
        currency_params = {}
        if needs_currency:
            amount_match = re.search(r"(\d+(?:[,\.]\d+)?)", query)
            if amount_match:
                currency_params["amount"] = float(amount_match.group(1).replace(",", ""))

            supported_codes = ["inr", "sgd", "usd", "eur", "gbp", "jpy", "aud", "cad"]
            found_codes = []
            for code in supported_codes:
                if code in query_lower:
                    found_codes.append(code)

            if len(found_codes) >= 2:
                # Keep the exact order of appearance in the query to avoid flipping
                # INR -> SGD into SGD -> INR when both currencies are present.
                first = None
                second = None
                for code in supported_codes:
                    if code in query_lower:
                        if first is None:
                            first = code
                        elif second is None:
                            second = code
                            break
                currency_params["from"] = (first or found_codes[0]).upper()
                currency_params["to"] = (second or found_codes[1]).upper()
            elif len(found_codes) == 1:
                if found_codes[0] != "sgd":
                    currency_params["from"] = found_codes[0].upper()
                    currency_params["to"] = "SGD"
                else:
                    currency_params["from"] = "SGD"
                    currency_params["to"] = "INR"

        # Detect whether the query needs the travel knowledge base
        kb_keywords = {
            "attraction",
            "visit",
            "see",
            "do",
            "activity",
            "itinerary",
            "plan",
            "recommend",
            "where",
            "what",
            "how",
            "transport",
            "accommodation",
            "restaurant",
            "neighborhood",
            "temple",
            "museum",
            "family",
            "outdoor",
            "indoor",
            "cultural",
            "heritage",
        }
        needs_kb = any(keyword in query_lower for keyword in kb_keywords)

        # Determine the overall query type
        if needs_kb and (needs_weather or needs_currency):
            query_type = "combined"
        elif needs_weather or needs_currency:
            query_type = "tools_only"
        else:
            query_type = "destination"

        return {
            "needs_weather": needs_weather,
            "needs_currency": needs_currency,
            "needs_kb": needs_kb,
            "query_type": query_type,
            "currency_params": currency_params,
            "tools_to_call": [
                tool
                for tool in ["weather", "currency"]
                if (tool == "weather" and needs_weather)
                or (tool == "currency" and needs_currency)
            ],
        }

    # ---------------------------------------------------------------------
    # Tool execution
    # ---------------------------------------------------------------------
    def get_weather(self, days: int = 3) -> Dict[str, Any]:
        """Fetch a short Singapore weather forecast from Open-Meteo."""
        try:
            params = {
                "latitude": self.singapore["latitude"],
                "longitude": self.singapore["longitude"],
                "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min,weather_code",
                "timezone": "Asia/Singapore",
                "forecast_days": min(days, 16),
            }

            response = requests.get(self.weather_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            daily = data.get("daily", {})

            forecast = []
            for i in range(min(len(daily.get("time", [])), days)):
                forecast.append(
                    {
                        "date": daily["time"][i],
                        "max_temp": daily["temperature_2m_max"][i],
                        "min_temp": daily["temperature_2m_min"][i],
                        "rain_prob": daily["precipitation_probability_max"][i],
                    }
                )

            return {"status": "success", "data": forecast}
        except Exception as e:
            return {"status": "error", "message": f"Weather unavailable: {str(e)}"}

    def convert_currency(self, amount: float, from_curr: str, to_curr: str) -> Dict[str, Any]:
        """Convert one currency to another using a reliable public API with a fallback map."""
        try:
            from_curr = from_curr.upper()
            to_curr = to_curr.upper()

            if amount <= 0:
                return {"status": "error", "message": "Amount must be greater than zero."}

            response = requests.get(f"{self.currency_url}/{from_curr}", timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            rate = data.get("rates", {}).get(to_curr)

            if rate is None or rate <= 0 or rate > 1000:
                fallback_rate = self.fallback_rates.get(from_curr, {}).get(to_curr)
                if fallback_rate is None:
                    return {"status": "error", "message": f"Cannot convert {from_curr} to {to_curr}"}
                rate = fallback_rate

            return {
                "status": "success",
                "data": {
                    "amount": amount,
                    "from": from_curr,
                    "to": to_curr,
                    "rate": rate,
                    "converted": amount * rate,
                },
            }
        except Exception as e:
            return {"status": "error", "message": f"Conversion failed: {str(e)}"}


# Create a shared instance for the application
mcp = MCPTools()