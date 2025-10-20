# =====================================
# src/concierge/request_handler.py
# =====================================
"""
Request handler for processing different types of user requests
"""
from typing import Dict, Any, Optional, Tuple
import re


class RequestHandler:
    """Handles and categorizes user requests"""

    def __init__(self):
        self.request_patterns = {
            "config_update": [
                r"update\s+(?:my\s+)?(\w+)",
                r"change\s+(?:my\s+)?(\w+)",
                r"set\s+(?:my\s+)?(\w+)"
            ],
            "query": [
                r"what\s+is",
                r"show\s+me",
                r"tell\s+me",
                r"current\s+status"
            ],
            "analysis": [
                r"analyze",
                r"flexibility",
                r"optimize",
                r"forecast"
            ]
        }

    def parse_request(self, request: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse user request to extract intent and entities

        Returns:
            Tuple of (intent, entities)
        """
        request_lower = request.lower()

        # Check patterns
        for intent, patterns in self.request_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    entities = self._extract_entities(request, intent)
                    return intent, entities

        return "general", {"request": request}

    def _extract_entities(self, request: str, intent: str) -> Dict[str, Any]:
        """Extract entities from request based on intent"""
        entities = {"original_request": request}

        if intent == "config_update":
            # Extract component and values
            # Simplified extraction
            entities["component"] = None
            entities["value"] = None

        return entities