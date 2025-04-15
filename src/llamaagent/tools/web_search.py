"""
Web search tool implementation
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from llamaagent.core.tool import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """
    Tool for searching the web
    """

    name: str = "web_search"
    description: str = "Search the web for information about a topic"
    keywords: List[str] = ["search", "google", "find", "lookup", "research"]
    requires_api_key: bool = True
    search_engine: str = "google"  # google, bing, etc.
    max_results: int = 5

    def run(self, input: str, **kwargs) -> str:
        """
        Search the web for the input query

        Args:
            input: The search query
            **kwargs: Additional parameters (num_results, search_engine)

        Returns:
            str: The search results
        """
        self._validate_api_key()

        # Override default parameters if provided
        num_results = kwargs.get("num_results", self.max_results)
        search_engine = kwargs.get("search_engine", self.search_engine)

        logger.info(f"Searching {search_engine} for: {input}")

        # Determine which search implementation to use
        if search_engine.lower() == "google":
            return self._search_google(input, num_results)
        elif search_engine.lower() == "bing":
            return self._search_bing(input, num_results)
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")

    def _search_google(self, query: str, num_results: int) -> str:
        """
        Search Google for the query
        """
        import requests

        # This is a mock implementation
        # In a real implementation, this would use the Google Search API

        try:
            # Note: This is a simplified mock of how you might use Google Custom Search API
            # In a real implementation, you would need to use the actual API with proper auth
            headers = {"Accept": "application/json", "User-Agent": "LlamaAgent/0.1"}

            # Mock response structure
            mock_results = [
                {
                    "title": f"Result {i+1} for '{query}'",
                    "link": f"https://example.com/result{i+1}",
                    "snippet": f"This is a sample result snippet for the query '{query}'. "
                    f"It would contain relevant information about {query}.",
                }
                for i in range(min(num_results, 5))
            ]

            # Format the results
            if not mock_results:
                return f"No results found for '{query}'"

            formatted_results = f"Search results for '{query}':\n\n"

            for i, result in enumerate(mock_results):
                formatted_results += f"{i+1}. {result['title']}\n"
                formatted_results += f"   URL: {result['link']}\n"
                formatted_results += f"   {result['snippet']}\n\n"

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching Google: {e}")
            return f"Error performing search: {str(e)}"

    def _search_bing(self, query: str, num_results: int) -> str:
        """
        Search Bing for the query
        """
        # This is a mock implementation
        # In a real implementation, this would use the Bing Search API

        try:
            # Mock response structure similar to Google but with Bing-specific formatting
            mock_results = [
                {
                    "name": f"Bing Result {i+1} for '{query}'",
                    "url": f"https://example.com/bing-result{i+1}",
                    "snippet": f"This is a sample Bing result for '{query}'. "
                    f"Contains information related to {query}.",
                }
                for i in range(min(num_results, 5))
            ]

            # Format the results
            if not mock_results:
                return f"No Bing results found for '{query}'"

            formatted_results = f"Bing search results for '{query}':\n\n"

            for i, result in enumerate(mock_results):
                formatted_results += f"{i+1}. {result['name']}\n"
                formatted_results += f"   URL: {result['url']}\n"
                formatted_results += f"   {result['snippet']}\n\n"

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching Bing: {e}")
            return f"Error performing Bing search: {str(e)}"
