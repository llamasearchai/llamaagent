"""
Web search tool for retrieving information from the internet.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import requests

from .base import BaseTool

logger = logging.getLogger(__name__)


class WebSearch(BaseTool):
    """
    Tool for searching the web and retrieving information.

    This tool uses search APIs to find relevant information on the web.
    It supports multiple search providers and can return structured results.
    """

    def __init__(
        self,
        provider: str = "google",
        api_key: Optional[str] = None,
        num_results: int = 5,
        include_snippets: bool = True,
        include_links: bool = True,
        safe_search: bool = True,
        **kwargs,
    ):
        """
        Initialize the web search tool.

        Args:
            provider: The search provider to use ('google', 'bing', 'serper', etc.)
            api_key: API key for the search provider (falls back to env vars if None)
            num_results: Number of results to return
            include_snippets: Whether to include snippets in the results
            include_links: Whether to include links in the results
            safe_search: Whether to enable safe search
        """
        super().__init__(
            name="WebSearch", description="Search the web for information", **kwargs
        )

        self.provider = provider.lower()
        self.num_results = num_results
        self.include_snippets = include_snippets
        self.include_links = include_links
        self.safe_search = safe_search

        # Set API key from args or environment variables
        self.api_key = api_key or self._get_api_key_from_env()

        if not self.api_key:
            logger.warning(
                f"No API key provided for {provider} search. Some providers may not work."
            )

        logger.debug(f"Initialized WebSearch tool with provider: {provider}")

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables based on provider."""
        env_var_map = {
            "google": "GOOGLE_SEARCH_API_KEY",
            "bing": "BING_SEARCH_API_KEY",
            "serper": "SERPER_API_KEY",
            "serpapi": "SERPAPI_API_KEY",
        }

        env_var = env_var_map.get(self.provider)
        return os.environ.get(env_var) if env_var else None

    def _run(self, query: str, **kwargs) -> str:
        """
        Search the web for the given query.

        Args:
            query: The search query
            **kwargs: Additional provider-specific parameters

        Returns:
            Formatted search results as a string
        """
        logger.info(f"Searching for: {query}")

        # Merge instance settings with any overrides
        params = {
            "num_results": kwargs.get("num_results", self.num_results),
            "include_snippets": kwargs.get("include_snippets", self.include_snippets),
            "include_links": kwargs.get("include_links", self.include_links),
            "safe_search": kwargs.get("safe_search", self.safe_search),
        }

        # Call the appropriate search provider
        if self.provider == "google":
            results = self._search_google(query, **params)
        elif self.provider == "bing":
            results = self._search_bing(query, **params)
        elif self.provider == "serper":
            results = self._search_serper(query, **params)
        else:
            raise ValueError(f"Unsupported search provider: {self.provider}")

        # Format results
        return self._format_results(results, **params)

    def _search_google(self, query: str, **params) -> List[Dict[str, Any]]:
        """
        Search using Google Programmable Search Engine.

        Args:
            query: The search query
            **params: Additional parameters

        Returns:
            List of search result items
        """
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"

        # Prepare request parameters
        request_params = {
            "q": query,
            "key": self.api_key,
            "cx": os.environ.get("GOOGLE_CSE_ID"),  # Custom Search Engine ID
            "num": params.get("num_results", 5),
            "safe": "active" if params.get("safe_search", True) else "off",
        }

        logger.debug(f"Sending request to Google Search API: {query}")

        try:
            response = requests.get(url, params=request_params)
            response.raise_for_status()

            data = response.json()

            # Extract and format results
            items = data.get("items", [])
            results = []

            for item in items:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "Google",
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error calling Google Search API: {e}")
            return []

    def _search_bing(self, query: str, **params) -> List[Dict[str, Any]]:
        """
        Search using Bing Search API.

        Args:
            query: The search query
            **params: Additional parameters

        Returns:
            List of search result items
        """
        # Bing Search API endpoint
        url = "https://api.bing.microsoft.com/v7.0/search"

        # Prepare headers and parameters
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}

        request_params = {
            "q": query,
            "count": params.get("num_results", 5),
            "safeSearch": "strict" if params.get("safe_search", True) else "off",
        }

        logger.debug(f"Sending request to Bing Search API: {query}")

        try:
            response = requests.get(url, headers=headers, params=request_params)
            response.raise_for_status()

            data = response.json()

            # Extract and format results
            web_pages = data.get("webPages", {}).get("value", [])
            results = []

            for page in web_pages:
                result = {
                    "title": page.get("name", ""),
                    "link": page.get("url", ""),
                    "snippet": page.get("snippet", ""),
                    "source": "Bing",
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error calling Bing Search API: {e}")
            return []

    def _search_serper(self, query: str, **params) -> List[Dict[str, Any]]:
        """
        Search using Serper.dev API.

        Args:
            query: The search query
            **params: Additional parameters

        Returns:
            List of search result items
        """
        # Serper.dev API endpoint
        url = "https://google.serper.dev/search"

        # Prepare headers and parameters
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        payload = {
            "q": query,
            "num": params.get("num_results", 5),
            "gl": "us",  # Geolocation (United States)
            "hl": "en",  # Language (English)
        }

        logger.debug(f"Sending request to Serper.dev API: {query}")

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract and format results
            organic = data.get("organic", [])
            results = []

            for item in organic:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "Serper.dev",
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error calling Serper.dev API: {e}")
            return []

    def _format_results(self, results: List[Dict[str, Any]], **params) -> str:
        """
        Format search results as a readable string.

        Args:
            results: List of search result items
            **params: Formatting parameters

        Returns:
            Formatted results as a string
        """
        if not results:
            return "No results found."

        include_snippets = params.get("include_snippets", True)
        include_links = params.get("include_links", True)

        formatted_results = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")

            parts = [f"{i}. {title}"]

            if include_links:
                parts.append(f"   URL: {link}")

            if include_snippets and snippet:
                parts.append(f"   {snippet}")

            formatted_results.append("\n".join(parts))

        output = "\n\n".join(formatted_results)

        # Add a header
        header = f"Search results for: '{results[0].get('query', '')}'\n"
        source = f"Source: {results[0].get('source', 'Unknown')}\n"

        return f"{header}{source}\n{output}"
