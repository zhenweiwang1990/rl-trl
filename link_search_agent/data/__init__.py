"""Data loading and types for Link Search Agent."""

from link_search_agent.data.types import LinkSearchQuery, ProfileDetail
from link_search_agent.data.query_loader import load_link_search_queries

__all__ = [
    "LinkSearchQuery",
    "ProfileDetail",
    "load_link_search_queries",
]
