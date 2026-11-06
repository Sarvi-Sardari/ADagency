"""Compatibility import for the shared vLLM lifecycle manager."""

from cluster_deployment.server_manager import ServerManager, managed_server_lifecycle

__all__ = ["ServerManager", "managed_server_lifecycle"]
