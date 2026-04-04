import httpx
from typing import Dict, Any, Optional

class CustomerSupportClient:
    """Client for interacting with the Customer Support Email Triage environment."""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str = "easy") -> Dict[str, Any]:
        """Reset the environment for a specific task."""
        response = httpx.post(f"{self.base_url}/api/reset?task={task_name}")
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take a step in the environment."""
        response = httpx.post(f"{self.base_url}/api/step", json=action)
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Check the health of the server."""
        response = httpx.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()
