from abc import ABC, abstractmethod
from typing import Optional

class LLMBase(ABC):
    """Base class for LLM services"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the LLM service with API key"""
        pass
    
    @abstractmethod
    def generate_bash_script(self, query: str, context: Optional[str] = None) -> tuple[str, str]:
        """Generate bash script from query
        Returns: tuple(bash_script, description)"""
        pass
    
    @abstractmethod
    def interpret_output(self, original_query: str, command_output: str) -> str:
        """Interpret command output in natural language"""
        pass