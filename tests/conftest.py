import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_mcp():
    """Mocks the FastMCP object to extract the registered tool functions for direct testing."""
    mcp = MagicMock()
    
    def tool_decorator(*args, **kwargs):
        def wrapper(func):
            # Attach the original async function to the mock object for easy access
            setattr(mcp, f"tool_{func.__name__}", func)
            return func
        return wrapper
    
    mcp.tool = tool_decorator
    
    # Also mock the prompt and resource decorators for testing main.py
    def generic_decorator(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
        
    mcp.prompt = generic_decorator
    mcp.resource = generic_decorator
    
    return mcp
