"""Tools and function calling implementation for OpenAI Agents SDK Demo"""

import json
import asyncio
import inspect
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Tool:
    """Represents a tool that can be used by the agent"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    
    def __post_init__(self):
        """Validate tool after initialization"""
        if not callable(self.function):
            raise ValueError(f"Tool '{self.name}' function must be callable")
        
        # Validate parameters schema
        if not isinstance(self.parameters, dict):
            raise ValueError(f"Tool '{self.name}' parameters must be a dictionary")
    
    @classmethod
    def from_function(cls, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'Tool':
        """Create a Tool from a function with automatic parameter extraction"""
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {tool_name}"
        
        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list:
                    param_info["type"] = "array"
                elif param.annotation == dict:
                    param_info["type"] = "object"
            
            parameters["properties"][param_name] = param_info
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return cls(
            name=tool_name,
            description=tool_description,
            function=func,
            parameters=parameters
        )
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments"""
        try:
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{self.name}': {str(e)}")


class ToolRegistry:
    """Registry for managing agent tools"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool"""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def register_function(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> Tool:
        """Register a function as a tool"""
        tool = Tool.from_function(func, name, description)
        self.register(tool)
        return tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool"""
        if name in self._tools:
            del self._tools[name]
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self._tools.keys())
    
    def clear(self) -> None:
        """Clear all tools"""
        self._tools.clear()
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Built-in tools for common use cases
class BuiltinTools:
    """Collection of built-in tools for common agent tasks"""
    
    @staticmethod
    def get_current_time() -> str:
        """Get the current date and time"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression"""
        try:
            # Only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def search_web(query: str, max_results: int = 5) -> str:
        """Simulate web search (placeholder implementation)"""
        return f"Web search results for '{query}' (simulated - {max_results} results)"
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """Read content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    @staticmethod
    def make_http_request(url: str, method: str = "GET", data: Optional[str] = None) -> str:
        """Make an HTTP request"""
        try:
            import requests
            
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, data=data)
            else:
                return f"Error: Unsupported HTTP method {method}"
            
            return f"Status: {response.status_code}\nContent: {response.text[:500]}..."
        except Exception as e:
            return f"Error making HTTP request: {str(e)}"
    
    @classmethod
    def register_all(cls, registry: ToolRegistry) -> None:
        """Register all built-in tools to a registry"""
        tools = [
            (cls.get_current_time, "get_current_time", "Get the current date and time"),
            (cls.calculate, "calculate", "Safely evaluate a mathematical expression"),
            (cls.search_web, "search_web", "Search the web for information"),
            (cls.read_file, "read_file", "Read content from a file"),
            (cls.write_file, "write_file", "Write content to a file"),
            (cls.make_http_request, "make_http_request", "Make an HTTP request")
        ]
        
        for func, name, description in tools:
            registry.register_function(func, name, description)


# Decorator for easy tool creation
def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to mark a function as a tool"""
    def decorator(func: Callable) -> Callable:
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or f"Execute {func.__name__}"
        func._is_tool = True
        return func
    return decorator


def extract_tools_from_module(module) -> List[Tool]:
    """Extract all tools from a module (functions decorated with @tool)"""
    tools = []
    
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, '_is_tool'):
            tool = Tool.from_function(
                obj, 
                getattr(obj, '_tool_name', name),
                getattr(obj, '_tool_description', obj.__doc__)
            )
            tools.append(tool)
    
    return tools