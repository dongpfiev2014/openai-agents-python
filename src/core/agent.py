"""Core Agent implementation for OpenAI Agents SDK Demo"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .tools import ToolRegistry, Tool
from .memory import Memory, ConversationMemory


@dataclass
class AgentConfig:
    """Configuration for OpenAI Agent"""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = "You are a helpful AI assistant."
    tools_enabled: bool = True
    memory_enabled: bool = True
    streaming: bool = False
    max_conversation_history: int = 50
    

class Message(BaseModel):
    """Message model for agent conversations"""
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Agent:
    """Main Agent class implementing OpenAI Agents SDK functionality"""
    
    def __init__(self, config: AgentConfig, api_key: str, org_id: Optional[str] = None):
        self.config = config
        self.client = OpenAI(api_key=api_key, organization=org_id)
        self.async_client = AsyncOpenAI(api_key=api_key, organization=org_id)
        
        # Initialize components
        self.tool_registry = ToolRegistry() if config.tools_enabled else None
        self.memory = ConversationMemory(max_history=config.max_conversation_history) if config.memory_enabled else None
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Conversation state
        self.conversation_id: Optional[str] = None
        
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit"""
        if self.tool_registry:
            self.tool_registry.register(tool)
        else:
            raise ValueError("Tools are not enabled for this agent")
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt"""
        self.config.system_prompt = prompt
        
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            
    def get_conversation_history(self) -> List[Message]:
        """Get current conversation history"""
        if self.memory:
            return [Message(**msg) for msg in self.memory.get_history()]
        return []
    
    def _prepare_messages(self, user_message: str, conversation_id: Optional[str] = None) -> List[Dict]:
        """Prepare messages for OpenAI API call"""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        
        # Add conversation history if memory is enabled
        if self.memory and conversation_id:
            history = self.memory.get_history(conversation_id)
            messages.extend(history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _prepare_tools(self) -> Optional[List[Dict]]:
        """Prepare tools for OpenAI API call"""
        if not self.tool_registry:
            return None
            
        tools = []
        for tool in self.tool_registry.get_all_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return tools if tools else None
    
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results"""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            try:
                tool = self.tool_registry.get_tool(function_name)
                if tool:
                    if asyncio.iscoroutinefunction(tool.function):
                        result = await tool.function(**function_args)
                    else:
                        result = tool.function(**function_args)
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": str(result)
                    })
                else:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: Tool '{function_name}' not found"
                    })
            except Exception as e:
                self.logger.error(f"Error executing tool {function_name}: {e}")
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        
        return results
    
    def chat(self, message: str, conversation_id: Optional[str] = None) -> str:
        """Synchronous chat with the agent"""
        messages = self._prepare_messages(message, conversation_id)
        tools = self._prepare_tools()
        
        # Store user message in memory
        if self.memory:
            self.memory.add_message(conversation_id or "default", "user", message)
        
        try:
            # Make API call
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            
            message_obj = response.choices[0].message
            
            # Handle tool calls
            if message_obj.tool_calls:
                # Add assistant message with tool calls to memory
                if self.memory:
                    self.memory.add_message(
                        conversation_id or "default", 
                        "assistant", 
                        message_obj.content or "",
                        tool_calls=[tc.model_dump() for tc in message_obj.tool_calls]
                    )
                
                # Execute tools synchronously (convert async to sync)
                # Use event loop creation instead of asyncio.run() for Streamlit compatibility
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tool_results = loop.run_until_complete(self._execute_tool_calls(
                        [tc.model_dump() for tc in message_obj.tool_calls]
                    ))
                finally:
                    loop.close()
                
                # Add tool results to memory
                if self.memory:
                    for result in tool_results:
                        self.memory.add_message(
                            conversation_id or "default",
                            "tool",
                            result["content"],
                            tool_call_id=result["tool_call_id"],
                            name=result["name"]
                        )
                
                # Make another API call with tool results
                messages.append(message_obj.model_dump())
                messages.extend(tool_results)
                
                final_response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                final_content = final_response.choices[0].message.content
            else:
                final_content = message_obj.content
            
            # Store assistant response in memory
            if self.memory:
                self.memory.add_message(conversation_id or "default", "assistant", final_content)
            
            return final_content
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            # Ensure we always return a string, never a coroutine
            error_msg = str(e)
            if "asyncio.Future" in error_msg or "coroutine" in error_msg:
                error_msg = "API connection error. Please check your API key and try again."
            return f"Error: {error_msg}"
    
    async def chat_async(self, message: str, conversation_id: Optional[str] = None) -> str:
        """Asynchronous chat with the agent"""
        messages = self._prepare_messages(message, conversation_id)
        tools = self._prepare_tools()
        
        # Store user message in memory
        if self.memory:
            self.memory.add_message(conversation_id or "default", "user", message)
        
        try:
            # Make API call
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.async_client.chat.completions.create(**kwargs)
            
            message_obj = response.choices[0].message
            
            # Handle tool calls
            if message_obj.tool_calls:
                # Add assistant message with tool calls to memory
                if self.memory:
                    self.memory.add_message(
                        conversation_id or "default", 
                        "assistant", 
                        message_obj.content or "",
                        tool_calls=[tc.model_dump() for tc in message_obj.tool_calls]
                    )
                
                # Execute tools
                tool_results = await self._execute_tool_calls(
                    [tc.model_dump() for tc in message_obj.tool_calls]
                )
                
                # Add tool results to memory
                if self.memory:
                    for result in tool_results:
                        self.memory.add_message(
                            conversation_id or "default",
                            "tool",
                            result["content"],
                            tool_call_id=result["tool_call_id"],
                            name=result["name"]
                        )
                
                # Make another API call with tool results
                messages.append(message_obj.model_dump())
                messages.extend(tool_results)
                
                final_response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                final_content = final_response.choices[0].message.content
            else:
                final_content = message_obj.content
            
            # Store assistant response in memory
            if self.memory:
                self.memory.add_message(conversation_id or "default", "assistant", final_content)
            
            return final_content
            
        except Exception as e:
            self.logger.error(f"Error in async chat: {e}")
            # Ensure we always return a string, never a coroutine
            error_msg = str(e)
            if "asyncio.Future" in error_msg or "coroutine" in error_msg:
                error_msg = "API connection error. Please check your API key and try again."
            return f"Error: {error_msg}"
    
    async def chat_stream(self, message: str, conversation_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream chat responses from the agent"""
        messages = self._prepare_messages(message, conversation_id)
        tools = self._prepare_tools()
        
        # Store user message in memory
        if self.memory:
            self.memory.add_message(conversation_id or "default", "user", message)
        
        try:
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            full_response = ""
            
            async for chunk in await self.async_client.chat.completions.create(**kwargs):
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Store assistant response in memory
            if self.memory:
                self.memory.add_message(conversation_id or "default", "assistant", full_response)
                
        except Exception as e:
            self.logger.error(f"Error in stream chat: {e}")
            yield f"Error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "model": self.config.model,
            "tools_enabled": self.config.tools_enabled,
            "memory_enabled": self.config.memory_enabled,
            "tools_count": len(self.tool_registry.get_all_tools()) if self.tool_registry else 0,
        }
        
        if self.memory:
            stats["conversations"] = len(self.memory.conversations)
            stats["total_messages"] = sum(len(conv) for conv in self.memory.conversations.values())
        
        return stats