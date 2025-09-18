"""Function calling example demonstrating tool usage with OpenAI Agent"""

import asyncio
import json
import random
from datetime import datetime
from typing import Optional, Dict, Any

from core.agent import Agent, AgentConfig
from core.tools import Tool, ToolRegistry, BuiltinTools, tool
from core.utils import load_config, setup_logging


# Custom tools for demonstration
@tool(name="get_weather", description="Get current weather information for a city")
def get_weather(city: str, units: str = "celsius") -> str:
    """Simulate getting weather information"""
    # Simulate weather data
    temperatures = {"celsius": random.randint(-10, 35), "fahrenheit": random.randint(14, 95)}
    conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
    
    temp = temperatures.get(units, temperatures["celsius"])
    condition = random.choice(conditions)
    
    return json.dumps({
        "city": city,
        "temperature": temp,
        "units": units,
        "condition": condition,
        "humidity": random.randint(30, 90),
        "timestamp": datetime.now().isoformat()
    })


@tool(name="create_task", description="Create a new task in the task management system")
def create_task(title: str, description: str = "", priority: str = "medium", due_date: str = "") -> str:
    """Create a new task"""
    task_id = f"task_{random.randint(1000, 9999)}"
    
    task = {
        "id": task_id,
        "title": title,
        "description": description,
        "priority": priority,
        "due_date": due_date,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    return json.dumps(task)


@tool(name="search_knowledge_base", description="Search the knowledge base for information")
def search_knowledge_base(query: str, category: str = "general") -> str:
    """Simulate searching a knowledge base"""
    # Simulate search results
    results = [
        {
            "title": f"Article about {query}",
            "summary": f"This article covers important aspects of {query} in the {category} category.",
            "relevance_score": random.uniform(0.7, 1.0),
            "url": f"https://kb.example.com/articles/{query.replace(' ', '-').lower()}"
        },
        {
            "title": f"Guide to {query}",
            "summary": f"A comprehensive guide explaining {query} concepts and best practices.",
            "relevance_score": random.uniform(0.6, 0.9),
            "url": f"https://kb.example.com/guides/{query.replace(' ', '-').lower()}"
        }
    ]
    
    return json.dumps({
        "query": query,
        "category": category,
        "results": results,
        "total_results": len(results)
    })


@tool(name="send_notification", description="Send a notification to the user")
def send_notification(message: str, notification_type: str = "info", recipient: str = "user") -> str:
    """Simulate sending a notification"""
    notification = {
        "id": f"notif_{random.randint(1000, 9999)}",
        "message": message,
        "type": notification_type,
        "recipient": recipient,
        "sent_at": datetime.now().isoformat(),
        "status": "delivered"
    }
    
    return json.dumps(notification)


class FunctionCallingExample:
    """Demonstrates function calling capabilities with OpenAI Agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        # Load configuration
        config_data = load_config()
        
        # Use provided API key or load from config
        self.api_key = api_key or config_data.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Create agent configuration with tools enabled
        self.agent_config = AgentConfig(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful AI assistant with access to various tools. Use the available tools to help users accomplish their tasks effectively.",
            tools_enabled=True,
            memory_enabled=True,
            streaming=False
        )
        
        # Initialize agent
        self.agent = Agent(
            config=self.agent_config,
            api_key=self.api_key,
            org_id=config_data.get("openai_org_id")
        )
        
        # Setup tools
        self._setup_tools()
        
        print(f"✅ Function Calling Agent initialized with model: {model}")
        print(f"🔧 Available tools: {len(self.agent.tool_registry.get_all_tools())}")
    
    def _setup_tools(self) -> None:
        """Setup tools for the agent"""
        # Add built-in tools
        BuiltinTools.register_all(self.agent.tool_registry)
        
        # Add custom tools
        custom_tools = [
            Tool.from_function(get_weather),
            Tool.from_function(create_task),
            Tool.from_function(search_knowledge_base),
            Tool.from_function(send_notification)
        ]
        
        for tool in custom_tools:
            self.agent.add_tool(tool)
    
    def list_available_tools(self) -> None:
        """Display all available tools"""
        print("\n🔧 Available Tools:")
        print("-" * 40)
        
        tools = self.agent.tool_registry.get_all_tools()
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
            print(f"   Description: {tool.description}")
            print(f"   Parameters: {list(tool.parameters.get('properties', {}).keys())}")
            print()
    
    def demo_weather_tool(self) -> None:
        """Demonstrate weather tool usage"""
        print("\n🌤️ Weather Tool Demo")
        print("-" * 30)
        
        messages = [
            "What's the weather like in New York?",
            "Can you check the weather in Tokyo and London?",
            "Get me the weather in Paris in Fahrenheit"
        ]
        
        for message in messages:
            print(f"\n👤 User: {message}")
            response = self.agent.chat(message, "weather_demo")
            print(f"🤖 Assistant: {response}")
    
    def demo_task_management(self) -> None:
        """Demonstrate task management tools"""
        print("\n📋 Task Management Demo")
        print("-" * 30)
        
        messages = [
            "Create a task to review the quarterly report with high priority",
            "Add a task for team meeting preparation due tomorrow",
            "Create a low priority task to update documentation"
        ]
        
        for message in messages:
            print(f"\n👤 User: {message}")
            response = self.agent.chat(message, "task_demo")
            print(f"🤖 Assistant: {response}")
    
    def demo_knowledge_search(self) -> None:
        """Demonstrate knowledge base search"""
        print("\n🔍 Knowledge Base Search Demo")
        print("-" * 35)
        
        messages = [
            "Search for information about machine learning",
            "Find articles about Python programming in the development category",
            "Look up best practices for API design"
        ]
        
        for message in messages:
            print(f"\n👤 User: {message}")
            response = self.agent.chat(message, "search_demo")
            print(f"🤖 Assistant: {response}")
    
    def demo_multi_tool_usage(self) -> None:
        """Demonstrate using multiple tools in one conversation"""
        print("\n🔄 Multi-Tool Usage Demo")
        print("-" * 30)
        
        messages = [
            "Check the weather in San Francisco, then create a task to pack for the trip",
            "Search for information about travel tips, then send me a notification with a summary",
            "What's the current time, calculate 15 * 24, and create a task with the result"
        ]
        
        for message in messages:
            print(f"\n👤 User: {message}")
            response = self.agent.chat(message, "multi_tool_demo")
            print(f"🤖 Assistant: {response}")
    
    async def demo_async_function_calling(self) -> None:
        """Demonstrate asynchronous function calling"""
        print("\n⚡ Async Function Calling Demo")
        print("-" * 35)
        
        messages = [
            "Get weather for multiple cities: New York, London, Tokyo",
            "Create three tasks: morning standup, code review, and documentation update",
            "Search for AI, machine learning, and deep learning topics"
        ]
        
        tasks = []
        for i, message in enumerate(messages):
            task = self.agent.chat_async(message, f"async_demo_{i}")
            tasks.append(task)
        
        print("\n🔄 Processing requests asynchronously...")
        responses = await asyncio.gather(*tasks)
        
        for i, (message, response) in enumerate(zip(messages, responses)):
            print(f"\n👤 User {i+1}: {message}")
            print(f"🤖 Assistant {i+1}: {response}")
    
    def interactive_tool_chat(self) -> None:
        """Interactive chat with tool capabilities"""
        print("\n🤖 Function Calling Demo - Interactive Mode")
        print("Available commands:")
        print("  - 'tools' to list available tools")
        print("  - 'clear' to clear conversation history")
        print("  - 'quit' to exit")
        print("-" * 50)
        
        conversation_id = "interactive_tools"
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'tools':
                    self.list_available_tools()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.agent.clear_memory()
                    print("\n🧹 Conversation history cleared.")
                    continue
                
                # Get response from agent (may use tools)
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.agent.chat(user_input, conversation_id)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def run_all_demos(self) -> None:
        """Run all demonstration scenarios"""
        print("\n🚀 Running All Function Calling Demos")
        print("=" * 50)
        
        # List available tools
        self.list_available_tools()
        
        # Run individual demos
        self.demo_weather_tool()
        self.demo_task_management()
        self.demo_knowledge_search()
        self.demo_multi_tool_usage()
        
        # Show agent statistics
        stats = self.agent.get_stats()
        print("\n📊 Agent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Main function to run the function calling example"""
    # Setup logging
    setup_logging("INFO")
    
    try:
        # Initialize the function calling example
        func_example = FunctionCallingExample()
        
        print("\n🚀 OpenAI Agents SDK - Function Calling Demo")
        print("=" * 50)
        
        # Run all demos
        func_example.run_all_demos()
        
        # Run async demo
        print("\n" + "=" * 50)
        asyncio.run(func_example.demo_async_function_calling())
        
        # Start interactive chat
        print("\n" + "=" * 50)
        func_example.interactive_tool_chat()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 To fix this:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to the .env file")
        print("3. Run the example again")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


if __name__ == "__main__":
    main()