import asyncio
import inspect
from src.core.agent import Agent, AgentConfig
from src.core.tools import Tool
from datetime import datetime

def get_current_time() -> str:
    """Get the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def create_demo_tools():
    """Create demo tools for testing"""
    tools = [
        Tool.from_function(get_current_time, "get_current_time", "Get the current time"),
        Tool.from_function(calculate, "calculate", "Calculate a mathematical expression")
    ]
    return tools

def debug_agent_response():
    """Debug what agent.chat() returns when there's an API error"""
    print("=== Debugging Agent Response Type ===")
    
    # Create tools and agent
    tools = create_demo_tools()
    config = AgentConfig(
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant.",
        tools_enabled=True
    )
    agent = Agent(config, api_key="invalid_key_for_testing")
    
    print(f"Config tools_enabled: {config.tools_enabled}")
    print(f"Agent tool_registry: {agent.tool_registry}")
    
    # Register tools
    for tool in tools:
        agent.tool_registry.register(tool)
    
    print("Agent created and tools registered successfully")
    
    # Test chat with invalid API key
    try:
        print("\nTesting agent.chat() with invalid API key...")
        result = agent.chat("Hello")
        
        print(f"Result type: {type(result)}")
        print(f"Result value: {result}")
        print(f"Is coroutine: {inspect.iscoroutine(result)}")
        print(f"Is awaitable: {inspect.isawaitable(result)}")
        print(f"Is string: {isinstance(result, str)}")
        
        # If it's a coroutine, try to run it
        if inspect.iscoroutine(result):
            print("\nResult is a coroutine, running with asyncio...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                final_result = loop.run_until_complete(result)
                print(f"Final result: {final_result}")
            except Exception as e:
                print(f"Error running coroutine: {e}")
            finally:
                loop.close()
        
    except Exception as e:
        print(f"Exception during agent.chat(): {e}")
        print(f"Exception type: {type(e)}")

if __name__ == "__main__":
    debug_agent_response()