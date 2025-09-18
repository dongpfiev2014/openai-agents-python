"""Basic chat example demonstrating core OpenAI Agent functionality"""

import asyncio
import os
from typing import Optional

from core.agent import Agent, AgentConfig
from core.utils import load_config, setup_logging


class BasicChatExample:
    """Demonstrates basic chat functionality with OpenAI Agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        # Load configuration
        config_data = load_config()
        
        # Use provided API key or load from config
        self.api_key = api_key or config_data.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Create agent configuration
        self.agent_config = AgentConfig(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful AI assistant. Provide clear, concise, and accurate responses.",
            tools_enabled=False,  # Basic chat doesn't use tools
            memory_enabled=True,
            streaming=False
        )
        
        # Initialize agent
        self.agent = Agent(
            config=self.agent_config,
            api_key=self.api_key,
            org_id=config_data.get("openai_org_id")
        )
        
        print(f"✅ Basic Chat Agent initialized with model: {model}")
    
    def chat_interactive(self) -> None:
        """Interactive chat session"""
        print("\n🤖 Basic Chat Demo - Interactive Mode")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("Type 'history' to view conversation history")
        print("-" * 50)
        
        conversation_id = "interactive_session"
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'clear':
                    self.agent.clear_memory()
                    print("\n🧹 Conversation history cleared.")
                    continue
                
                elif user_input.lower() == 'history':
                    self._display_history(conversation_id)
                    continue
                
                # Get response from agent
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.agent.chat(user_input, conversation_id)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def chat_single(self, message: str, conversation_id: str = "single_chat") -> str:
        """Single chat interaction"""
        print(f"\n👤 User: {message}")
        response = self.agent.chat(message, conversation_id)
        print(f"🤖 Assistant: {response}")
        return response
    
    async def chat_async(self, message: str, conversation_id: str = "async_chat") -> str:
        """Asynchronous chat interaction"""
        print(f"\n👤 User: {message}")
        response = await self.agent.chat_async(message, conversation_id)
        print(f"🤖 Assistant: {response}")
        return response
    
    def _display_history(self, conversation_id: str) -> None:
        """Display conversation history"""
        history = self.agent.get_conversation_history()
        
        if not history:
            print("\n📝 No conversation history found.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 30)
        
        for i, message in enumerate(history, 1):
            role = message.role.upper()
            content = message.content[:100] + "..." if len(message.content) > 100 else message.content
            print(f"{i}. {role}: {content}")
    
    def demo_conversation(self) -> None:
        """Demonstrate a sample conversation"""
        print("\n🎭 Basic Chat Demo - Sample Conversation")
        print("-" * 50)
        
        conversation_id = "demo_conversation"
        
        # Sample conversation
        messages = [
            "Hello! Can you introduce yourself?",
            "What are your main capabilities?",
            "Can you help me write a short poem about technology?",
            "Thank you! That was great."
        ]
        
        for message in messages:
            response = self.chat_single(message, conversation_id)
            print()  # Add spacing
        
        # Show conversation stats
        stats = self.agent.get_stats()
        print("\n📊 Agent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    async def demo_async_conversation(self) -> None:
        """Demonstrate asynchronous conversation"""
        print("\n⚡ Basic Chat Demo - Async Conversation")
        print("-" * 50)
        
        conversation_id = "async_demo"
        
        messages = [
            "Hello! Can you explain async programming?",
            "What are the benefits of asynchronous operations?",
            "Can you give me a simple example?"
        ]
        
        # Process messages concurrently
        tasks = []
        for i, message in enumerate(messages):
            task = self.chat_async(f"[Message {i+1}] {message}", f"{conversation_id}_{i}")
            tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks)
        
        print(f"\n✅ Processed {len(responses)} messages asynchronously")


def main():
    """Main function to run the basic chat example"""
    # Setup logging
    setup_logging("INFO")
    
    try:
        # Initialize the chat example
        chat_example = BasicChatExample()
        
        print("\n🚀 OpenAI Agents SDK - Basic Chat Demo")
        print("=" * 50)
        
        # Run demo conversation
        chat_example.demo_conversation()
        
        # Run async demo
        print("\n" + "=" * 50)
        asyncio.run(chat_example.demo_async_conversation())
        
        # Start interactive chat
        print("\n" + "=" * 50)
        chat_example.chat_interactive()
        
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