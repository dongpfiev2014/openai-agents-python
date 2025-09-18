"""Streaming chat example demonstrating real-time responses with OpenAI Agent"""

import asyncio
import time
from typing import Optional, AsyncGenerator

from core.agent import Agent, AgentConfig
from core.utils import load_config, setup_logging


class StreamingChatExample:
    """Demonstrates streaming chat functionality with OpenAI Agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        # Load configuration
        config_data = load_config()
        
        # Use provided API key or load from config
        self.api_key = api_key or config_data.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Create agent configuration with streaming enabled
        self.agent_config = AgentConfig(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful AI assistant. Provide detailed, informative responses that demonstrate the streaming capabilities.",
            tools_enabled=False,  # Streaming example focuses on text generation
            memory_enabled=True,
            streaming=True
        )
        
        # Initialize agent
        self.agent = Agent(
            config=self.agent_config,
            api_key=self.api_key,
            org_id=config_data.get("openai_org_id")
        )
        
        print(f"✅ Streaming Chat Agent initialized with model: {model}")
    
    async def stream_response(self, message: str, conversation_id: str = "stream_chat") -> str:
        """Stream a single response and return the complete text"""
        print(f"\n👤 User: {message}")
        print("🤖 Assistant: ", end="", flush=True)
        
        full_response = ""
        start_time = time.time()
        
        async for chunk in self.agent.chat_stream(message, conversation_id):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        end_time = time.time()
        print(f"\n\n⏱️ Streaming completed in {end_time - start_time:.2f} seconds")
        print(f"📝 Total characters: {len(full_response)}")
        
        return full_response
    
    async def demo_basic_streaming(self) -> None:
        """Demonstrate basic streaming functionality"""
        print("\n🌊 Basic Streaming Demo")
        print("-" * 30)
        
        messages = [
            "Tell me a detailed story about a robot learning to paint.",
            "Explain the concept of machine learning in simple terms with examples.",
            "Write a short poem about the beauty of streaming data."
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n--- Example {i} ---")
            await self.stream_response(message, f"basic_stream_{i}")
            
            if i < len(messages):
                print("\n" + "="*50)
                await asyncio.sleep(1)  # Brief pause between examples
    
    async def demo_concurrent_streaming(self) -> None:
        """Demonstrate concurrent streaming (multiple conversations)"""
        print("\n⚡ Concurrent Streaming Demo")
        print("-" * 35)
        
        messages = [
            ("Tell me about the history of artificial intelligence.", "ai_history"),
            ("Explain quantum computing in detail.", "quantum_computing"),
            ("Describe the process of photosynthesis.", "photosynthesis")
        ]
        
        print("🔄 Starting concurrent streaming sessions...\n")
        
        # Create tasks for concurrent streaming
        tasks = []
        for message, conv_id in messages:
            task = self._stream_with_label(message, conv_id)
            tasks.append(task)
        
        # Wait for all streams to complete
        results = await asyncio.gather(*tasks)
        
        print(f"\n✅ Completed {len(results)} concurrent streaming sessions")
    
    async def _stream_with_label(self, message: str, conversation_id: str) -> str:
        """Helper method for labeled streaming"""
        label = conversation_id.replace("_", " ").title()
        print(f"[{label}] Starting stream...")
        
        full_response = ""
        async for chunk in self.agent.chat_stream(message, conversation_id):
            full_response += chunk
        
        print(f"[{label}] Stream completed ({len(full_response)} chars)")
        return full_response
    
    async def demo_interactive_streaming(self) -> None:
        """Interactive streaming chat session"""
        print("\n🤖 Interactive Streaming Chat")
        print("Type your messages and watch responses stream in real-time!")
        print("Commands:")
        print("  - 'quit', 'exit', 'bye' to end")
        print("  - 'clear' to clear conversation history")
        print("  - 'speed' to test streaming speed")
        print("-" * 50)
        
        conversation_id = "interactive_stream"
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'clear':
                    self.agent.clear_memory()
                    print("\n🧹 Conversation history cleared.")
                    continue
                
                elif user_input.lower() == 'speed':
                    await self._speed_test()
                    continue
                
                # Stream the response
                await self.stream_response(user_input, conversation_id)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    async def _speed_test(self) -> None:
        """Test streaming speed with different message lengths"""
        print("\n🏃 Streaming Speed Test")
        print("-" * 25)
        
        test_messages = [
            ("Short response: What is AI?", "Short"),
            ("Medium response: Explain machine learning algorithms and their applications.", "Medium"),
            ("Long response: Write a comprehensive guide about the history, current state, and future of artificial intelligence, including its impact on society.", "Long")
        ]
        
        for message, length in test_messages:
            print(f"\n📏 {length} message test:")
            start_time = time.time()
            
            char_count = 0
            first_chunk_time = None
            
            async for chunk in self.agent.chat_stream(message, f"speed_test_{length.lower()}"):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                char_count += len(chunk)
            
            end_time = time.time()
            total_time = end_time - start_time
            time_to_first_chunk = first_chunk_time - start_time if first_chunk_time else 0
            
            print(f"\n📊 {length} Response Stats:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Time to first chunk: {time_to_first_chunk:.2f}s")
            print(f"  Characters: {char_count}")
            print(f"  Speed: {char_count/total_time:.1f} chars/sec")
    
    async def demo_streaming_with_memory(self) -> None:
        """Demonstrate streaming with conversation memory"""
        print("\n🧠 Streaming with Memory Demo")
        print("-" * 35)
        
        conversation_id = "memory_stream"
        
        # Multi-turn conversation to show memory
        messages = [
            "My name is Alice and I'm a software developer.",
            "What did I just tell you about myself?",
            "Can you suggest some programming projects based on what you know about me?",
            "Remember my name and profession for our future conversations."
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n--- Turn {i} ---")
            await self.stream_response(message, conversation_id)
            
            if i < len(messages):
                await asyncio.sleep(1)
        
        # Show conversation history
        history = self.agent.get_conversation_history()
        print(f"\n📚 Conversation history contains {len(history)} messages")
    
    async def demo_error_handling(self) -> None:
        """Demonstrate error handling in streaming"""
        print("\n⚠️ Error Handling Demo")
        print("-" * 25)
        
        # Test with various edge cases
        test_cases = [
            ("", "Empty message"),
            ("A" * 10000, "Very long message"),
            ("Tell me about 🚀🤖🌟", "Unicode/emoji message")
        ]
        
        for message, description in test_cases:
            print(f"\n🧪 Testing: {description}")
            try:
                if message:  # Skip empty message for actual streaming
                    await self.stream_response(message, f"error_test_{description.replace(' ', '_').lower()}")
                else:
                    print("👤 User: [empty message]")
                    print("🤖 Assistant: Please provide a message to respond to.")
            except Exception as e:
                print(f"❌ Error handled: {e}")
    
    async def run_all_demos(self) -> None:
        """Run all streaming demonstrations"""
        print("\n🚀 Running All Streaming Demos")
        print("=" * 50)
        
        # Run individual demos
        await self.demo_basic_streaming()
        
        print("\n" + "=" * 50)
        await self.demo_streaming_with_memory()
        
        print("\n" + "=" * 50)
        await self.demo_concurrent_streaming()
        
        print("\n" + "=" * 50)
        await self.demo_error_handling()
        
        # Show final statistics
        stats = self.agent.get_stats()
        print("\n📊 Final Agent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Main function to run the streaming chat example"""
    # Setup logging
    setup_logging("INFO")
    
    try:
        # Initialize the streaming example
        stream_example = StreamingChatExample()
        
        print("\n🚀 OpenAI Agents SDK - Streaming Chat Demo")
        print("=" * 50)
        
        # Run all demos
        asyncio.run(stream_example.run_all_demos())
        
        # Start interactive streaming chat
        print("\n" + "=" * 50)
        asyncio.run(stream_example.demo_interactive_streaming())
        
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