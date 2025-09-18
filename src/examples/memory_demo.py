"""Memory demonstration example showing conversation history and context management"""

import asyncio
import json
from typing import Optional, Dict, List
from datetime import datetime

from core.agent import Agent, AgentConfig
from core.memory import ConversationMemory, PersistentMemory
from core.utils import load_config, setup_logging


class MemoryDemoExample:
    """Demonstrates memory management and conversation context with OpenAI Agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        # Load configuration
        config_data = load_config()
        
        # Use provided API key or load from config
        self.api_key = api_key or config_data.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Create agent configuration with memory enabled
        self.agent_config = AgentConfig(
            model=model,
            temperature=0.7,
            max_tokens=1500,
            system_prompt="You are a helpful AI assistant with excellent memory. Remember details from our conversation and reference them when relevant.",
            tools_enabled=False,
            memory_enabled=True,
            streaming=False
        )
        
        # Initialize agent
        self.agent = Agent(
            config=self.agent_config,
            api_key=self.api_key,
            org_id=config_data.get("openai_org_id")
        )
        
        # Initialize different memory types for demonstration
        self.conversation_memory = ConversationMemory(max_messages=50)
        self.persistent_memory = PersistentMemory(
            memory_file="demo_memory.json",
            max_messages=100
        )
        
        print(f"✅ Memory Demo Agent initialized with model: {model}")
    
    async def demo_basic_memory(self) -> None:
        """Demonstrate basic conversation memory"""
        print("\n🧠 Basic Memory Demo")
        print("-" * 25)
        
        conversation_id = "basic_memory"
        
        # Multi-turn conversation to build context
        messages = [
            "Hi! My name is Sarah and I'm a data scientist working on machine learning projects.",
            "What's my name and profession?",
            "I'm currently working on a project involving natural language processing. Can you suggest some Python libraries?",
            "Based on what you know about me, what kind of career advice would you give?",
            "Can you summarize everything you know about me from our conversation?"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n--- Turn {i} ---")
            print(f"👤 User: {message}")
            
            response = await self.agent.chat(message, conversation_id)
            print(f"🤖 Assistant: {response}")
            
            # Show memory state after each turn
            history = self.agent.get_conversation_history(conversation_id)
            print(f"📚 Memory: {len(history)} messages stored")
            
            if i < len(messages):
                await asyncio.sleep(1)
    
    async def demo_memory_persistence(self) -> None:
        """Demonstrate persistent memory across sessions"""
        print("\n💾 Memory Persistence Demo")
        print("-" * 30)
        
        # First session - store information
        print("\n📝 Session 1: Storing information")
        session1_messages = [
            "Remember that I prefer Python over JavaScript for backend development.",
            "My favorite machine learning framework is PyTorch.",
            "I have 5 years of experience in software development."
        ]
        
        conversation_id = "persistent_session"
        
        for message in session1_messages:
            print(f"👤 User: {message}")
            response = await self.agent.chat(message, conversation_id)
            print(f"🤖 Assistant: {response}\n")
        
        # Save memory state
        history = self.agent.get_conversation_history(conversation_id)
        self.persistent_memory.clear()
        for msg in history:
            self.persistent_memory.add_message(
                msg['role'], 
                msg['content'], 
                conversation_id,
                metadata=msg.get('metadata', {})
            )
        self.persistent_memory.save_to_file()
        print("💾 Memory saved to file")
        
        # Simulate new session - load information
        print("\n🔄 Session 2: Loading previous information")
        
        # Create new agent instance to simulate fresh session
        new_agent = Agent(
            config=self.agent_config,
            api_key=self.api_key
        )
        
        # Load previous memory
        loaded_memory = PersistentMemory(memory_file="demo_memory.json")
        loaded_memory.load_from_file()
        
        # Restore conversation history
        loaded_history = loaded_memory.get_messages(conversation_id)
        for msg in loaded_history:
            new_agent.memory.add_message(
                msg['role'],
                msg['content'],
                conversation_id,
                metadata=msg.get('metadata', {})
            )
        
        # Test if information is remembered
        test_message = "What do you remember about my programming preferences and experience?"
        print(f"👤 User: {test_message}")
        response = await new_agent.chat(test_message, conversation_id)
        print(f"🤖 Assistant: {response}")
        
        print(f"\n✅ Successfully loaded {len(loaded_history)} previous messages")
    
    async def demo_memory_search(self) -> None:
        """Demonstrate memory search functionality"""
        print("\n🔍 Memory Search Demo")
        print("-" * 25)
        
        conversation_id = "search_demo"
        
        # Build a rich conversation history
        context_messages = [
            "I work at TechCorp as a senior software engineer.",
            "My main programming languages are Python, JavaScript, and Go.",
            "I'm passionate about artificial intelligence and machine learning.",
            "I have a pet dog named Max who loves to play fetch.",
            "My favorite books are about science fiction and technology.",
            "I enjoy hiking and photography in my free time.",
            "I'm currently learning about quantum computing.",
            "My team is working on a microservices architecture project."
        ]
        
        print("📚 Building conversation context...")
        for message in context_messages:
            await self.agent.chat(message, conversation_id)
        
        print(f"✅ Built context with {len(context_messages)} messages\n")
        
        # Demonstrate search functionality
        search_queries = [
            "programming",
            "pet",
            "hobbies",
            "work",
            "learning"
        ]
        
        for query in search_queries:
            print(f"🔍 Searching for: '{query}'")
            
            # Search in memory
            results = self.agent.memory.search_messages(query, conversation_id)
            
            print(f"📋 Found {len(results)} relevant messages:")
            for i, result in enumerate(results[:3], 1):  # Show top 3 results
                content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                print(f"  {i}. {content}")
            
            # Ask agent to use the search results
            search_message = f"Based on what you know about {query}, tell me more."
            print(f"\n👤 User: {search_message}")
            response = await self.agent.chat(search_message, conversation_id)
            print(f"🤖 Assistant: {response}\n")
            print("-" * 40)
    
    async def demo_memory_limits(self) -> None:
        """Demonstrate memory limits and management"""
        print("\n📏 Memory Limits Demo")
        print("-" * 25)
        
        # Create agent with limited memory
        limited_config = AgentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            system_prompt="You are an assistant with limited memory.",
            memory_enabled=True
        )
        
        limited_agent = Agent(
            config=limited_config,
            api_key=self.api_key
        )
        
        # Set small memory limit for demonstration
        limited_agent.memory.max_messages = 6  # Very small limit
        
        conversation_id = "limited_memory"
        
        # Send more messages than the limit
        messages = [
            "Message 1: I like pizza.",
            "Message 2: I work in tech.",
            "Message 3: I have a cat named Whiskers.",
            "Message 4: I enjoy reading books.",
            "Message 5: I live in San Francisco.",
            "Message 6: I play guitar in my spare time.",
            "Message 7: I'm learning French.",
            "Message 8: Do you remember what I said about pizza?"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n--- Message {i} ---")
            print(f"👤 User: {message}")
            
            response = await limited_agent.chat(message, conversation_id)
            print(f"🤖 Assistant: {response}")
            
            # Show current memory state
            history = limited_agent.get_conversation_history(conversation_id)
            print(f"📚 Memory: {len(history)} messages (limit: {limited_agent.memory.max_messages})")
            
            if len(history) > 0:
                oldest_msg = history[0]['content'][:50] + "..." if len(history[0]['content']) > 50 else history[0]['content']
                print(f"📜 Oldest message: {oldest_msg}")
    
    async def demo_memory_metadata(self) -> None:
        """Demonstrate memory with metadata and timestamps"""
        print("\n🏷️ Memory Metadata Demo")
        print("-" * 25)
        
        conversation_id = "metadata_demo"
        
        # Messages with different contexts
        messages_with_context = [
            ("I'm planning a vacation to Japan.", {"topic": "travel", "priority": "high"}),
            ("What's the weather like today?", {"topic": "weather", "priority": "low"}),
            ("I need help with my Python code.", {"topic": "programming", "priority": "high"}),
            ("Tell me a joke.", {"topic": "entertainment", "priority": "low"}),
            ("How do I book flights to Tokyo?", {"topic": "travel", "priority": "high"})
        ]
        
        for message, metadata in messages_with_context:
            print(f"\n👤 User: {message}")
            print(f"🏷️ Metadata: {metadata}")
            
            # Add message with metadata
            self.agent.memory.add_message(
                "user", 
                message, 
                conversation_id,
                metadata=metadata
            )
            
            response = await self.agent.chat(message, conversation_id)
            print(f"🤖 Assistant: {response}")
        
        # Demonstrate filtering by metadata
        print("\n🔍 Filtering by metadata:")
        
        # Get high priority messages
        all_messages = self.agent.memory.get_messages(conversation_id)
        high_priority = [msg for msg in all_messages if msg.get('metadata', {}).get('priority') == 'high']
        
        print(f"📋 High priority messages ({len(high_priority)}):")
        for msg in high_priority:
            content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
            print(f"  • {content}")
        
        # Get travel-related messages
        travel_messages = [msg for msg in all_messages if msg.get('metadata', {}).get('topic') == 'travel']
        
        print(f"\n✈️ Travel-related messages ({len(travel_messages)}):")
        for msg in travel_messages:
            content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
            print(f"  • {content}")
    
    async def demo_memory_export_import(self) -> None:
        """Demonstrate memory export and import functionality"""
        print("\n📤📥 Memory Export/Import Demo")
        print("-" * 35)
        
        conversation_id = "export_demo"
        
        # Create some conversation history
        messages = [
            "I'm a software architect with 10 years of experience.",
            "I specialize in cloud computing and distributed systems.",
            "My current project involves building a scalable microservices platform."
        ]
        
        for message in messages:
            await self.agent.chat(message, conversation_id)
        
        # Export memory
        print("📤 Exporting memory...")
        exported_data = {
            "conversation_id": conversation_id,
            "messages": self.agent.memory.get_messages(conversation_id),
            "export_timestamp": datetime.now().isoformat(),
            "agent_config": {
                "model": self.agent_config.model,
                "temperature": self.agent_config.temperature
            }
        }
        
        # Save to file
        export_file = "exported_memory.json"
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(exported_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Memory exported to {export_file}")
        print(f"📊 Exported {len(exported_data['messages'])} messages")
        
        # Import memory (simulate new session)
        print("\n📥 Importing memory...")
        
        # Create new agent
        new_agent = Agent(
            config=self.agent_config,
            api_key=self.api_key
        )
        
        # Load exported data
        with open(export_file, 'r', encoding='utf-8') as f:
            imported_data = json.load(f)
        
        # Restore messages
        for msg in imported_data['messages']:
            new_agent.memory.add_message(
                msg['role'],
                msg['content'],
                imported_data['conversation_id'],
                metadata=msg.get('metadata', {})
            )
        
        print(f"✅ Memory imported from {export_file}")
        print(f"📊 Imported {len(imported_data['messages'])} messages")
        
        # Test imported memory
        test_message = "What do you know about my professional background?"
        print(f"\n👤 User: {test_message}")
        response = await new_agent.chat(test_message, imported_data['conversation_id'])
        print(f"🤖 Assistant: {response}")
    
    async def run_all_demos(self) -> None:
        """Run all memory demonstrations"""
        print("\n🚀 Running All Memory Demos")
        print("=" * 50)
        
        # Run individual demos
        await self.demo_basic_memory()
        
        print("\n" + "=" * 50)
        await self.demo_memory_persistence()
        
        print("\n" + "=" * 50)
        await self.demo_memory_search()
        
        print("\n" + "=" * 50)
        await self.demo_memory_limits()
        
        print("\n" + "=" * 50)
        await self.demo_memory_metadata()
        
        print("\n" + "=" * 50)
        await self.demo_memory_export_import()
        
        # Show final statistics
        stats = self.agent.get_stats()
        print("\n📊 Final Agent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Main function to run the memory demo example"""
    # Setup logging
    setup_logging("INFO")
    
    try:
        # Initialize the memory example
        memory_example = MemoryDemoExample()
        
        print("\n🚀 OpenAI Agents SDK - Memory Demo")
        print("=" * 50)
        
        # Run all demos
        asyncio.run(memory_example.run_all_demos())
        
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