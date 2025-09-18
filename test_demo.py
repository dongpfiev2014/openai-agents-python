#!/usr/bin/env python3
"""Test Script - Validate OpenAI Agents SDK functionality"""

import asyncio
import sys
import os
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent
from core.tools import Tool, ToolRegistry, tool
from core.memory import ConversationMemory, PersistentMemory
from core.utils import load_config, setup_logging


class TestOpenAIAgentsSDK(unittest.TestCase):
    """Test suite for OpenAI Agents SDK"""
    
    def setUp(self):
        """Set up test environment"""
        self.api_key = "test-api-key"
        self.model = "gpt-4"
        
        # Mock OpenAI client
        self.mock_openai_client = Mock()
        
    def test_agent_initialization(self):
        """Test agent initialization"""
        from src.core.agent import AgentConfig
        
        config = AgentConfig(
            model=self.model,
            system_prompt="Test system prompt"
        )
        
        agent = Agent(
            config=config,
            api_key=self.api_key
        )
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.config.model, self.model)
        self.assertEqual(agent.config.system_prompt, "Test system prompt")
        self.assertIsNotNone(agent.client)
        self.assertIsInstance(agent.tool_registry, ToolRegistry)
        
    def test_tool_creation(self):
        """Test tool creation and registration"""
        # Test function-based tool creation
        @tool("test_tool", "A test tool")
        def test_function(param: str) -> str:
            return f"Test result: {param}"
        
        self.assertEqual(test_function._tool_name, "test_tool")
        self.assertEqual(test_function._tool_description, "A test tool")
        self.assertTrue(test_function._is_tool)
        
        # Test Tool dataclass
        tool_obj = Tool(
            name="manual_tool",
            description="Manually created tool",
            function=lambda x: f"Manual: {x}",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }
        )
        
        self.assertEqual(tool_obj.name, "manual_tool")
        self.assertTrue(callable(tool_obj.function))
        
    def test_tool_registry(self):
        """Test tool registry functionality"""
        registry = ToolRegistry()
        
        # Create test tool using Tool.from_function
        def registry_test_func(value: int) -> int:
            return value * 2
        
        # Register tool using register_function
        registry.register_function(registry_test_func, "registry_test", "Registry test tool")
        
        # Test retrieval
        retrieved_tool = registry.get_tool("registry_test")
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.name, "registry_test")
        
        # Test executing tool
        result = retrieved_tool.function(5)
        self.assertEqual(result, 10)
        
        # Test listing
        tool_names = registry.list_tools()
        self.assertIn("registry_test", tool_names)
        
    def test_memory_functionality(self):
        """Test memory management"""
        memory = ConversationMemory()
        
        # Test adding messages
        memory.add_message("session1", "user", "Hello")
        memory.add_message("session1", "assistant", "Hi there!")
        
        # Test retrieval
        history = memory.get_history("session1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[0]['content'], 'Hello')
        
        # Test search
        results = memory.search_messages("Hello", "session1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, 'Hello')
        
        # Test session management
        conversation_ids = memory.get_conversation_ids()
        self.assertIn("session1", conversation_ids)
        
    def test_persistent_memory(self):
        """Test persistent memory functionality"""
        # Use temporary file for testing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir='.') as tmp:
            tmp_path = tmp.name
        
        try:
            memory = ConversationMemory()
            
            # Add messages
            memory.add_message("test_session", "user", "Persistent test")
            memory.add_message("test_session", "assistant", "Persistent response")
            
            # Save to file
            memory.save(tmp_path)
            
            # Create new instance and load
            new_memory = ConversationMemory()
            new_memory.load(tmp_path)
            
            # Verify data persistence
            history = new_memory.get_history("test_session")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]['content'], 'Persistent test')
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Test with environment variables
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'DEFAULT_MODEL': 'gpt-3.5-turbo'}):
            config = load_config()
            self.assertEqual(config['openai_api_key'], 'test-key')
            self.assertEqual(config['default_model'], 'gpt-3.5-turbo')
    
    @patch('openai.AsyncOpenAI')
    async def test_agent_chat_mock(self, mock_openai):
        """Test agent chat with mocked OpenAI client"""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mocked response"
        mock_response.choices[0].message.tool_calls = None
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create agent and test chat
        agent = Agent(api_key=self.api_key, model=self.model)
        response = await agent.chat("Test message")
        
        self.assertEqual(response, "Mocked response")
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('openai.AsyncOpenAI')
    async def test_agent_streaming_mock(self, mock_openai):
        """Test agent streaming with mocked OpenAI client"""
        # Setup mock streaming response
        async def mock_stream():
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.choices = [Mock()]
                mock_chunk.choices[0].delta = Mock()
                mock_chunk.choices[0].delta.content = chunk
                mock_chunk.choices[0].delta.tool_calls = None
                yield mock_chunk
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_stream()
        mock_openai.return_value = mock_client
        
        # Create agent and test streaming
        agent = Agent(api_key=self.api_key, model=self.model)
        
        result = ""
        async for chunk in agent.chat_stream("Test streaming"):
            result += chunk
        
        self.assertEqual(result, "Hello world!")


class IntegrationTests:
    """Integration tests for the complete SDK"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.results = {}
    
    async def test_basic_functionality(self) -> bool:
        """Test basic agent functionality"""
        try:
            print("🧪 Testing basic functionality...")
            
            agent = Agent(
                api_key=self.api_key,
                model=self.model,
                name="TestAgent",
                description="Integration test agent"
            )
            
            # Test simple chat
            response = await agent.chat("Say 'Integration test successful' if you can understand this.")
            
            if "integration test successful" in response.lower():
                print("✅ Basic functionality test passed")
                return True
            else:
                print(f"❌ Basic functionality test failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Basic functionality test error: {e}")
            return False
    
    async def test_tool_integration(self) -> bool:
        """Test tool integration"""
        try:
            print("🧪 Testing tool integration...")
            
            # Create test tool
            @tool("test_calculator", "Simple calculator for testing")
            def test_calculator(operation: str, a: float, b: float) -> float:
                """Perform basic math operations"""
                if operation == "add":
                    return a + b
                elif operation == "multiply":
                    return a * b
                else:
                    return 0
            
            agent = Agent(
                api_key=self.api_key,
                model=self.model,
                name="ToolTestAgent"
            )
            
            # Register tool
            agent.tools.register(test_calculator)
            
            # Test tool usage
            response = await agent.chat("Use the test_calculator to add 5 and 3")
            
            if "8" in response:
                print("✅ Tool integration test passed")
                return True
            else:
                print(f"❌ Tool integration test failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Tool integration test error: {e}")
            return False
    
    async def test_memory_integration(self) -> bool:
        """Test memory integration"""
        try:
            print("🧪 Testing memory integration...")
            
            agent = Agent(
                api_key=self.api_key,
                model=self.model,
                name="MemoryTestAgent"
            )
            
            # First conversation
            await agent.chat("Remember that my favorite color is purple.", "memory_test")
            
            # Second conversation - test memory
            response = await agent.chat("What is my favorite color?", "memory_test")
            
            if "purple" in response.lower():
                print("✅ Memory integration test passed")
                return True
            else:
                print(f"❌ Memory integration test failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Memory integration test error: {e}")
            return False
    
    async def test_streaming_integration(self) -> bool:
        """Test streaming integration"""
        try:
            print("🧪 Testing streaming integration...")
            
            agent = Agent(
                api_key=self.api_key,
                model=self.model,
                name="StreamTestAgent"
            )
            
            # Test streaming
            chunks = []
            async for chunk in agent.chat_stream("Count from 1 to 3, one number per sentence."):
                chunks.append(chunk)
            
            full_response = "".join(chunks)
            
            if len(chunks) > 1 and ("1" in full_response and "2" in full_response and "3" in full_response):
                print("✅ Streaming integration test passed")
                return True
            else:
                print(f"❌ Streaming integration test failed: {full_response}")
                return False
                
        except Exception as e:
            print(f"❌ Streaming integration test error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        print("\n🧪 RUNNING INTEGRATION TESTS")
        print("=" * 50)
        
        tests = [
            ("basic_functionality", self.test_basic_functionality),
            ("tool_integration", self.test_tool_integration),
            ("memory_integration", self.test_memory_integration),
            ("streaming_integration", self.test_streaming_integration)
        ]
        
        results = {}
        successful = 0
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name}...")
            try:
                success = await test_func()
                results[test_name] = success
                if success:
                    successful += 1
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 50)
        
        print(f"\n📈 Results: {successful}/{len(tests)} tests passed")
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name:25} {status}")
        
        if successful == len(tests):
            print("\n🎉 All integration tests passed!")
        else:
            print(f"\n⚠️ {len(tests) - successful} test(s) failed.")
        
        return results


async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI Agents SDK Test Suite")
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--api-key', help='OpenAI API key for integration tests')
    parser.add_argument('--model', default='gpt-4', help='Model to use for integration tests')
    
    args = parser.parse_args()
    
    if args.unit or args.all:
        print("\n🧪 RUNNING UNIT TESTS")
        print("=" * 40)
        
        # Run unit tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    
    if args.integration or args.all:
        # Get API key
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("\n❌ OpenAI API key required for integration tests")
            print("Set OPENAI_API_KEY environment variable or use --api-key")
            return
        
        # Run integration tests
        integration_tester = IntegrationTests(api_key, args.model)
        await integration_tester.run_all_tests()
    
    if not (args.unit or args.integration or args.all):
        print("\n📋 OpenAI Agents SDK Test Suite")
        print("Use --unit, --integration, or --all to run tests")
        print("Use --help for more options")


if __name__ == "__main__":
    asyncio.run(main())