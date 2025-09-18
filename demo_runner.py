#!/usr/bin/env python3
"""Demo Runner - Execute all OpenAI Agents SDK examples"""

import asyncio
import sys
import os
from typing import Optional
import argparse
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.utils import load_config, setup_logging
from examples.basic_chat import BasicChatExample
from examples.function_calling import FunctionCallingExample
from examples.streaming_chat import StreamingChatExample
from examples.memory_demo import MemoryDemoExample
from examples.multi_agent import MultiAgentExample


class DemoRunner:
    """Main demo runner for all OpenAI Agents SDK examples"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        # Load configuration
        config_data = load_config()
        
        # Use provided API key or load from config
        self.api_key = api_key or config_data.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.examples = {}
        self.results = {}
        
        print(f"🚀 Demo Runner initialized with model: {model}")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _initialize_examples(self):
        """Initialize all example classes"""
        try:
            print("\n🔧 Initializing examples...")
            
            self.examples = {
                'basic_chat': BasicChatExample(self.api_key, self.model),
                'function_calling': FunctionCallingExample(self.api_key, self.model),
                'streaming_chat': StreamingChatExample(self.api_key, self.model),
                'memory_demo': MemoryDemoExample(self.api_key, self.model),
                'multi_agent': MultiAgentExample(self.api_key, self.model)
            }
            
            print(f"✅ Initialized {len(self.examples)} examples successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize examples: {e}")
            raise
    
    async def run_basic_chat_demo(self) -> bool:
        """Run basic chat demonstration"""
        print("\n" + "=" * 60)
        print("🗨️ BASIC CHAT DEMO")
        print("=" * 60)
        
        try:
            example = self.examples['basic_chat']
            await example.run_all_demos()
            print("\n✅ Basic chat demo completed successfully")
            return True
        except Exception as e:
            print(f"\n❌ Basic chat demo failed: {e}")
            return False
    
    async def run_function_calling_demo(self) -> bool:
        """Run function calling demonstration"""
        print("\n" + "=" * 60)
        print("🔧 FUNCTION CALLING DEMO")
        print("=" * 60)
        
        try:
            example = self.examples['function_calling']
            await example.run_all_demos()
            print("\n✅ Function calling demo completed successfully")
            return True
        except Exception as e:
            print(f"\n❌ Function calling demo failed: {e}")
            return False
    
    async def run_streaming_demo(self) -> bool:
        """Run streaming chat demonstration"""
        print("\n" + "=" * 60)
        print("🌊 STREAMING CHAT DEMO")
        print("=" * 60)
        
        try:
            example = self.examples['streaming_chat']
            await example.run_all_demos()
            print("\n✅ Streaming chat demo completed successfully")
            return True
        except Exception as e:
            print(f"\n❌ Streaming chat demo failed: {e}")
            return False
    
    async def run_memory_demo(self) -> bool:
        """Run memory management demonstration"""
        print("\n" + "=" * 60)
        print("🧠 MEMORY MANAGEMENT DEMO")
        print("=" * 60)
        
        try:
            example = self.examples['memory_demo']
            await example.run_all_demos()
            print("\n✅ Memory management demo completed successfully")
            return True
        except Exception as e:
            print(f"\n❌ Memory management demo failed: {e}")
            return False
    
    async def run_multi_agent_demo(self) -> bool:
        """Run multi-agent coordination demonstration"""
        print("\n" + "=" * 60)
        print("🤝 MULTI-AGENT COORDINATION DEMO")
        print("=" * 60)
        
        try:
            example = self.examples['multi_agent']
            await example.run_all_demos()
            print("\n✅ Multi-agent coordination demo completed successfully")
            return True
        except Exception as e:
            print(f"\n❌ Multi-agent coordination demo failed: {e}")
            return False
    
    async def run_quick_test(self) -> bool:
        """Run a quick test of core functionality"""
        print("\n" + "=" * 60)
        print("⚡ QUICK FUNCTIONALITY TEST")
        print("=" * 60)
        
        try:
            # Test basic chat
            print("\n🧪 Testing basic chat...")
            example = self.examples['basic_chat']
            response = await example.agent.chat("Hello! Please confirm the system is working properly.")
            print(f"✅ Basic chat: {response[:100]}...")
            
            # Test function calling
            print("\n🧪 Testing function calling...")
            func_example = self.examples['function_calling']
            func_response = await func_example.agent.chat("What time is it right now?")
            print(f"✅ Function calling: {func_response[:100]}...")
            
            # Test streaming
            print("\n🧪 Testing streaming...")
            stream_example = self.examples['streaming_chat']
            stream_text = ""
            async for chunk in stream_example.agent.chat_stream("Count from 1 to 5."):
                stream_text += chunk
            print(f"✅ Streaming: {stream_text[:100]}...")
            
            # Test memory
            print("\n🧪 Testing memory...")
            memory_example = self.examples['memory_demo']
            await memory_example.agent.chat("Remember: my favorite color is blue.", "test_memory")
            memory_response = await memory_example.agent.chat("What's my favorite color?", "test_memory")
            print(f"✅ Memory: {memory_response[:100]}...")
            
            print("\n✅ Quick test completed successfully")
            return True
            
        except Exception as e:
            print(f"\n❌ Quick test failed: {e}")
            return False
    
    async def run_all_demos(self) -> dict:
        """Run all demonstrations"""
        print("\n🚀 RUNNING ALL OPENAI AGENTS SDK DEMOS")
        print("=" * 80)
        
        # Initialize examples
        self._initialize_examples()
        
        # Run all demos
        demos = [
            ('basic_chat', self.run_basic_chat_demo),
            ('function_calling', self.run_function_calling_demo),
            ('streaming_chat', self.run_streaming_demo),
            ('memory_demo', self.run_memory_demo),
            ('multi_agent', self.run_multi_agent_demo)
        ]
        
        results = {}
        successful = 0
        total = len(demos)
        
        for demo_name, demo_func in demos:
            print(f"\n📋 Running {demo_name} demo...")
            start_time = datetime.now()
            
            try:
                success = await demo_func()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                results[demo_name] = {
                    'success': success,
                    'duration': duration,
                    'error': None
                }
                
                if success:
                    successful += 1
                    print(f"✅ {demo_name} completed in {duration:.2f}s")
                else:
                    print(f"❌ {demo_name} failed after {duration:.2f}s")
                    
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                results[demo_name] = {
                    'success': False,
                    'duration': duration,
                    'error': str(e)
                }
                
                print(f"❌ {demo_name} failed with error: {e}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 DEMO EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"\n📈 Overall Results: {successful}/{total} demos successful")
        print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📋 Individual Results:")
        for demo_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            print(f"  {demo_name:20} {status:8} ({duration:6.2f}s)")
            if result['error']:
                print(f"    Error: {result['error']}")
        
        total_duration = sum(r['duration'] for r in results.values())
        print(f"\n⏱️ Total execution time: {total_duration:.2f} seconds")
        
        if successful == total:
            print("\n🎉 All demos completed successfully!")
        else:
            print(f"\n⚠️ {total - successful} demo(s) failed. Check the logs above for details.")
        
        return results
    
    async def run_interactive_demo(self):
        """Run interactive demo selection"""
        print("\n🎮 INTERACTIVE DEMO MODE")
        print("=" * 40)
        
        # Initialize examples
        self._initialize_examples()
        
        while True:
            print("\n📋 Available Demos:")
            print("1. Basic Chat")
            print("2. Function Calling")
            print("3. Streaming Chat")
            print("4. Memory Management")
            print("5. Multi-Agent Coordination")
            print("6. Quick Test")
            print("7. Run All Demos")
            print("0. Exit")
            
            try:
                choice = input("\n👉 Select demo (0-7): ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye!")
                    break
                elif choice == '1':
                    await self.run_basic_chat_demo()
                elif choice == '2':
                    await self.run_function_calling_demo()
                elif choice == '3':
                    await self.run_streaming_demo()
                elif choice == '4':
                    await self.run_memory_demo()
                elif choice == '5':
                    await self.run_multi_agent_demo()
                elif choice == '6':
                    await self.run_quick_test()
                elif choice == '7':
                    await self.run_all_demos()
                else:
                    print("❌ Invalid choice. Please select 0-7.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_runner.py --all                    # Run all demos
  python demo_runner.py --quick                  # Quick functionality test
  python demo_runner.py --interactive            # Interactive mode
  python demo_runner.py --demo basic_chat        # Run specific demo
  python demo_runner.py --model gpt-3.5-turbo   # Use different model
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Run all demonstrations'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Run quick functionality test'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true', 
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--demo', 
        choices=['basic_chat', 'function_calling', 'streaming_chat', 'memory_demo', 'multi_agent'],
        help='Run specific demo'
    )
    
    parser.add_argument(
        '--model', 
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini - cheapest option)'
    )
    
    parser.add_argument(
        '--api-key', 
        help='OpenAI API key (overrides environment variable)'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Initialize demo runner
        runner = DemoRunner(api_key=args.api_key, model=args.model)
        
        # Run based on arguments
        if args.all:
            asyncio.run(runner.run_all_demos())
        elif args.quick:
            asyncio.run(runner.run_quick_test())
        elif args.interactive:
            asyncio.run(runner.run_interactive_demo())
        elif args.demo:
            runner._initialize_examples()
            if args.demo == 'basic_chat':
                asyncio.run(runner.run_basic_chat_demo())
            elif args.demo == 'function_calling':
                asyncio.run(runner.run_function_calling_demo())
            elif args.demo == 'streaming_chat':
                asyncio.run(runner.run_streaming_demo())
            elif args.demo == 'memory_demo':
                asyncio.run(runner.run_memory_demo())
            elif args.demo == 'multi_agent':
                asyncio.run(runner.run_multi_agent_demo())
        else:
            # Default to interactive mode
            asyncio.run(runner.run_interactive_demo())
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 To fix this:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to the .env file")
        print("3. Run the demo again")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Demo runner interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()