# OpenAI Agents SDK Demo

A comprehensive demonstration repository showcasing the full capabilities of OpenAI Agents SDK with practical examples, advanced features, and multi-agent coordination.

## 🚀 Features

- **Core Agent Functionality**: Basic chat, streaming responses, and conversation management
- **Function Calling**: Custom tools and external API integrations
- **Memory Management**: Persistent conversation history and context retention
- **Multi-Agent Coordination**: Agent collaboration, debates, and structured workflows
- **Advanced Tools**: Built-in utilities for web search, file operations, and calculations
- **Streaming Support**: Real-time response generation with performance monitoring
- **Configuration Management**: Flexible environment setup and agent customization

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Internet connection for API calls and web search functionality

## 🛠️ Installation

### 1. Clone or Download

If you have this repository, navigate to the project directory:

```bash
cd openai-agents-python
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` file and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_ORG_ID=your_org_id_here  # Optional
   ```

3. Configure other settings as needed (see Configuration section below)

## 🎯 Quick Start

### Basic Chat Example

```python
from src.examples.basic_chat import BasicChatExample

# Initialize and run basic chat
chat_example = BasicChatExample()
await chat_example.demo_basic_chat()
```

### Run Individual Examples

```bash
# Basic chat functionality
python -m src.examples.basic_chat

# Function calling and tools
python -m src.examples.function_calling

# Streaming responses
python -m src.examples.streaming_chat

# Memory management
python -m src.examples.memory_demo

# Multi-agent coordination
python -m src.examples.multi_agent
```

## 📚 Examples Overview

### 1. Basic Chat (`src/examples/basic_chat.py`)

**Features Demonstrated:**
- Simple agent initialization
- Synchronous and asynchronous chat
- Interactive conversation loops
- Basic configuration options

**Key Methods:**
- `demo_basic_chat()`: Simple question-answer interactions
- `demo_async_conversation()`: Asynchronous conversation handling
- `interactive_chat()`: Real-time user interaction

### 2. Function Calling (`src/examples/function_calling.py`)

**Features Demonstrated:**
- Custom tool creation and registration
- External API integrations
- Multi-tool workflows
- Asynchronous function calling

**Available Tools:**
- Weather information retrieval
- Task management system
- Knowledge base search
- Notification system
- Built-in utilities (time, calculations, web search)

### 3. Streaming Chat (`src/examples/streaming_chat.py`)

**Features Demonstrated:**
- Real-time response streaming
- Concurrent streaming sessions
- Performance monitoring
- Interactive streaming chat
- Error handling in streams

**Key Features:**
- Character-by-character streaming
- Speed testing and metrics
- Memory integration with streaming
- Concurrent conversation handling

### 4. Memory Management (`src/examples/memory_demo.py`)

**Features Demonstrated:**
- Conversation history persistence
- Memory search and filtering
- Memory limits and management
- Metadata and timestamps
- Export/import functionality

**Memory Types:**
- `ConversationMemory`: In-memory conversation storage
- `PersistentMemory`: File-based persistence
- `MemoryManager`: Multi-conversation management

### 5. Multi-Agent System (`src/examples/multi_agent.py`)

**Features Demonstrated:**
- Specialized agent roles
- Inter-agent communication
- Collaborative problem-solving
- Structured debates
- Workflow coordination

**Agent Roles:**
- **Researcher**: Data gathering and analysis
- **Creative**: Innovation and ideation
- **Technical**: Architecture and implementation
- **Coordinator**: Task management and synthesis
- **QA**: Quality assurance and validation

## ⚙️ Configuration

### Environment Variables

```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_ORG_ID=your_org_id_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Agent Configuration
AGENT_SYSTEM_PROMPT=You are a helpful AI assistant.
AGENT_TOOLS_ENABLED=true
AGENT_MEMORY_ENABLED=true
AGENT_STREAMING=false

# Application Settings
APP_LOG_LEVEL=INFO
APP_DEBUG=false

# FastAPI Server (for web interfaces)
SERVER_HOST=localhost
SERVER_PORT=8000
SERVER_RELOAD=true

# Streamlit Configuration (for UI demos)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Agent Configuration Options

```python
from src.core.agent import AgentConfig

config = AgentConfig(
    model="gpt-4",                    # OpenAI model to use
    temperature=0.7,                 # Response creativity (0.0-1.0)
    max_tokens=2000,                 # Maximum response length
    system_prompt="Custom prompt",   # System behavior instructions
    tools_enabled=True,              # Enable function calling
    memory_enabled=True,             # Enable conversation memory
    streaming=False,                 # Enable response streaming
    timeout=30.0,                    # Request timeout in seconds
    max_retries=3                    # Maximum retry attempts
)
```

## 🔧 Core Components

### Agent Class (`src/core/agent.py`)

The main agent implementation with:
- Configuration management
- Message handling
- Tool integration
- Memory management
- Streaming support
- Statistics tracking

### Tools System (`src/core/tools.py`)

Flexible tool system featuring:
- Tool registration and validation
- Built-in utility tools
- Custom tool creation
- Function decorators
- Tool registry management

### Memory Management (`src/core/memory.py`)

Comprehensive memory system with:
- Multiple memory backends
- Search and filtering
- Persistence options
- Metadata support
- Memory limits and cleanup

### Utilities (`src/core/utils.py`)

Helper functions for:
- Configuration loading
- Logging setup
- JSON handling
- Text formatting
- Project structure management

## 🧪 Testing

### Run All Examples

```bash
# Test basic functionality
python -m src.examples.basic_chat

# Test function calling
python -m src.examples.function_calling

# Test streaming
python -m src.examples.streaming_chat

# Test memory management
python -m src.examples.memory_demo

# Test multi-agent coordination
python -m src.examples.multi_agent
```

### Verify Installation

```python
# Quick verification script
from src.core.agent import Agent, AgentConfig
from src.core.utils import load_config

# Load configuration
config_data = load_config()
api_key = config_data.get("openai_api_key")

if api_key:
    # Create test agent
    agent_config = AgentConfig(model="gpt-3.5-turbo")
    agent = Agent(agent_config, api_key)
    
    # Test basic functionality
    response = await agent.chat("Hello! Can you confirm the setup is working?")
    print(f"✅ Setup verified: {response[:100]}...")
else:
    print("❌ Please configure your OpenAI API key in .env file")
```

## 📖 Usage Patterns

### 1. Simple Chat Agent

```python
from src.core.agent import Agent, AgentConfig

# Create agent
config = AgentConfig(model="gpt-4", temperature=0.7)
agent = Agent(config, api_key="your_key")

# Chat
response = await agent.chat("What is machine learning?")
print(response)
```

### 2. Agent with Custom Tools

```python
from src.core.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """Get current weather for a location"""
    # Your weather API implementation
    return f"Weather in {location}: Sunny, 72°F"

# Register tool
agent.tool_registry.register_tool(get_current_weather)

# Use tool
response = await agent.chat("What's the weather in New York?")
```

### 3. Streaming Responses

```python
# Enable streaming
config = AgentConfig(streaming=True)
agent = Agent(config, api_key)

# Stream response
async for chunk in agent.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### 4. Multi-Agent Coordination

```python
# Create specialized agents
researcher = Agent(research_config, api_key)
creative = Agent(creative_config, api_key)

# Coordinate tasks
research_result = await researcher.chat("Research topic X")
creative_result = await creative.chat(f"Create content based on: {research_result}")
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   ValueError: OpenAI API key is required
   ```
   **Solution**: Ensure your `.env` file contains a valid `OPENAI_API_KEY`

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   **Solution**: Run scripts from the project root directory

3. **Rate Limiting**
   ```
   RateLimitError: Rate limit exceeded
   ```
   **Solution**: Implement delays between requests or upgrade your OpenAI plan

4. **Memory Issues**
   ```
   Memory limit exceeded
   ```
   **Solution**: Adjust `max_messages` in memory configuration

### Debug Mode

Enable debug logging:

```python
from src.core.utils import setup_logging

setup_logging("DEBUG")
```

Or set in environment:
```env
APP_LOG_LEVEL=DEBUG
```

## 🔒 Security Considerations

- **API Keys**: Never commit API keys to version control
- **Input Validation**: Validate all user inputs before processing
- **Rate Limiting**: Implement appropriate rate limiting for production use
- **Error Handling**: Avoid exposing sensitive information in error messages
- **Memory Management**: Clear sensitive data from memory when appropriate

## 📈 Performance Tips

1. **Model Selection**: Use `gpt-3.5-turbo` for faster, cheaper responses
2. **Token Management**: Set appropriate `max_tokens` limits
3. **Streaming**: Use streaming for long responses to improve perceived performance
4. **Memory Limits**: Set reasonable memory limits to prevent excessive token usage
5. **Caching**: Implement response caching for repeated queries

## 🤝 Contributing

To extend this demo:

1. **Add New Examples**: Create new files in `src/examples/`
2. **Custom Tools**: Add tools in `src/core/tools.py`
3. **Memory Backends**: Extend `src/core/memory.py`
4. **Agent Configurations**: Modify `src/core/agent.py`

### Example Structure

```python
# src/examples/your_example.py
class YourExample:
    def __init__(self, api_key=None):
        # Initialize your example
        pass
    
    async def demo_feature(self):
        # Demonstrate specific feature
        pass
    
    async def run_all_demos(self):
        # Run all demonstrations
        pass

def main():
    # Main execution function
    pass

if __name__ == "__main__":
    main()
```

## 📄 License

This demo repository is provided as-is for educational and demonstration purposes. Please ensure compliance with OpenAI's usage policies and terms of service.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the example code for usage patterns
3. Ensure your environment is properly configured
4. Verify your OpenAI API key and quota

## 🔗 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Best Practices](https://platform.openai.com/docs/guides/production-best-practices)

---

**Happy coding with OpenAI Agents! 🚀**