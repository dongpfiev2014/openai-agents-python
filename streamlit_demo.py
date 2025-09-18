#!/usr/bin/env python3
"""Streamlit Web Interface for OpenAI Agents SDK Demo"""

import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent
from core.tools import tool
from core.memory import ConversationMemory
from core.utils import load_config


# Page configuration
st.set_page_config(
    page_title="OpenAI Agents SDK Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.demo-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}

.success-message {
    background: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #28a745;
}

.error-message {
    background: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #dc3545;
}

.info-message {
    background: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'demo_results' not in st.session_state:
    st.session_state.demo_results = {}


def create_demo_tools():
    """Create demonstration tools"""
    
    @tool("get_current_time", "Get the current date and time")
    def get_current_time() -> str:
        """Returns the current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @tool("calculate", "Perform basic mathematical calculations")
    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Simple safety check
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool("text_analyzer", "Analyze text properties")
    def text_analyzer(text: str) -> str:
        """Analyze text and return statistics"""
        words = text.split()
        chars = len(text)
        chars_no_spaces = len(text.replace(' ', ''))
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        return f"""
        Text Analysis Results:
        - Characters: {chars}
        - Characters (no spaces): {chars_no_spaces}
        - Words: {len(words)}
        - Sentences: {sentences}
        - Average word length: {chars_no_spaces / len(words) if words else 0:.1f}
        """
    
    @tool("word_reverser", "Reverse words or sentences")
    def word_reverser(text: str, mode: str = "words") -> str:
        """Reverse text by words or characters"""
        if mode == "words":
            return " ".join(text.split()[::-1])
        elif mode == "characters":
            return text[::-1]
        else:
            return "Error: Mode must be 'words' or 'characters'"
    
    return [get_current_time, calculate, text_analyzer, word_reverser]


def get_or_create_agent(agent_name: str, api_key: str, model: str) -> Agent:
    """Get existing agent or create new one"""
    agent_key = f"{agent_name}_{model}"
    
    if agent_key not in st.session_state.agents:
        from core.agent import AgentConfig
        config = AgentConfig(
            model=model,
            system_prompt=f"You are {agent_name}, a helpful AI assistant. {f'Demo agent: {agent_name}'}"
        )
        agent = Agent(
            config=config,
            api_key=api_key
        )
        
        # Add demo tools
        demo_functions = create_demo_tools()
        for func in demo_functions:
            if hasattr(func, '_is_tool'):
                from core.tools import Tool
                tool = Tool.from_function(
                    func,
                    getattr(func, '_tool_name', func.__name__),
                    getattr(func, '_tool_description', func.__doc__)
                )
                agent.tool_registry.register(tool)
        
        st.session_state.agents[agent_key] = agent
    
    return st.session_state.agents[agent_key]


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 OpenAI Agents SDK Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key",
        value=os.getenv('OPENAI_API_KEY', '')
    )
    
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key to continue")
        st.info("👈 Please enter your OpenAI API key in the sidebar to start using the demo.")
        return
    
    # Model selection
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o", "gpt-4", "gpt-4-turbo-preview"],
        index=0,
        help="Select the OpenAI model to use (gpt-4o-mini is cheapest - 60% cheaper than gpt-3.5-turbo)"
    )
    
    # Demo mode selection
    demo_mode = st.sidebar.radio(
        "Demo Mode",
        ["💬 Interactive Chat", "🔧 Function Calling", "🌊 Streaming Chat", "🧠 Memory Demo", "🤝 Multi-Agent"]
    )
    
    # Main content area
    if demo_mode == "💬 Interactive Chat":
        interactive_chat_demo(api_key, model)
    elif demo_mode == "🔧 Function Calling":
        function_calling_demo(api_key, model)
    elif demo_mode == "🌊 Streaming Chat":
        streaming_chat_demo(api_key, model)
    elif demo_mode == "🧠 Memory Demo":
        memory_demo(api_key, model)
    elif demo_mode == "🤝 Multi-Agent":
        multi_agent_demo(api_key, model)


def interactive_chat_demo(api_key: str, model: str):
    """Interactive chat demonstration"""
    st.header("💬 Interactive Chat Demo")
    
    st.markdown("""
    <div class="demo-card">
    <h4>🎯 What this demo shows:</h4>
    <ul>
        <li>Basic conversation with AI agent</li>
        <li>Context awareness within conversation</li>
        <li>Natural language understanding</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get or create agent
    agent = get_or_create_agent("ChatAgent", api_key, model)
    
    # Chat interface
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call synchronous chat method directly
                    response = agent.chat(prompt)
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()


def function_calling_demo(api_key: str, model: str):
    """Function calling demonstration"""
    st.header("🔧 Function Calling Demo")
    
    st.markdown("""
    <div class="demo-card">
    <h4>🎯 What this demo shows:</h4>
    <ul>
        <li>AI agent using custom tools and functions</li>
        <li>Automatic function selection based on user input</li>
        <li>Tool execution and result integration</li>
    </ul>
    <h4>🛠️ Available Tools:</h4>
    <ul>
        <li><strong>get_current_time</strong>: Get current date and time</li>
        <li><strong>calculate</strong>: Perform mathematical calculations</li>
        <li><strong>text_analyzer</strong>: Analyze text properties</li>
        <li><strong>word_reverser</strong>: Reverse words or characters</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get or create agent
    agent = get_or_create_agent("FunctionAgent", api_key, model)
    
    # Predefined examples
    st.subheader("🎯 Try these examples:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⏰ What time is it?"):
            run_function_example(agent, "What time is it right now?")
        
        if st.button("🧮 Calculate 15 * 23 + 7"):
            run_function_example(agent, "Calculate 15 * 23 + 7")
    
    with col2:
        if st.button("📊 Analyze text"):
            run_function_example(agent, "Analyze this text: 'Hello world! This is a sample text for analysis.'")
        
        if st.button("🔄 Reverse words"):
            run_function_example(agent, "Reverse the words in this sentence: 'OpenAI Agents SDK is awesome'")
    
    # Custom input
    st.subheader("💬 Custom Function Call")
    custom_prompt = st.text_input(
        "Enter your request:",
        placeholder="Ask the agent to use any of the available tools..."
    )
    
    if st.button("🚀 Execute") and custom_prompt:
        run_function_example(agent, custom_prompt)


def run_function_example(agent: Agent, prompt: str):
    """Run a function calling example"""
    with st.spinner("Executing..."):
        try:
            # Call synchronous chat method directly
            response = agent.chat(prompt)
            st.success(f"**Request:** {prompt}")
            st.info(f"**Response:** {response}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def streaming_chat_demo(api_key: str, model: str):
    """Streaming chat demonstration"""
    st.header("🌊 Streaming Chat Demo")
    
    st.markdown("""
    <div class="demo-card">
    <h4>🎯 What this demo shows:</h4>
    <ul>
        <li>Real-time streaming responses from AI agent</li>
        <li>Progressive text generation</li>
        <li>Better user experience for long responses</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get or create agent
    agent = get_or_create_agent("StreamAgent", api_key, model)
    
    # Streaming examples
    st.subheader("🎯 Try these streaming examples:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Write a short story"):
            run_streaming_example(agent, "Write a short story about a robot learning to paint")
        
        if st.button("🔢 Count to 10"):
            run_streaming_example(agent, "Count from 1 to 10, explaining each number")
    
    with col2:
        if st.button("🌟 Explain AI"):
            run_streaming_example(agent, "Explain artificial intelligence in simple terms")
        
        if st.button("🎵 Write a poem"):
            run_streaming_example(agent, "Write a poem about technology and humanity")
    
    # Custom streaming input
    st.subheader("💬 Custom Streaming Request")
    streaming_prompt = st.text_area(
        "Enter your request for streaming response:",
        placeholder="Ask for something that would benefit from streaming (long explanations, stories, etc.)..."
    )
    
    if st.button("🌊 Stream Response") and streaming_prompt:
        run_streaming_example(agent, streaming_prompt)


def run_streaming_example(agent: Agent, prompt: str):
    """Run a streaming example"""
    st.success(f"**Request:** {prompt}")
    
    # Create placeholder for streaming content
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        async def stream_response():
            nonlocal full_response
            async for chunk in agent.chat_stream(prompt):
                full_response += chunk
                response_placeholder.info(f"**Streaming Response:**\n\n{full_response}▋")
            
            # Final response without cursor
            response_placeholder.info(f"**Final Response:**\n\n{full_response}")
        
        # Use asyncio.get_event_loop() instead of asyncio.run() for Streamlit compatibility
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(stream_response())
        finally:
            loop.close()
        
    except Exception as e:
        st.error(f"Streaming error: {str(e)}")


def memory_demo(api_key: str, model: str):
    """Memory demonstration"""
    st.header("🧠 Memory Demo")
    
    st.markdown("""
    <div class="demo-card">
    <h4>🎯 What this demo shows:</h4>
    <ul>
        <li>Conversation memory across multiple interactions</li>
        <li>Context retention within sessions</li>
        <li>Session-based conversation management</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Session selection
    session_id = st.selectbox(
        "Select Session:",
        ["session_1", "session_2", "session_3", "new_session"],
        help="Each session maintains separate conversation memory"
    )
    
    if session_id == "new_session":
        session_id = st.text_input("Enter new session name:", "custom_session")
    
    # Get or create agent
    agent = get_or_create_agent("MemoryAgent", api_key, model)
    
    # Memory examples
    st.subheader("🎯 Try these memory examples:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Set preference"):
            run_memory_example(agent, "Remember that my favorite color is blue and I like pizza", session_id)
        
        if st.button("❓ Ask about preference"):
            run_memory_example(agent, "What do you remember about my preferences?", session_id)
    
    with col2:
        if st.button("🏠 Share location"):
            run_memory_example(agent, "I live in San Francisco and work as a software engineer", session_id)
        
        if st.button("🔍 Ask about me"):
            run_memory_example(agent, "Tell me what you know about me", session_id)
    
    # Custom memory input
    st.subheader("💬 Custom Memory Interaction")
    memory_prompt = st.text_input(
        "Enter your message:",
        placeholder="Share information or ask about previous conversations..."
    )
    
    if st.button("🧠 Send with Memory") and memory_prompt:
        run_memory_example(agent, memory_prompt, session_id)
    
    # Show memory contents
    if st.button("📋 Show Session Memory"):
        show_session_memory(agent, session_id)


def run_memory_example(agent: Agent, prompt: str, session_id: str):
    """Run a memory example"""
    with st.spinner("Processing with memory..."):
        try:
            # Call synchronous chat method directly
            response = agent.chat(prompt, session_id)
            st.success(f"**Session:** {session_id}")
            st.success(f"**Request:** {prompt}")
            st.info(f"**Response:** {response}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def show_session_memory(agent: Agent, session_id: str):
    """Show session memory contents"""
    try:
        history = agent.memory.get_history(session_id)
        if history:
            st.subheader(f"📋 Memory for session: {session_id}")
            for i, entry in enumerate(history):
                with st.expander(f"{entry.role.title()} - Message {i+1}"):
                    st.write(f"**Content:** {entry.content}")
                    st.write(f"**Timestamp:** {entry.timestamp}")
        else:
            st.info(f"No memory found for session: {session_id}")
    except Exception as e:
        st.error(f"Error retrieving memory: {str(e)}")


def multi_agent_demo(api_key: str, model: str):
    """Multi-agent demonstration"""
    st.header("🤝 Multi-Agent Demo")
    
    st.markdown("""
    <div class="demo-card">
    <h4>🎯 What this demo shows:</h4>
    <ul>
        <li>Multiple specialized AI agents working together</li>
        <li>Agent-to-agent communication</li>
        <li>Collaborative problem solving</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create specialized agents
    researcher = get_or_create_agent("Researcher", api_key, model)
    creative = get_or_create_agent("Creative", api_key, model)
    technical = get_or_create_agent("Technical", api_key, model)
    
    # Multi-agent examples
    st.subheader("🎯 Try these multi-agent scenarios:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 Research Project"):
            run_multi_agent_example(
                [researcher, creative, technical],
                "Create a plan for developing a mobile app for plant care",
                ["Research market and user needs", "Design creative features", "Plan technical implementation"]
            )
        
        if st.button("📚 Content Creation"):
            run_multi_agent_example(
                [researcher, creative, technical],
                "Create educational content about renewable energy",
                ["Research renewable energy facts", "Create engaging content structure", "Suggest technical diagrams"]
            )
    
    with col2:
        if st.button("🚀 Product Launch"):
            run_multi_agent_example(
                [researcher, creative, technical],
                "Plan the launch of a new AI-powered fitness app",
                ["Research target audience", "Design marketing campaign", "Plan technical rollout"]
            )
        
        if st.button("🎯 Problem Solving"):
            run_multi_agent_example(
                [researcher, creative, technical],
                "Solve the problem of food waste in restaurants",
                ["Research the problem scope", "Brainstorm creative solutions", "Design technical implementation"]
            )
    
    # Custom multi-agent input
    st.subheader("💬 Custom Multi-Agent Task")
    multi_agent_prompt = st.text_area(
        "Enter a complex task for the agents to collaborate on:",
        placeholder="Describe a problem or project that would benefit from multiple perspectives..."
    )
    
    if st.button("🤝 Collaborate") and multi_agent_prompt:
        run_multi_agent_example(
            [researcher, creative, technical],
            multi_agent_prompt,
            ["Research and analyze", "Creative ideation", "Technical planning"]
        )


def run_multi_agent_example(agents: list, task: str, subtasks: list):
    """Run a multi-agent example"""
    st.success(f"**Collaborative Task:** {task}")
    
    results = []
    
    for i, (agent, subtask) in enumerate(zip(agents, subtasks)):
        agent_name = f"Agent {i+1}"
        with st.spinner(f"{agent_name} working on: {subtask}"):
            try:
                # Create specific prompt for each agent
                prompt = f"Task: {task}\n\nYour role: {subtask}\n\nProvide your perspective and recommendations."
                # Call synchronous chat method directly
                response = agent.chat(prompt)
                results.append((agent_name, subtask, response))
            except Exception as e:
                results.append((agent_name, subtask, f"Error: {str(e)}"))
    
    # Display results
    st.subheader("🤝 Collaboration Results")
    
    for agent_name, subtask, response in results:
        with st.expander(f"🤖 {agent_name}: {subtask}"):
            st.write(response)
    
    # Generate synthesis
    if len(results) > 1:
        with st.spinner("Synthesizing results..."):
            try:
                synthesis_prompt = f"""
                Task: {task}
                
                Here are the perspectives from different agents:
                
                {chr(10).join([f"{name}: {response}" for name, _, response in results])}
                
                Please synthesize these perspectives into a comprehensive solution or plan.
                """
                
                # Call synchronous chat method directly
                synthesis = agents[0].chat(synthesis_prompt)
                
                st.subheader("🎯 Synthesized Solution")
                st.info(synthesis)
                
            except Exception as e:
                st.error(f"Synthesis error: {str(e)}")


if __name__ == "__main__":
    main()