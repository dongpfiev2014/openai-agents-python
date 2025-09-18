"""Multi-agent example demonstrating agent coordination and collaboration"""

import asyncio
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

from core.agent import Agent, AgentConfig
from core.tools import ToolRegistry, tool
from core.utils import load_config, setup_logging


class MultiAgentExample:
    """Demonstrates multi-agent coordination and collaboration"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        # Load configuration
        config_data = load_config()
        
        # Use provided API key or load from config
        self.api_key = api_key or config_data.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize different specialized agents
        self.agents = self._create_specialized_agents(model)
        
        # Shared communication channel
        self.communication_log = []
        
        # Task coordination
        self.task_queue = []
        self.completed_tasks = []
        
        print(f"✅ Multi-Agent System initialized with {len(self.agents)} specialized agents")
    
    def _create_specialized_agents(self, model: str) -> Dict[str, Agent]:
        """Create specialized agents for different roles"""
        agents = {}
        
        # Research Agent
        research_config = AgentConfig(
            model=model,
            temperature=0.3,
            max_tokens=2000,
            system_prompt="You are a Research Agent specialized in gathering and analyzing information. You provide detailed, accurate research on any topic. Focus on facts, data, and credible sources.",
            tools_enabled=True,
            memory_enabled=True
        )
        agents['researcher'] = Agent(research_config, self.api_key)
        
        # Creative Agent
        creative_config = AgentConfig(
            model=model,
            temperature=0.8,
            max_tokens=2000,
            system_prompt="You are a Creative Agent specialized in generating innovative ideas, creative content, and artistic solutions. Think outside the box and provide imaginative responses.",
            tools_enabled=True,
            memory_enabled=True
        )
        agents['creative'] = Agent(creative_config, self.api_key)
        
        # Technical Agent
        technical_config = AgentConfig(
            model=model,
            temperature=0.2,
            max_tokens=2000,
            system_prompt="You are a Technical Agent specialized in software development, system architecture, and technical problem-solving. Provide precise, implementable technical solutions.",
            tools_enabled=True,
            memory_enabled=True
        )
        agents['technical'] = Agent(technical_config, self.api_key)
        
        # Coordinator Agent
        coordinator_config = AgentConfig(
            model=model,
            temperature=0.5,
            max_tokens=1500,
            system_prompt="You are a Coordinator Agent responsible for managing tasks, delegating work to other agents, and synthesizing their outputs into cohesive results.",
            tools_enabled=True,
            memory_enabled=True
        )
        agents['coordinator'] = Agent(coordinator_config, self.api_key)
        
        # Quality Assurance Agent
        qa_config = AgentConfig(
            model=model,
            temperature=0.1,
            max_tokens=1500,
            system_prompt="You are a Quality Assurance Agent responsible for reviewing work, identifying issues, and ensuring high standards. Provide constructive feedback and validation.",
            tools_enabled=True,
            memory_enabled=True
        )
        agents['qa'] = Agent(qa_config, self.api_key)
        
        return agents
    
    def _setup_inter_agent_tools(self):
        """Setup tools for inter-agent communication"""
        
        @tool
        def send_message_to_agent(target_agent: str, message: str, sender: str = "unknown") -> str:
            """Send a message to another agent
            
            Args:
                target_agent: Name of the target agent (researcher, creative, technical, coordinator, qa)
                message: Message content to send
                sender: Name of the sending agent
            
            Returns:
                Confirmation of message delivery
            """
            if target_agent not in self.agents:
                return f"Error: Agent '{target_agent}' not found. Available agents: {list(self.agents.keys())}"
            
            # Log the communication
            comm_entry = {
                "timestamp": datetime.now().isoformat(),
                "from": sender,
                "to": target_agent,
                "message": message
            }
            self.communication_log.append(comm_entry)
            
            return f"Message sent to {target_agent}: '{message[:50]}...'"
        
        @tool
        def get_agent_response(agent_name: str, query: str, conversation_id: str = "multi_agent") -> str:
            """Get a response from another agent
            
            Args:
                agent_name: Name of the agent to query
                query: Question or task for the agent
                conversation_id: Conversation identifier
            
            Returns:
                Response from the queried agent
            """
            if agent_name not in self.agents:
                return f"Error: Agent '{agent_name}' not found"
            
            # Get response from the target agent
            try:
                response = asyncio.create_task(
                    self.agents[agent_name].chat(query, conversation_id)
                )
                return asyncio.get_event_loop().run_until_complete(response)
            except Exception as e:
                return f"Error getting response from {agent_name}: {str(e)}"
        
        @tool
        def create_task(task_description: str, assigned_agent: str, priority: str = "medium") -> str:
            """Create a new task and assign it to an agent
            
            Args:
                task_description: Description of the task
                assigned_agent: Agent responsible for the task
                priority: Task priority (low, medium, high)
            
            Returns:
                Task creation confirmation
            """
            task = {
                "id": len(self.task_queue) + 1,
                "description": task_description,
                "assigned_agent": assigned_agent,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            self.task_queue.append(task)
            return f"Task {task['id']} created and assigned to {assigned_agent}"
        
        @tool
        def get_communication_log() -> str:
            """Get the inter-agent communication log
            
            Returns:
                Formatted communication history
            """
            if not self.communication_log:
                return "No communications logged yet."
            
            log_summary = f"Communication Log ({len(self.communication_log)} entries):\n"
            for entry in self.communication_log[-10:]:  # Show last 10 entries
                timestamp = entry['timestamp'][:19]  # Remove microseconds
                log_summary += f"[{timestamp}] {entry['from']} → {entry['to']}: {entry['message'][:100]}...\n"
            
            return log_summary
        
        # Register tools for all agents
        for agent in self.agents.values():
            agent.tool_registry.register_tool(send_message_to_agent)
            agent.tool_registry.register_tool(get_agent_response)
            agent.tool_registry.register_tool(create_task)
            agent.tool_registry.register_tool(get_communication_log)
    
    async def demo_agent_collaboration(self) -> None:
        """Demonstrate agents working together on a complex task"""
        print("\n🤝 Agent Collaboration Demo")
        print("-" * 30)
        
        # Setup inter-agent communication tools
        self._setup_inter_agent_tools()
        
        # Complex task requiring multiple agents
        task = "Create a comprehensive plan for developing a sustainable smart city mobile application"
        
        print(f"📋 Complex Task: {task}")
        print("\n🎯 Coordinating agents...")
        
        # Step 1: Coordinator breaks down the task
        coordinator_prompt = f"""
        Task: {task}
        
        As the coordinator, break this down into subtasks and assign them to appropriate agents:
        - researcher: for market research and sustainability data
        - creative: for innovative features and user experience ideas
        - technical: for architecture and implementation details
        - qa: for testing strategy and quality requirements
        
        Use the create_task tool to assign specific subtasks to each agent.
        """
        
        print("\n👨‍💼 Coordinator planning...")
        coordinator_response = await self.agents['coordinator'].chat(
            coordinator_prompt, 
            "collaboration_demo"
        )
        print(f"📝 Coordinator: {coordinator_response}")
        
        # Step 2: Execute subtasks
        print("\n🔄 Executing subtasks...")
        
        # Research phase
        research_task = "Research current smart city initiatives, sustainability metrics, and mobile app market trends for smart city applications"
        print(f"\n🔬 Researcher working on: {research_task}")
        research_response = await self.agents['researcher'].chat(
            research_task,
            "collaboration_demo"
        )
        print(f"📊 Research findings: {research_response[:200]}...")
        
        # Creative phase
        creative_task = f"Based on the research findings, generate innovative features and user experience ideas for the smart city app. Research context: {research_response[:500]}"
        print(f"\n🎨 Creative agent working on features...")
        creative_response = await self.agents['creative'].chat(
            creative_task,
            "collaboration_demo"
        )
        print(f"💡 Creative ideas: {creative_response[:200]}...")
        
        # Technical phase
        technical_task = f"Design the technical architecture for the smart city app. Consider these features: {creative_response[:500]}"
        print(f"\n⚙️ Technical agent designing architecture...")
        technical_response = await self.agents['technical'].chat(
            technical_task,
            "collaboration_demo"
        )
        print(f"🏗️ Technical design: {technical_response[:200]}...")
        
        # QA phase
        qa_task = f"Review the proposed solution and create a quality assurance plan. Solution overview: Research: {research_response[:200]} Creative: {creative_response[:200]} Technical: {technical_response[:200]}"
        print(f"\n🔍 QA agent reviewing solution...")
        qa_response = await self.agents['qa'].chat(
            qa_task,
            "collaboration_demo"
        )
        print(f"✅ QA assessment: {qa_response[:200]}...")
        
        # Step 3: Coordinator synthesizes results
        synthesis_task = f"""
        Synthesize the work from all agents into a comprehensive project plan:
        
        Research: {research_response[:300]}
        Creative: {creative_response[:300]}
        Technical: {technical_response[:300]}
        QA: {qa_response[:300]}
        
        Create a final executive summary and project roadmap.
        """
        
        print(f"\n📋 Coordinator synthesizing results...")
        final_plan = await self.agents['coordinator'].chat(
            synthesis_task,
            "collaboration_demo"
        )
        
        print(f"\n🎯 Final Project Plan:\n{final_plan}")
        
        # Show communication statistics
        print(f"\n📊 Collaboration Statistics:")
        print(f"  Tasks created: {len(self.task_queue)}")
        print(f"  Communications: {len(self.communication_log)}")
        print(f"  Agents involved: {len(self.agents)}")
    
    async def demo_agent_debate(self) -> None:
        """Demonstrate agents having a structured debate"""
        print("\n🗣️ Agent Debate Demo")
        print("-" * 25)
        
        debate_topic = "Should AI development prioritize safety or innovation speed?"
        print(f"🎯 Debate Topic: {debate_topic}")
        
        # Assign positions
        positions = {
            'researcher': 'Present factual evidence and data on both sides',
            'creative': 'Argue for innovation speed with creative solutions',
            'technical': 'Argue for safety-first approach with technical reasoning',
            'coordinator': 'Moderate the debate and summarize key points'
        }
        
        print("\n📋 Debate Positions:")
        for agent, position in positions.items():
            print(f"  {agent}: {position}")
        
        # Round 1: Opening statements
        print("\n🎤 Round 1: Opening Statements")
        print("-" * 35)
        
        opening_responses = {}
        
        for agent_name, position in positions.items():
            if agent_name == 'coordinator':
                continue
                
            prompt = f"""
            Debate Topic: {debate_topic}
            Your Position: {position}
            
            Provide your opening statement (2-3 key points, keep it concise).
            """
            
            print(f"\n{agent_name.title()} opening statement:")
            response = await self.agents[agent_name].chat(prompt, f"debate_{agent_name}")
            opening_responses[agent_name] = response
            print(f"💬 {response[:300]}...")
        
        # Round 2: Rebuttals
        print("\n🔄 Round 2: Rebuttals")
        print("-" * 25)
        
        # Creative responds to Technical
        creative_rebuttal_prompt = f"""
        The technical agent argued: {opening_responses['technical'][:200]}
        
        Provide a rebuttal defending innovation speed. Address their safety concerns while emphasizing the benefits of rapid AI development.
        """
        
        print("\nCreative agent rebuttal:")
        creative_rebuttal = await self.agents['creative'].chat(
            creative_rebuttal_prompt, 
            "debate_creative"
        )
        print(f"💬 {creative_rebuttal[:300]}...")
        
        # Technical responds to Creative
        technical_rebuttal_prompt = f"""
        The creative agent argued: {opening_responses['creative'][:200]}
        And then responded: {creative_rebuttal[:200]}
        
        Provide a rebuttal defending the safety-first approach. Address their innovation arguments while emphasizing the risks of moving too fast.
        """
        
        print("\nTechnical agent rebuttal:")
        technical_rebuttal = await self.agents['technical'].chat(
            technical_rebuttal_prompt, 
            "debate_technical"
        )
        print(f"💬 {technical_rebuttal[:300]}...")
        
        # Round 3: Coordinator summary
        print("\n📊 Round 3: Moderator Summary")
        print("-" * 35)
        
        summary_prompt = f"""
        As the debate moderator, summarize this debate on: {debate_topic}
        
        Research Evidence: {opening_responses['researcher'][:200]}
        Innovation Position: {opening_responses['creative'][:200]} + {creative_rebuttal[:200]}
        Safety Position: {opening_responses['technical'][:200]} + {technical_rebuttal[:200]}
        
        Provide:
        1. Key points from each side
        2. Areas of agreement
        3. Remaining disagreements
        4. Potential compromise solutions
        """
        
        coordinator_summary = await self.agents['coordinator'].chat(
            summary_prompt, 
            "debate_coordinator"
        )
        
        print(f"\n🎯 Moderator Summary:\n{coordinator_summary}")
        
        print(f"\n📊 Debate Statistics:")
        print(f"  Participants: {len(positions)}")
        print(f"  Rounds completed: 3")
        print(f"  Total responses: {len(opening_responses) + 2 + 1}")
    
    async def demo_agent_workflow(self) -> None:
        """Demonstrate a structured workflow with multiple agents"""
        print("\n⚙️ Agent Workflow Demo")
        print("-" * 25)
        
        # Setup workflow tools
        self._setup_inter_agent_tools()
        
        workflow_task = "Design and validate a new feature for an e-commerce platform: AI-powered personal shopping assistant"
        
        print(f"📋 Workflow Task: {workflow_task}")
        
        # Stage 1: Requirements gathering (Researcher)
        print("\n📋 Stage 1: Requirements Gathering")
        requirements_prompt = f"""
        Task: {workflow_task}
        
        Research and define:
        1. User requirements and pain points
        2. Market analysis of existing solutions
        3. Technical feasibility considerations
        4. Success metrics
        
        Provide a structured requirements document.
        """
        
        requirements = await self.agents['researcher'].chat(
            requirements_prompt,
            "workflow_requirements"
        )
        print(f"📄 Requirements: {requirements[:300]}...")
        
        # Stage 2: Solution design (Creative + Technical)
        print("\n🎨 Stage 2: Solution Design")
        
        # Creative design
        creative_design_prompt = f"""
        Based on these requirements: {requirements[:500]}
        
        Design the user experience and innovative features for the AI-powered personal shopping assistant.
        Focus on user interface, interaction flows, and unique value propositions.
        """
        
        creative_design = await self.agents['creative'].chat(
            creative_design_prompt,
            "workflow_design"
        )
        print(f"🎨 Creative Design: {creative_design[:300]}...")
        
        # Technical architecture
        technical_design_prompt = f"""
        Requirements: {requirements[:300]}
        Creative Design: {creative_design[:300]}
        
        Design the technical architecture, including:
        1. System components and APIs
        2. AI/ML model requirements
        3. Data flow and storage
        4. Integration points
        5. Scalability considerations
        """
        
        technical_design = await self.agents['technical'].chat(
            technical_design_prompt,
            "workflow_architecture"
        )
        print(f"⚙️ Technical Design: {technical_design[:300]}...")
        
        # Stage 3: Quality review (QA)
        print("\n🔍 Stage 3: Quality Review")
        
        qa_review_prompt = f"""
        Review this complete solution design:
        
        Requirements: {requirements[:200]}
        Creative Design: {creative_design[:200]}
        Technical Design: {technical_design[:200]}
        
        Evaluate:
        1. Requirements coverage
        2. Design consistency
        3. Technical feasibility
        4. Potential risks and mitigation
        5. Testing strategy
        
        Provide approval status and recommendations.
        """
        
        qa_review = await self.agents['qa'].chat(
            qa_review_prompt,
            "workflow_qa"
        )
        print(f"✅ QA Review: {qa_review[:300]}...")
        
        # Stage 4: Final coordination (Coordinator)
        print("\n📊 Stage 4: Project Coordination")
        
        coordination_prompt = f"""
        Synthesize the complete workflow results:
        
        Requirements: {requirements[:200]}
        Creative: {creative_design[:200]}
        Technical: {technical_design[:200]}
        QA Review: {qa_review[:200]}
        
        Create:
        1. Executive summary
        2. Implementation roadmap
        3. Resource requirements
        4. Timeline estimates
        5. Next steps
        """
        
        final_coordination = await self.agents['coordinator'].chat(
            coordination_prompt,
            "workflow_final"
        )
        
        print(f"\n🎯 Final Coordination:\n{final_coordination}")
        
        # Workflow statistics
        print(f"\n📊 Workflow Statistics:")
        print(f"  Stages completed: 4")
        print(f"  Agents involved: {len(self.agents)}")
        print(f"  Total processing time: ~{len(requirements) + len(creative_design) + len(technical_design) + len(qa_review) + len(final_coordination)} characters processed")
    
    async def run_all_demos(self) -> None:
        """Run all multi-agent demonstrations"""
        print("\n🚀 Running All Multi-Agent Demos")
        print("=" * 50)
        
        # Run individual demos
        await self.demo_agent_collaboration()
        
        print("\n" + "=" * 50)
        await self.demo_agent_debate()
        
        print("\n" + "=" * 50)
        await self.demo_agent_workflow()
        
        # Show final statistics for all agents
        print("\n📊 Final Multi-Agent Statistics:")
        for agent_name, agent in self.agents.items():
            stats = agent.get_stats()
            print(f"\n{agent_name.title()} Agent:")
            for key, value in stats.items():
                print(f"  {key}: {value}")


def main():
    """Main function to run the multi-agent example"""
    # Setup logging
    setup_logging("INFO")
    
    try:
        # Initialize the multi-agent example
        multi_agent_example = MultiAgentExample()
        
        print("\n🚀 OpenAI Agents SDK - Multi-Agent Demo")
        print("=" * 50)
        
        # Run all demos
        asyncio.run(multi_agent_example.run_all_demos())
        
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