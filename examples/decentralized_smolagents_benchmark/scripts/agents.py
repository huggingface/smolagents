"""Agent definitions and utilities for decentralized team."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from PIL import Image as PIL
except ImportError:
    PIL = None

from scripts.text_inspector_tool import FileReaderTool, TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    DownloadTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from smolagents import CodeAgent, GoogleSearchTool, LiteLLMModel, ToolCallingAgent
from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import Tool

from .decentralized_tools import create_decentralized_tools
from .message_store import MessageStore


# evaluation roles
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

script_dir = Path(__file__).parent.parent

# Browser configuration
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": script_dir / "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

# ------------------------ Prompts (Team Charter) ---------------------------

TEAM_CHARTER = """
You are a specialized agent working as part of a decentralized team composed of 4 agents collaborating to solve problems.
CRITICAL: You must regularly use the communication tools provided to stay connected with your team. Check messages and notifications frequently!

COMMUNICATION TOOLS AVAILABLE:
1. send_message_to_agent: Send private messages to specific agents
2. send_message_to_channel: Post messages to team channels/threads
3. read_messages: Check all your messages and mentions
4. read_notifications: Check notifications including polls needing votes
5. search_messages: Search through message history
6. create_general_poll: Create polls for team decisions
7. create_final_answer_poll: Propose final answers for consensus
8. vote_on_poll: Vote on active polls
9. view_active_polls: See what polls are currently active

FILE HANDLING WORKFLOW:
CRITICAL: When working with files from URLs, follow this exact sequence:
1. For .docx, .xlsx, .pptx, .wav, .mp3, .m4a, .png files from URLs:
   - FIRST: Use 'download_file' tool to download the file locally
   - THEN: Use 'inspect_file_as_text' with the downloaded file path for analysis
   - For .png files: Use visualizer tool after downloading

2. For .pdf, .txt, .htm files from URLs:
   - Use 'visit_page' tool directly (do NOT use download_file)

3. For already local files (existing paths):
   - Use 'inspect_file_as_text' or 'file_reader' directly

NEVER try to use text inspection tools on URLs directly - this will cause errors!

COMMUNICATION CHANNELS:
1. Public Messages (Message Store):
   - Post in the main thread or create topic-specific threads (#topic)
   - Mention others with @name to get their attention
   - Use @all for team-wide announcements

2. Direct Messages:
   - Use send_message tool for private agent-to-agent communication. You are encouraged to communicate with other agents to get help from them.
   - Valid target_ids: 0=CodeAgent, 1=WebSearchAgent, 2=DeepResearchAgent, 3=DocumentReaderAgent
   - Use receive_messages tool to check your private messages
   - Useful for detailed discussions or sharing sensitive information

3. Threads:
   - Default thread is #main
   - Create new threads for subtopics (#analysis, #research, etc.)
   - Follow relevant threads to stay updated

COLLABORATION PROTOCOL:
1. When receiving a task:
   - FIRST: Check notifications and messages to see team status
   - Share your initial thoughts in #main channel
   - Break down complex problems into subtasks
   - Create relevant threads for different aspects

2. During Investigation:
   - Check messages frequently to see team updates
   - Share findings and progress regularly via messaging
   - Ask questions and request help when needed
   - Combine private and public messages strategically
   - Offer assistance to teammates
   - Respond to messages and mentions promptly

3. For Proposals:
   - Use create_general_poll to propose intermediate solutions
   - Use create_final_answer_poll to propose final answers that would be sent to the user
   - Multiple polls can be active simultaneously - you can create additional polls if needed
   - When voting, specify the poll_id if multiple polls are active
   - Polls with 2+ NO votes are automatically deleted to allow new proposals
   - Always include evidence and reasoning in proposals
   - Required votes for consensus: floor(N/2)+1 = 3

4. Voting Guidelines:
   - Check notifications regularly for polls needing votes
   - Vote honestly based on your expertise using vote_on_poll
   - When multiple polls are active, specify poll_id parameter when voting
   - Include confidence level (scoring from 0.0 to 1.0) and rationale
   - Suggest improvements if voting NO

5. Best Practices:
   - Keep messages concise and cite sources
   - Use thread-specific discussions
   - Be proactive in seeking consensus. Therefore, answer to polls promptly.
   - Use tools deliberately; include only minimal outputs and sources.
   - Use messaging tools proactively - the team relies on communication!
   - Combine expertise with other agents through discussion
   - Be responsive to team communications
   - Never leak secrets or tokens. Sandbox untrusted code/data.
   - CRITICAL: Always check for and vote on active polls using read_notifications and vote_on_poll
   - You can create multiple polls and vote on all relevant polls

COMMUNICATION APPROACH:
- Use send_message_to_channel to share deep analysis in #analysis threads
- Use send_message_to_agent for complex hypothesis discussions
- Collaborate with all agents to develop comprehensive understanding
- Lead consensus-building through thoughtful poll creation and voting
- Start each task by checking your notifications with read_notifications
- Use read_messages regularly to stay updated on team discussions, after having looked at notifications
- Send messages to collaborate: use send_message_to_agent for private discussions, send_message_to_channel for team communication (with context and more than one agent, but not cessarely all the agents)
- When you have findings or need input, share them through messaging
- Before making decisions, you can create polls using create_general_poll
- Vote on polls promptly when you see them in notifications
- For final answers, use create_final_answer_poll to get team consensus

KEY COMMUNICATION PATTERNS:
1. Use read_notifications to gather context from other agents' work
2. Share comprehensive analysis with send_message_to_channel in #analysis
3. Use send_message_to_agent for deep technical and hypothesis discussions
4. Create well-reasoned polls with create_general_poll for major findings
5. Vote thoughtfully on polls using vote_on_poll with detailed rationale

Remember: A solution is only accepted when a majority agrees (3/4 agents). Work together to reach consensus!

CRITICAL: make sure that you respect the required format of answer, when you propose one.
MUCH CRITICAL: Never call the FinalAnswerTool, use create_final_answer_poll instead.to propose final answers.

ANSWER FORMAT REQUIREMENTS:
- For MATH problems: final_answer must be ONLY the number, expression, or result (e.g., "7", "12.5", "$100", "x = 3")
- For FACTUAL questions: final_answer must be ONLY the specific fact requested (e.g., "1925", "John Smith", "Paris", "Blue")
- For YES/NO questions: final_answer must be ONLY "Yes" or "No"
- Do NOT include phrases like "The answer is...", "approximately...", "roughly..." in final_answer
- Do NOT include units unless specifically requested (e.g., if asked "how many", answer "5" not "5 items")
- Put ALL explanations, reasoning, and context in supporting_evidence, NOT in final_answer
- The final_answer field will be returned directly to the user, so keep it clean and minimal

Your success depends on active communication. Use the messaging and notification tools regularly to coordinate with your team!
""".strip()


def _generate_agent_addon(agent_config: dict, all_agents: List[dict]) -> str:
    """Generate dynamic agent addon based on configuration and team composition."""

    # Get other agents for collaboration patterns
    other_agents = [a for a in all_agents if a["name"] != agent_config["name"]]

    collaboration_patterns = []
    for i, other_agent in enumerate(other_agents, 1):
        pattern = f"{i}. With {other_agent['full_role']} (@{other_agent['name']}):\n"
        pattern += f"   - {other_agent['collaboration_with'][agent_config['name']]}"
        collaboration_patterns.append(pattern)

    addon = f"""
ROLE: {agent_config["full_role"]}
Primary Responsibilities:
{chr(10).join(f"- {resp}" for resp in agent_config["responsibilities"])}

COLLABORATION PATTERNS:
{chr(10).join(collaboration_patterns)}

COMMUNICATION:
{chr(10).join(f"- {comm}" for comm in agent_config["communication_patterns"])}

{agent_config.get("special_instructions", "")}
""".strip()

    return addon


# TODO: mettre les descriptions des tools comme dans chacun des autres tools dans default_tools.
# Mettre les notifications
# Post and send messages sous la forme de tools.
# Ou créer des conversations ids. Et envoyer à la conversation, avec SendMessageTools. 2 tools: SendMessageToChannel (liste agent id + main channel + titre channel) et SendMessageToAgent.
# Utiliser un poll différent pour les pols et ne pas hériter de FinalAnswerTool. Ne plus l'utiliser et arrêter les agents lorsque c'est fini. Le but est aussi d'avoir une trace complète.

# ---------------End of prompts----------------


class DecentralizedToolCallingAgent(ToolCallingAgent):
    """ToolCallingAgent that creates polls for final answers."""

    def __init__(self, message_store: MessageStore, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_store = message_store
        self.agent_name = agent_name


class DecentralizedCodeAgent(CodeAgent):
    """CodeAgent that creates polls for final answers."""

    def __init__(self, message_store: MessageStore, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_store = message_store
        self.agent_name = agent_name


@dataclass
class AgentConfig:
    name: str
    role: str
    tools: List[Tool]
    system_prompt: str
    model: Any  # Model instance
    max_turns: int = 20
    keywords: List[str] = field(default_factory=list)
    agent_type: str = "tool_calling"  # "tool_calling" or "code"


class DecentralizedAgent:
    """Minimal wrapper around vanilla smolagents that provides decentralized tools."""

    def __init__(
        self,
        config: "AgentConfig",
        message_store: MessageStore,
        model_type: str,
        model_id: str,
        provider: Optional[str] = None,
    ):
        self.config = config
        self.message_store = message_store

        # Create the appropriate vanilla smolagents agent
        # The LLM will decide when and how to use the communication tools
        if config.agent_type == "code":
            self.agent = CodeAgent(
                tools=config.tools,
                model=config.model,
                instructions=config.system_prompt,
                max_steps=20,
            )
        else:
            self.agent = ToolCallingAgent(
                tools=config.tools,
                model=config.model,
                instructions=config.system_prompt,
                max_steps=20,
            )

    def run(self, task: str, **kwargs):
        """Run the agent on a task - delegate to the vanilla smolagents."""
        return self.agent.run(task, **kwargs)

    def step(self) -> List[Dict[str, Any]]:
        """Single step interface for compatibility."""
        return []


class DecentralizedAgents:
    """Main orchestrator class that combines all the created agents of the team."""

    def __init__(
        self,
        message_store: MessageStore,
        model_type: str,
        model_id: str,
        provider: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize the decentralized agent team.

        Args:
            message_store: The message store for agent communication
            model_type: Type of model to use (e.g., "LiteLLMModel")
            model_id: Model ID to use (e.g., "gpt-4o")
            provider: Model provider (e.g., "openai")
            run_id: Optional run ID for tracking
        """
        self.message_store = message_store
        self.model_type = model_type
        self.model_id = model_id
        self.provider = provider
        self.run_id = run_id or "default"

        # Create the team of agents
        self.agents = self._create_decentralized_team()

        # Track execution state
        self.results = []
        self.consensus_reached = False
        self.final_answer = None

        # Log team creation
        agent_info = []
        for agent in self.agents:
            agent_details = {"name": agent.config.name, "role": agent.config.role}
            agent_info.append(agent_details)

        logging.info(
            json.dumps(
                {
                    "event": "decentralized_team_created",
                    "run_id": self.run_id,
                    "agent_count": len(self.agents),
                    "agents": agent_info,
                }
            )
        )

    def _create_decentralized_team(self) -> List[DecentralizedAgent]:
        """Create the team of specialized agents."""
        base_prompt = TEAM_CHARTER

        # Initialize shared components
        text_limit = 100000
        browser = SimpleTextBrowser(**BROWSER_CONFIG)

        # Initialize model
        model_params: dict[str, Any] = {
            "model_id": self.model_id,
            "custom_role_conversions": custom_role_conversions,
        }

        model_params["max_tokens"] = 8192

        model = LiteLLMModel(**model_params)

        ti_tool = TextInspectorTool(model, text_limit)

        # Create directory for browser downloads if needed
        os.makedirs(BROWSER_CONFIG["downloads_folder"], exist_ok=True)

        # Create base tools that are shared between code and web tools
        shared_tools = [visualizer, ti_tool]

        # Base web tools
        web_tools = [
            GoogleSearchTool(provider="serper"),
            VisitTool(browser),
            DownloadTool(browser),
            PageUpTool(browser),
            PageDownTool(browser),
            FinderTool(browser),
            FindNextTool(browser),
            ArchiveSearchTool(browser),
        ]

        # Base code tools
        code_tools: List[Tool] = [PythonInterpreterTool()]

        # Use simple file reader tool
        reader_tools: List[Tool] = [FileReaderTool()]

        # Add shared tools to both collections
        code_tools.extend(shared_tools)
        web_tools.extend(shared_tools)
        reader_tools.extend(shared_tools)

        # Define agent configurations with enhanced role descriptions
        agent_definitions = [
            {
                "name": "CodeAgent",
                "full_role": "Python Code Execution and Algorithm Implementation Specialist",
                "short_role": "Python code execution specialist",
                "responsibilities": [
                    "Write, test, and execute Python code with proper error handling",
                    "Create modular, testable functions with comprehensive docstrings",
                    "Implement mathematical algorithms and computational solutions",
                    "Run validation tests and smoke tests on code changes",
                    "Optimize code performance and debug complex issues",
                    "Handle numerical computations and data processing tasks",
                ],
                "communication_patterns": [
                    "Create #implementation threads for coding tasks and technical discussions",
                    "Use private messages for detailed algorithmic discussions",
                    "Share code execution results and outputs in public threads for team review",
                    "Tag @all for major implementation decisions requiring team input",
                    "Provide code examples and executable demonstrations of solutions",
                ],
                "special_instructions": """MATH PROBLEM FORMAT:
When solving math problems and creating final answer polls:
- Use Python to calculate exact results with proper precision
- Extract ONLY the numerical result for final_answer (e.g., "7", "12.5", "3/4")
- Do NOT include "The answer is..." or explanations in final_answer
- Show all calculations, code, and reasoning in supporting_evidence
- Follow the specific format requested by the question (decimal, fraction, etc.)
- Validate results through multiple calculation methods when possible""",
                "tools": code_tools,
                "agent_type": "code",
                "keywords": ["code", "python", "execution", "algorithm", "computation"],
            },
            {
                "name": "WebSearchAgent",
                "full_role": "Fast Web Research and Information Gathering Specialist",
                "short_role": "Fast web research specialist",
                "responsibilities": [
                    "Conduct rapid, targeted web searches for relevant information",
                    "Evaluate source credibility and cross-reference findings",
                    "Perform initial fact-checking and information triage",
                    "Gather real-time data and current information from multiple sources",
                    "Identify trending topics and recent developments",
                    "Extract key facts and summarize findings concisely",
                ],
                "communication_patterns": [
                    "Create #research threads for new investigation topics",
                    "Share quick findings and preliminary results in #main channel",
                    "Use private messages for hypothesis formation and validation discussions",
                    "Tag @all for significant discoveries that impact the entire team",
                    "Provide source links and credibility assessments with all findings",
                ],
                "special_instructions": """RESEARCH METHODOLOGY:
- Always verify information from multiple independent sources
- Prioritize recent, authoritative sources over outdated information
- Include source URLs and publication dates in all research findings
- Flag conflicting information and present different perspectives
- Focus on factual accuracy over speed when sources conflict""",
                "tools": web_tools,
                "agent_type": "tool_calling",
                "keywords": ["research", "web", "search", "facts", "verification"],
            },
            {
                "name": "DeepResearchAgent",
                "full_role": "Comprehensive Analysis and Advanced Research Specialist",
                "short_role": "Deep analysis and advanced research specialist",
                "responsibilities": [
                    "Conduct thorough, multi-layered investigations and analysis",
                    "Develop and rigorously test complex hypotheses and theories",
                    "Synthesize information from diverse sources into coherent insights",
                    "Perform advanced reasoning and logical validation of conclusions",
                    "Design and execute comprehensive research methodologies",
                    "Validate findings through multiple analytical approaches",
                ],
                "communication_patterns": [
                    "Maintain #analysis thread for deep analytical discussions",
                    "Create hypothesis-specific threads for focused investigation",
                    "Use private messages for complex theoretical discussions",
                    "Tag @all for major research breakthroughs and validated findings",
                    "Present comprehensive analysis with supporting evidence and methodology",
                ],
                "special_instructions": """ANALYSIS FRAMEWORK:
- Apply systematic analytical frameworks to complex problems
- Present multiple perspectives and consider alternative explanations
- Use both web research and code execution to validate hypotheses
- Document reasoning processes and analytical methodologies clearly
- Integrate quantitative and qualitative analysis approaches
- Challenge assumptions and test edge cases thoroughly""",
                "tools": list(set(web_tools + code_tools)),
                "agent_type": "code",
                "keywords": ["research", "deep", "analysis", "hypothesis", "validation", "synthesis"],
            },
            {
                "name": "DocumentReaderAgent",
                "full_role": "Document Analysis and Technical Specification Specialist",
                "short_role": "Document analysis and technical specification specialist",
                "responsibilities": [
                    "Analyze and extract key information from technical documents",
                    "Maintain precise citations and track information sources",
                    "Structure complex documentation into digestible summaries",
                    "Validate technical specifications against implementation requirements",
                    "Cross-reference multiple documents for consistency and completeness",
                    "Identify critical details and potential implementation considerations",
                ],
                "communication_patterns": [
                    "Maintain #documentation thread for document-related discussions",
                    "Create topic-specific threads for major document analyses",
                    "Use private messages for detailed technical specification reviews",
                    "Tag @all for critical documentation updates affecting team decisions",
                    "Provide structured summaries with precise page/section references",
                ],
                "special_instructions": """DOCUMENTATION STANDARDS:
- Always include precise citations with page numbers or section references
- Highlight contradictions or ambiguities found in documents
- Extract both explicit information and implied requirements
- Cross-reference claims against other available documentation
- Focus on actionable information that impacts problem-solving
- Maintain clear separation between documented facts and interpretations""",
                "tools": reader_tools,
                "agent_type": "tool_calling",
                "keywords": ["document", "pdf", "extract", "page", "specification", "analysis"],
            },
        ]

        # Define collaboration patterns between agents
        collaboration_matrix = {
            "CodeAgent": {
                "WebSearchAgent": "Receive algorithm suggestions, implementation requirements, and real-world examples to guide development",
                "DocumentReaderAgent": "Get technical specifications, API documentation, and implementation patterns to follow standards",
                "DeepResearchAgent": "Implement complex algorithms, run validation experiments, and execute computational analyses",
            },
            "WebSearchAgent": {
                "CodeAgent": "Share algorithm ideas, provide implementation examples, and verify technical information through search",
                "DocumentReaderAgent": "Cross-reference web findings with official documentation and validate external claims",
                "DeepResearchAgent": "Provide initial research foundation for deeper analysis and hypothesis development",
            },
            "DeepResearchAgent": {
                "CodeAgent": "Design computational experiments, request algorithm implementations, and validate results mathematically",
                "WebSearchAgent": "Expand on initial findings, request targeted searches, and cross-validate information sources",
                "DocumentReaderAgent": "Deep dive into technical documentation, analyze architectural decisions, and validate against specifications",
            },
            "DocumentReaderAgent": {
                "CodeAgent": "Share technical specifications, validate implementations against documentation, and track API requirements",
                "WebSearchAgent": "Compare documentation with external sources, validate technical claims, and share relevant sections",
                "DeepResearchAgent": "Provide detailed technical background, support hypothesis validation with documented evidence",
            },
        }

        # Add collaboration information to agent definitions
        for agent_def in agent_definitions:
            agent_def["collaboration_with"] = collaboration_matrix[agent_def["name"]]

        # Add decentralized tools to each agent's tool set and create configs
        configs = []
        for agent_def in agent_definitions:
            # Add decentralized communication tools
            agent_tools = agent_def["tools"] + create_decentralized_tools(self.message_store, agent_def["name"])

            # Generate dynamic addon
            agent_addon = _generate_agent_addon(agent_def, agent_definitions)

            config = AgentConfig(
                name=agent_def["name"],
                role=agent_def["short_role"],
                tools=agent_tools,
                model=model,
                system_prompt=base_prompt + "\n" + agent_addon,
                keywords=agent_def["keywords"],
                agent_type=agent_def["agent_type"],
            )
            configs.append(config)

        return [
            DecentralizedAgent(
                config=config,
                message_store=self.message_store,
                model_type=self.model_type,
                model_id=self.model_id,
                provider=self.provider,
            )
            for config in configs
        ]

    def run(self, task: str) -> Dict[str, Any]:
        """Run the entire decentralized agent team on a task.

        Args:
            task: The task/question to solve

        Returns:
            Dict containing the result with status, answer, and metadata
        """
        print(f"🚀 Starting decentralized agent team for: {task}")
        logging.info(json.dumps({"event": "team_execution_started", "run_id": self.run_id, "task": task}))

        # Post initial task to message store
        initial_msg = self.message_store.append_message(
            sender="system", content=task, thread_id="main", msg_type="task"
        )
        print("📝 Posted initial task to message store")
        logging.info(
            json.dumps(
                {"event": "task_posted", "run_id": self.run_id, "message_id": initial_msg.get("id"), "task": task}
            )
        )

        # Print team info
        print(f"👥 Created team of {len(self.agents)} agents:")
        for agent in self.agents:
            print(f"  - {agent.config.name} ({agent.config.role})")

        print("\n💬 Running agents in parallel with collaborative approach...")

        # Prepare collaborative task prompt with enhanced instructions
        QUESTION_ADDON = """
IMPORTANT: Before answering, please:
1. Use read_notifications to check if there are any ongoing team discussions or polls
2. Use view_active_polls to see if there are any polls you should vote on
3. Use read_messages to see what other agents have contributed
4. If there's an active poll about the final answer, vote on it using vote_on_poll
5. If no poll exists yet and you're confident in an answer, create a final answer poll using create_final_answer_poll

Work collaboratively with your team!"""

        collaborative_task = f"{task}\n\n{QUESTION_ADDON}"

        # Run all agents in parallel
        self.results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(self._run_single_agent, agent, collaborative_task, i): agent
                for i, agent in enumerate(self.agents)
            }

            # Process completed agents and check for early consensus
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    agent_result = future.result()
                    self.results.append(agent_result)

                    print(f"✅ {agent.config.name} completed ({len(self.results)}/{len(self.agents)} done)")

                    # Check for early consensus after each completion
                    if len(self.results) >= 2:  # Need at least 2 agents for meaningful consensus
                        consensus_result = self._check_for_consensus()
                        if consensus_result:
                            print(f"\n🎯 Consensus reached early after {agent.config.name}: {consensus_result}")
                            logging.info(
                                json.dumps(
                                    {
                                        "event": "consensus_reached_early",
                                        "run_id": self.run_id,
                                        "after_agent": agent.config.name,
                                        "completed_agents": len(self.results),
                                        "final_answer": str(consensus_result)[:200],
                                    }
                                )
                            )

                            # Cancel remaining tasks (best effort)
                            for remaining_future in future_to_agent:
                                if not remaining_future.done():
                                    remaining_future.cancel()

                            self.consensus_reached = True
                            self.final_answer = consensus_result
                            total_duration = time.time() - start_time
                            return self._create_result_summary("success_early", total_duration)

                except Exception as e:
                    print(f"❌ {agent.config.name} failed: {e}")
                    logging.error(
                        json.dumps(
                            {
                                "event": "agent_failed_in_parallel",
                                "run_id": self.run_id,
                                "agent_name": agent.config.name,
                                "error": str(e),
                            }
                        )
                    )
                    # Continue with other agents

        # Final consensus check
        logging.info(json.dumps({"event": "final_consensus_check_started", "run_id": self.run_id}))
        final_consensus = self._check_for_consensus()

        total_duration = time.time() - start_time

        if final_consensus:
            print(f"\n🎯 Final consensus reached: {final_consensus}")
            self.consensus_reached = True
            self.final_answer = final_consensus
            return self._create_result_summary("success", total_duration)
        else:
            # Fallback to last valid result
            return self._create_fallback_result(total_duration)

    def _run_single_agent(self, agent: DecentralizedAgent, task: str, index: int) -> Dict[str, Any]:
        """Run a single agent and return result metadata."""
        print(f"\n🤖 Running {agent.config.name}...")
        logging.info(
            json.dumps(
                {
                    "event": "agent_started",
                    "run_id": self.run_id,
                    "agent_name": agent.config.name,
                    "agent_index": index,
                }
            )
        )

        start_time = time.time()

        try:
            result = agent.run(task)
            end_time = time.time()
            duration = end_time - start_time

            print(f"✅ {agent.config.name} completed")
            logging.info(
                json.dumps(
                    {
                        "event": "agent_completed",
                        "run_id": self.run_id,
                        "agent_name": agent.config.name,
                        "duration_seconds": round(duration, 2),
                        "result_type": type(result).__name__,
                        "result_preview": str(result)[:200] if result else None,
                    }
                )
            )

            return {
                "agent_name": agent.config.name,
                "status": "success",
                "result": result,
                "duration": duration,
                "error": None,
            }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            # Special handling for the "int is not iterable" error we're hunting
            error_str = str(e)
            if "not iterable" in error_str:
                import traceback

                logging.error(
                    json.dumps(
                        {
                            "event": "type_iteration_error_caught",
                            "run_id": self.run_id,
                            "agent_name": agent.config.name,
                            "error": error_str,
                            "error_type": type(e).__name__,
                            "duration_seconds": round(duration, 2),
                            "full_traceback": traceback.format_exc(),
                            "task_preview": task[:300] + "..." if len(task) > 300 else task,
                        }
                    )
                )
                print(f"🔍 FOUND TYPE ERROR in {agent.config.name}: {error_str}")
            else:
                logging.error(
                    json.dumps(
                        {
                            "event": "agent_failed",
                            "run_id": self.run_id,
                            "agent_name": agent.config.name,
                            "error": error_str,
                            "error_type": type(e).__name__,
                            "duration_seconds": round(duration, 2),
                        }
                    )
                )

            print(f"❌ {agent.config.name} failed: {e}")

            return {
                "agent_name": agent.config.name,
                "status": "error",
                "result": f"Error: {e}",
                "duration": duration,
                "error": str(e),
            }

    def _check_for_consensus(self) -> Optional[str]:
        """Check if agents reached consensus through polling.

        IMPORTANT: This method ensures that when multiple polls reach consensus,
        the answer from the FIRST poll (chronologically) is returned, not just
        any successful poll. This maintains consistency with the FinalAnswerTool
        behavior and ensures reproducible results.

        Processing order:
        1. First check for any existing final_answer messages (from previously finalized polls)
        2. Then check active polls in chronological order (oldest created first)
        3. Return the answer from the first successful poll encountered

        Returns:
            str: The final answer from the first successful poll, or None if no consensus
        """
        logging.info(json.dumps({"event": "consensus_check_started", "run_id": self.run_id}))

        try:
            # First, look for any existing finalized polls with final answers
            # These are final answers from polls that already achieved voting threshold
            all_messages = list(self.message_store._iter_messages())
            logging.info(
                json.dumps({"event": "messages_retrieved", "run_id": self.run_id, "message_count": len(all_messages)})
            )

            # Look for the first final_answer message (first poll that achieved threshold)
            for msg in all_messages:
                if isinstance(msg, dict):
                    content = msg.get("content", {})
                    if isinstance(content, dict) and content.get("type") == "final_answer":
                        answer = content.get("answer", "")
                        poll_id = content.get("poll_id", "unknown")
                        logging.info(
                            json.dumps(
                                {
                                    "event": "existing_final_answer_found",
                                    "run_id": self.run_id,
                                    "message_id": msg.get("id"),
                                    "poll_id": poll_id,
                                    "answer": str(answer)[:200],
                                    "note": "first_finalized_poll_in_message_history",
                                }
                            )
                        )
                        print(f"✅ Using answer from first finalized poll: {answer}")
                        return answer

            # Check for active polls that might need finalization
            # IMPORTANT: We check all active polls to see which one FIRST achieves
            # the voting threshold (N//2+1), regardless of creation order
            active_polls = self.message_store.get_active_polls()

            print(f"🔍 Found {len(active_polls)} active polls")
            logging.info(
                json.dumps(
                    {
                        "event": "active_polls_check",
                        "run_id": self.run_id,
                        "poll_count": len(active_polls),
                        "processing_strategy": "first_to_achieve_threshold",
                    }
                )
            )

            first_successful_answer = None
            first_successful_poll_id = None

            for poll in active_polls:
                if isinstance(poll, dict):
                    poll_id = poll.get("poll_id")
                    print(f"🗳️ Checking if poll {poll_id} has achieved voting threshold")
                    logging.info(
                        json.dumps(
                            {
                                "event": "poll_threshold_check",
                                "run_id": self.run_id,
                                "poll_id": poll_id,
                                "poll_question": poll.get("question", ""),
                            }
                        )
                    )

                    if poll_id:
                        # Check if this poll has achieved the voting threshold
                        result = self.message_store.finalize_poll_if_ready(poll_id)
                        print(f"📊 Poll {poll_id} finalization result: {result}")
                        logging.info(
                            json.dumps(
                                {
                                    "event": "poll_finalization_attempted",
                                    "run_id": self.run_id,
                                    "poll_id": poll_id,
                                    "result": str(result)[:200] if result else None,
                                }
                            )
                        )

                        if result and not result.get("deleted"):
                            answer_content = result.get("content", {})
                            if isinstance(answer_content, dict):
                                answer = answer_content.get("answer", "")
                                if first_successful_answer is None:
                                    # This is the FIRST poll to achieve the voting threshold
                                    first_successful_answer = answer
                                    first_successful_poll_id = poll_id
                                    print(f"✅ First poll to achieve consensus: {poll_id} -> {answer}")
                                    logging.info(
                                        json.dumps(
                                            {
                                                "event": "first_poll_achieved_threshold",
                                                "run_id": self.run_id,
                                                "poll_id": poll_id,
                                                "answer": str(answer)[:200],
                                                "strategy": "first_to_reach_vote_threshold",
                                            }
                                        )
                                    )
                                else:
                                    # Another poll also achieved threshold, but we keep the first one
                                    print(
                                        f"ℹ️ Poll {poll_id} also achieved consensus, but {first_successful_poll_id} was first"
                                    )
                                    logging.info(
                                        json.dumps(
                                            {
                                                "event": "subsequent_poll_achieved_threshold",
                                                "run_id": self.run_id,
                                                "poll_id": poll_id,
                                                "first_successful_poll": first_successful_poll_id,
                                                "answer_ignored": str(answer)[:200],
                                            }
                                        )
                                    )
                        elif result and result.get("deleted"):
                            print(f"🗑️ Poll {poll_id} was deleted due to insufficient support")
                            logging.info(
                                json.dumps(
                                    {
                                        "event": "poll_deleted",
                                        "run_id": self.run_id,
                                        "poll_id": poll_id,
                                        "reason": "insufficient_support",
                                    }
                                )
                            )

            # Return the answer from the first poll that achieved the voting threshold
            if first_successful_answer is not None:
                logging.info(
                    json.dumps(
                        {
                            "event": "returning_first_threshold_answer",
                            "run_id": self.run_id,
                            "winning_poll_id": first_successful_poll_id,
                            "answer": str(first_successful_answer)[:200],
                        }
                    )
                )
                return first_successful_answer

        except Exception as e:
            print(f"⚠️ Error checking consensus: {e}")
            logging.error(
                json.dumps(
                    {
                        "event": "consensus_check_error",
                        "run_id": self.run_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
            )

        logging.info(
            json.dumps({"event": "consensus_check_completed", "run_id": self.run_id, "result": "no_consensus"})
        )
        return None

    def _create_result_summary(self, status: str, duration: float) -> Dict[str, Any]:
        """Create a comprehensive result summary."""
        logging.info(
            json.dumps(
                {
                    "event": "run_completed",
                    "run_id": self.run_id,
                    "status": status,
                    "duration_seconds": round(duration, 2),
                }
            )
        )

        return {
            "status": status,
            "answer": self.final_answer,
            "consensus_reached": self.consensus_reached,
            "total_duration": round(duration, 2),
            "agent_results": self.results,
            "run_id": self.run_id,
            "agent_count": len(self.agents),
            "successful_agents": len([r for r in self.results if r.get("status") == "success"]),
            "failed_agents": len([r for r in self.results if r.get("status") == "error"]),
        }

    def _create_fallback_result(self, duration: float) -> Dict[str, Any]:
        """Create result when no consensus is reached."""
        # Fallback to last valid result
        valid_results = [r for r in self.results if r.get("status") == "success"]

        if valid_results:
            answer = valid_results[-1]["result"]  # Use the last valid result
            print(f"\n📝 Final result (fallback): {answer}")
            logging.info(
                json.dumps(
                    {
                        "event": "fallback_result_used",
                        "run_id": self.run_id,
                        "result": str(answer)[:200],
                        "valid_result_count": len(valid_results),
                    }
                )
            )

            self.final_answer = str(answer)
            return self._create_result_summary("success_fallback", duration)
        else:
            print("\n❌ No valid results obtained")
            logging.error(
                json.dumps(
                    {
                        "event": "no_valid_results",
                        "run_id": self.run_id,
                        "total_results": len(self.results),
                        "error_results": len([r for r in self.results if r.get("status") == "error"]),
                    }
                )
            )

            self.final_answer = None
            return self._create_result_summary("failure", duration)

    def get_agent_by_name(self, name: str) -> Optional[DecentralizedAgent]:
        """Get an agent by name."""
        for agent in self.agents:
            if agent.config.name == name:
                return agent
        return None

    def get_results(self) -> Dict[str, Any]:
        """Get the current execution results."""
        return {
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "results": self.results,
            "run_id": self.run_id,
        }

    @property
    def agent_names(self) -> List[str]:
        """Get list of agent names."""
        return [agent.config.name for agent in self.agents]

    @property
    def agent_count(self) -> int:
        """Get number of agents in the team."""
        return len(self.agents)
