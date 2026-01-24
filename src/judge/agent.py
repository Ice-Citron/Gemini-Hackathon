"""
LLM-Judge Agent

This agent uses Anthropic's Claude models or GDM's Gemini models to evaluate
whether tasks completed by another agent (ethical hacking agent) were done
correctly. It uses MCP tools to verify the state of the system.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import JudgeConfig
from .prompts import SYSTEM_PROMPT, create_evaluation_prompt


class TaskEvaluation:
    """Result of a task evaluation"""

    def __init__(
        self,
        score: float,
        summary: str,
        verification_steps: List[Dict[str, Any]] = None
    ):
        self.score = score
        self.summary = summary
        self.verification_steps = verification_steps or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary (API response format)"""
        return {
            "score": self.score,
            "summary": self.summary
        }

    def to_dict_detailed(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary with verification steps"""
        return {
            "score": self.score,
            "summary": self.summary,
            "verification_steps": self.verification_steps
        }

    def __str__(self) -> str:
        return f"Score: {self.score}\n{self.summary}"


class LLMJudgeAgent:
    """
    LLM-powered judge agent that evaluates task completion using MCP tools.

    The agent:
    1. Receives a task description and success criteria
    2. Uses MCP verification tools to check system state
    3. Reasons about whether the task was completed correctly
    4. Returns a structured evaluation result
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig()
        self.config.validate_config()

        self.client = Anthropic(api_key=self.config.anthropic_api_key)
        self.mcp_session: Optional[ClientSession] = None
        self.stdio_context = None
        self.available_tools: List[Dict[str, Any]] = []

    async def __aenter__(self):
        """Async context manager entry - connects to MCP server"""
        await self.connect_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnects from MCP server"""
        await self.disconnect_mcp()

    async def connect_mcp(self):
        """Connect to the MCP server and load available tools"""
        server_params = StdioServerParameters(
            command=self.config.mcp_server_command.split()[0],
            args=self.config.mcp_server_command.split()[1:],
            env=None
        )

        # Create stdio client connection (it's a context manager)
        self.stdio_context = stdio_client(server_params)
        self.stdio, self.write = await self.stdio_context.__aenter__()

        # Create MCP session
        self.mcp_session = ClientSession(self.stdio, self.write)
        await self.mcp_session.__aenter__()

        # Initialize and get available tools
        await self.mcp_session.initialize()
        tools_response = await self.mcp_session.list_tools()

        # Convert MCP tools to Anthropic tool format
        self.available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in tools_response.tools
        ]

        print(f"Connected to MCP server. Available tools: {[t['name'] for t in self.available_tools]}")

    async def disconnect_mcp(self):
        """Disconnect from the MCP server"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
            self.mcp_session = None
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)
            self.stdio_context = None

    async def evaluate_task(
        self,
        task_description: str,
        agent_response: str
    ) -> TaskEvaluation:
        """
        Evaluate whether a task was completed successfully based on agent's response.

        Args:
            task_description: Description of the task that was assigned
            agent_response: The response/output from the agent being evaluated

        Returns:
            TaskEvaluation with score (0.0-1.0) and summary
        """
        if not self.mcp_session:
            raise RuntimeError("MCP session not connected. Use 'async with' context manager.")

        # Create the evaluation prompt
        user_prompt = create_evaluation_prompt(
            task_description=task_description,
            agent_response=agent_response
        )

        # Run the agentic loop with tool use
        messages = [{"role": "user", "content": user_prompt}]
        verification_steps = []
        tool_call_count = 0

        while tool_call_count < self.config.max_tool_calls:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=SYSTEM_PROMPT,
                tools=self.available_tools,
                messages=messages
            )

            # Check if we're done (no more tool calls)
            if response.stop_reason == "end_turn":
                # Extract final evaluation from response
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text

                return self._parse_final_evaluation(final_text, verification_steps)

            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant's response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call_count += 1

                        # Call the MCP tool
                        result = await self.mcp_session.call_tool(
                            block.name,
                            block.input
                        )

                        # Extract text content from result
                        result_text = ""
                        for content in result.content:
                            if hasattr(content, 'text'):
                                result_text += content.text

                        # Record verification step
                        verification_steps.append({
                            "tool": block.name,
                            "input": block.input,
                            "result": result_text
                        })

                        # Add tool result to messages
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text
                        })

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason
                raise RuntimeError(f"Unexpected stop reason: {response.stop_reason}")

        # Max tool calls reached without conclusion
        return TaskEvaluation(
            score=0.0,
            summary="Evaluation incomplete: maximum tool calls reached without conclusion",
            verification_steps=verification_steps
        )

    def _parse_final_evaluation(
        self,
        response_text: str,
        verification_steps: List[Dict[str, Any]]
    ) -> TaskEvaluation:
        """
        Parse the final evaluation from the LLM's response.

        Expected format in response:
        SCORE: 0.0-1.0
        SUMMARY: explanation
        """
        lines = response_text.strip().split('\n')
        score = 0.0
        summary = response_text

        # Try to parse structured output
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    # Clamp score between 0 and 1
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    score = 0.0
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        return TaskEvaluation(
            score=score,
            summary=summary,
            verification_steps=verification_steps
        )
