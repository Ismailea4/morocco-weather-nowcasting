"""
ReAct (Reason + Act) Agent for Weather Queries.

This module implements a Tool-Using LLM agent that can reason about weather
questions and autonomously call weather prediction functions to answer user queries.

The agent follows the ReAct pattern:
1. Thought: Reason about what information is needed
2. Action: Call appropriate weather tools
3. Observation: Process tool results
4. Final Answer: Provide natural language response to user

Example conversation:
    User: "Is it safe for a helicopter flight to Ceuta right now?"
    
    Agent Thought: "I need to check comprehensive flight safety for Ceuta."
    Agent Action: get_flight_safety_assessment(location="Ceuta", aircraft_type="helicopter")
    Agent Observation: "Overall assessment: NO-GO. Wind speed 45 knots exceeds limits."
    Agent Final Answer: "No, it is not safe. The model predicts winds of 45 knots..."
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .weather_tools import WEATHER_TOOLS, TOOL_FUNCTIONS


class ReActAgent:
    """
    ReAct agent that uses LLM to reason and call weather tools.
    
    Supports multiple LLM providers:
    - OpenAI (gpt-3.5-turbo, gpt-4)
    - Groq (fast inference with free tier)
    - Anthropic Claude
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """
        Initialize ReAct agent.
        
        Args:
            provider: LLM provider ('openai', 'groq', 'anthropic')
            model: Model name
            api_key: API key (or set via environment variable)
            max_iterations: Maximum reasoning iterations
            verbose: Print reasoning steps
        """
        self.provider = provider.lower()
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize LLM client
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("Install openai: pip install openai")
                
        elif self.provider == "groq":
            try:
                import groq
                self.client = groq.Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
            except ImportError:
                raise ImportError("Install groq: pip install groq")
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.tools = WEATHER_TOOLS
        self.conversation_history: List[Dict[str, str]] = []
        
    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent."""
        return """You are a weather analysis assistant for Morocco with access to real-time weather data tools.

Your role is to help users make informed decisions about weather conditions, particularly for aviation safety, by:
1. Understanding what weather information the user needs
2. Calling appropriate weather tools to gather data
3. Analyzing the results with aviation safety in mind
4. Providing clear, actionable answers

Available locations: Ceuta, Tangier, Casablanca, Rabat, Marrakech, Fes, Agadir, Tetouan, Nador, Oujda, Meknes, Kenitra, Safi

Guidelines:
- For flight safety questions, ALWAYS use get_flight_safety_assessment tool
- For wind-specific questions, use get_wind_conditions
- For visibility questions, use get_visibility_conditions
- For temperature/icing questions, use get_temperature_conditions
- For severe weather detection, use detect_severe_weather
- Be direct and safety-focused in your recommendations
- Cite specific numbers (wind speeds in knots, visibility in km, etc.)
- If conditions are unsafe, clearly state why

Remember: Aviation safety is paramount. When in doubt, recommend caution."""

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call LLM with messages and tools."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1  # Low temperature for consistent reasoning
            )
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "finish_reason": response.choices[0].finish_reason
            }
            
        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1
            )
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "finish_reason": response.choices[0].finish_reason
            }
            
        elif self.provider == "anthropic":
            # Anthropic has different API format
            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in self.tools:
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"]
                })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=messages[0]["content"],
                messages=messages[1:],
                tools=anthropic_tools,
                temperature=0.1
            )
            
            # Convert response to unified format
            tool_calls = None
            content = None
            
            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    # Create OpenAI-compatible tool call format
                    class ToolCall:
                        def __init__(self, id, function_name, arguments):
                            self.id = id
                            self.function = type('obj', (object,), {
                                'name': function_name,
                                'arguments': json.dumps(arguments)
                            })()
                    
                    tool_calls.append(ToolCall(
                        block.id,
                        block.name,
                        block.input
                    ))
            
            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": response.stop_reason
            }
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a weather tool with given arguments."""
        if tool_name not in TOOL_FUNCTIONS:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        try:
            func = TOOL_FUNCTIONS[tool_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
    
    def query(self, user_question: str, reset_history: bool = False) -> str:
        """
        Process a user query using ReAct reasoning loop.
        
        Args:
            user_question: User's question
            reset_history: Whether to reset conversation history
            
        Returns:
            Agent's final answer
        """
        if reset_history:
            self.conversation_history = []
        
        # Initialize messages with system prompt
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add user question
        messages.append({"role": "user", "content": user_question})
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User: {user_question}")
            print(f"{'='*60}\n")
        
        # ReAct loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"--- Iteration {iteration} ---")
            
            # Call LLM
            response = self._call_llm(messages)
            
            # Check if LLM wants to use tools
            if response["tool_calls"]:
                if self.verbose:
                    print(f"Thought: Agent decided to use {len(response['tool_calls'])} tool(s)")
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response["tool_calls"]
                    ] if self.provider != "anthropic" else None
                })
                
                # Execute each tool
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if self.verbose:
                        print(f"\nAction: {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})")
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    if self.verbose:
                        print(f"Observation: {tool_result[:500]}..." if len(tool_result) > 500 else f"Observation: {tool_result}")
                    
                    # Add tool result to messages
                    if self.provider == "anthropic":
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call.id,
                                    "content": tool_result
                                }
                            ]
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": tool_result
                        })
                
                # Continue loop to let LLM process tool results
                continue
            
            # No more tool calls - LLM has final answer
            if response["content"]:
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Final Answer:")
                    print(f"{'='*60}")
                    print(response["content"])
                    print(f"{'='*60}\n")
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_question})
                self.conversation_history.append({"role": "assistant", "content": response["content"]})
                
                return response["content"]
            
            # If we get here, something went wrong
            if self.verbose:
                print("Warning: LLM returned no content and no tool calls")
            break
        
        # Max iterations reached
        error_msg = f"Agent reached maximum iterations ({self.max_iterations}) without providing a final answer."
        if self.verbose:
            print(f"\nError: {error_msg}")
        return error_msg
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history


def create_agent(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True
) -> ReActAgent:
    """
    Factory function to create a ReAct agent with sensible defaults.
    
    Args:
        provider: LLM provider ('openai', 'groq', 'anthropic')
        model: Model name (uses default if None)
        api_key: API key (uses environment variable if None)
        verbose: Print reasoning steps
        
    Returns:
        Configured ReActAgent
    """
    # Default models for each provider
    default_models = {
        "openai": "gpt-3.5-turbo",
        "groq": "llama-3.3-70b-versatile",  # Fast and free on Groq
        "anthropic": "claude-3-haiku-20240307"  # Lightweight Claude
    }
    
    if model is None:
        model = default_models.get(provider, "gpt-3.5-turbo")
    
    return ReActAgent(
        provider=provider,
        model=model,
        api_key=api_key,
        verbose=verbose
    )
