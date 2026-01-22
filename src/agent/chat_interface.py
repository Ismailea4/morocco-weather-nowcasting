"""
Interactive chat interface for the Morocco Weather Agent.

This module provides a simple command-line chat interface for interacting
with the weather ReAct agent.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from .react_agent import create_agent, ReActAgent


class WeatherChatInterface:
    """Interactive chat interface for weather queries."""
    
    def __init__(self, agent: ReActAgent):
        """
        Initialize chat interface.
        
        Args:
            agent: Configured ReActAgent instance
        """
        self.agent = agent
        self.running = False
    
    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "="*70)
        print("🌤️  Morocco Weather Assistant (ReAct Agent)")
        print("="*70)
        print("\nAsk me about weather conditions in Morocco!")
        print("\nExample questions:")
        print("  • Is it safe for a helicopter flight to Ceuta right now?")
        print("  • What are the wind conditions in Tangier?")
        print("  • Can I fly a small plane to Casablanca today?")
        print("  • What's the visibility like in Agadir?")
        print("  • Are there any severe weather risks in Fes?")
        print("\nCommands:")
        print("  • /help    - Show this help message")
        print("  • /reset   - Reset conversation history")
        print("  • /history - Show conversation history")
        print("  • /quit    - Exit the chat")
        print("="*70 + "\n")
    
    def print_help(self):
        """Print help message."""
        print("\n" + "="*70)
        print("Available Commands:")
        print("="*70)
        print("/help    - Show this help message")
        print("/reset   - Reset conversation history and start fresh")
        print("/history - Display conversation history")
        print("/quit    - Exit the weather assistant")
        print("\nAvailable Locations:")
        print("Ceuta, Tangier, Casablanca, Rabat, Marrakech, Fes, Agadir,")
        print("Tetouan, Nador, Oujda, Meknes, Kenitra, Safi")
        print("\nYou can also use coordinates: '35.8,-5.3'")
        print("="*70 + "\n")
    
    def print_history(self):
        """Print conversation history."""
        history = self.agent.get_conversation_history()
        if not history:
            print("\n[No conversation history yet]\n")
            return
        
        print("\n" + "="*70)
        print("Conversation History:")
        print("="*70)
        for i, msg in enumerate(history, 1):
            role = "You" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            print(f"\n[{i}] {role}:")
            print(content)
        print("="*70 + "\n")
    
    def handle_command(self, user_input: str) -> bool:
        """
        Handle special commands.
        
        Args:
            user_input: User input
            
        Returns:
            True if command was handled, False otherwise
        """
        if user_input.startswith("/"):
            command = user_input.lower().strip()
            
            if command == "/quit" or command == "/exit" or command == "/q":
                print("\n👋 Goodbye! Stay safe and check the weather before you fly!\n")
                self.running = False
                return True
            
            elif command == "/help" or command == "/h":
                self.print_help()
                return True
            
            elif command == "/reset" or command == "/r":
                self.agent.reset_conversation()
                print("\n✅ Conversation history reset.\n")
                return True
            
            elif command == "/history" or command == "/hist":
                self.print_history()
                return True
            
            else:
                print(f"\n❌ Unknown command: {command}")
                print("Type /help for available commands.\n")
                return True
        
        return False
    
    def run(self):
        """Run the interactive chat loop."""
        self.running = True
        self.print_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if self.handle_command(user_input):
                    continue
                
                # Process question with agent
                print()  # Add spacing
                response = self.agent.query(user_input)
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Stay safe and check the weather before you fly!\n")
                break
            except EOFError:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
                if self.agent.verbose:
                    import traceback
                    traceback.print_exc()


def run_interactive_chat(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True
):
    """
    Run interactive chat interface with weather agent.
    
    Args:
        provider: LLM provider ('openai', 'groq', 'anthropic')
        model: Model name (uses default if None)
        api_key: API key (uses environment variable if None)
        verbose: Print agent reasoning steps
    """
    # Check for API key
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    env_key = env_keys.get(provider)
    if not api_key and env_key and not os.getenv(env_key):
        print(f"\n❌ Error: {env_key} not found in environment variables.")
        print(f"Please set it: export {env_key}=your_api_key_here\n")
        sys.exit(1)
    
    # Create agent
    try:
        agent = create_agent(
            provider=provider,
            model=model,
            api_key=api_key,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n❌ Error creating agent: {e}\n")
        sys.exit(1)
    
    # Create and run chat interface
    chat = WeatherChatInterface(agent)
    chat.run()


def run_single_query(
    question: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Run a single query without interactive chat.
    
    Args:
        question: Question to ask
        provider: LLM provider
        model: Model name
        api_key: API key
        verbose: Print reasoning steps
        
    Returns:
        Agent's response
    """
    agent = create_agent(
        provider=provider,
        model=model,
        api_key=api_key,
        verbose=verbose
    )
    
    return agent.query(question)
