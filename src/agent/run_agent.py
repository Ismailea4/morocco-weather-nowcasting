"""
CLI runner for Morocco Weather ReAct Agent.

This script provides a command-line interface to interact with the weather agent.

Usage:
    # Interactive chat mode
    python -m src.agent.run_agent
    
    # Single query mode
    python -m src.agent.run_agent --query "Is it safe to fly to Ceuta?"
    
    # Use different LLM provider
    python -m src.agent.run_agent --provider groq --model llama-3.3-70b-versatile
    
    # Non-verbose mode (no reasoning steps)
    python -m src.agent.run_agent --no-verbose

Environment Variables:
    OPENAI_API_KEY   - OpenAI API key
    GROQ_API_KEY     - Groq API key
    ANTHROPIC_API_KEY - Anthropic API key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.agent.chat_interface import run_interactive_chat, run_single_query


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Morocco Weather ReAct Agent - Interactive weather assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat with OpenAI
  python -m src.agent.run_agent
  
  # Single query
  python -m src.agent.run_agent -q "What are the wind conditions in Tangier?"
  
  # Use Groq (fast and free)
  python -m src.agent.run_agent --provider groq
  
  # Use specific model
  python -m src.agent.run_agent --provider openai --model gpt-4
  
  # Quiet mode (no reasoning steps)
  python -m src.agent.run_agent --no-verbose

Supported Providers:
  openai    - OpenAI GPT models (requires OPENAI_API_KEY)
  groq      - Groq inference (requires GROQ_API_KEY, free tier available)
  anthropic - Anthropic Claude (requires ANTHROPIC_API_KEY)

Recommended Free Options:
  • Groq: Fast inference with free tier
    - llama-3.3-70b-versatile (default, fast and capable)
    - llama-3.1-8b-instant (faster, lighter)
  
  • OpenAI: Free trial credits
    - gpt-3.5-turbo (default, balanced)
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query mode - ask one question and exit"
    )
    
    parser.add_argument(
        "-p", "--provider",
        type=str,
        choices=["openai", "groq", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name (uses provider default if not specified)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set via environment variable)"
    )
    
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output (hide reasoning steps)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (YAML)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Override with command line arguments
    provider = args.provider or config.get("llm", {}).get("provider", "openai")
    model = args.model or config.get("llm", {}).get("model")
    api_key = args.api_key or config.get("llm", {}).get("api_key")
    verbose = not args.no_verbose and config.get("llm", {}).get("verbose", True)
    
    # Print startup info
    print(f"\n🚀 Starting Morocco Weather Agent")
    print(f"   Provider: {provider}")
    if model:
        print(f"   Model: {model}")
    print(f"   Verbose: {verbose}\n")
    
    try:
        if args.query:
            # Single query mode
            response = run_single_query(
                question=args.query,
                provider=provider,
                model=model,
                api_key=api_key,
                verbose=verbose
            )
            print(f"\nQuestion: {args.query}\n")
            print(f"Answer: {response}\n")
        else:
            # Interactive chat mode
            run_interactive_chat(
                provider=provider,
                model=model,
                api_key=api_key,
                verbose=verbose
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
