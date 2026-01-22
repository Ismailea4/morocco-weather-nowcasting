"""
Test script for the Morocco Weather Agent.

This script demonstrates the agent's capabilities by running predefined test queries.
Use this to verify the agent is working correctly before interactive use.

Usage:
    python test_agent.py --provider openai
    python test_agent.py --provider groq --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.agent import create_agent


# Test queries covering different capabilities
TEST_QUERIES = [
    {
        "category": "Wind Conditions",
        "query": "What are the wind conditions in Tangier?",
        "expected_tools": ["get_wind_conditions"]
    },
    {
        "category": "Flight Safety",
        "query": "Is it safe for a helicopter flight to Ceuta right now?",
        "expected_tools": ["get_flight_safety_assessment"]
    },
    {
        "category": "Visibility",
        "query": "What's the visibility like in Casablanca?",
        "expected_tools": ["get_visibility_conditions"]
    },
    {
        "category": "Temperature",
        "query": "What's the temperature in Marrakech?",
        "expected_tools": ["get_temperature_conditions"]
    },
    {
        "category": "Severe Weather",
        "query": "Are there any weather hazards in Fes?",
        "expected_tools": ["detect_severe_weather"]
    },
]


def run_test(agent, test_case: dict, verbose: bool = False):
    """Run a single test query."""
    print(f"\n{'='*70}")
    print(f"Test Category: {test_case['category']}")
    print(f"{'='*70}")
    print(f"Query: {test_case['query']}")
    print(f"Expected Tools: {', '.join(test_case['expected_tools'])}")
    print(f"{'='*70}\n")
    
    try:
        response = agent.query(test_case['query'], reset_history=True)
        
        if not verbose:
            print(f"\nResponse:\n{response}\n")
        
        print(f"✅ Test completed successfully\n")
        return True
    
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}\n")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test Morocco Weather Agent")
    parser.add_argument(
        "-p", "--provider",
        type=str,
        choices=["openai", "groq", "anthropic"],
        default="openai",
        help="LLM provider"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed agent reasoning"
    )
    parser.add_argument(
        "-t", "--test",
        type=int,
        help="Run specific test number (1-5)"
    )
    
    args = parser.parse_args()
    
    print("\n🧪 Morocco Weather Agent Test Suite")
    print(f"Provider: {args.provider}")
    if args.model:
        print(f"Model: {args.model}")
    print()
    
    # Create agent
    try:
        agent = create_agent(
            provider=args.provider,
            model=args.model,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        print("\nMake sure you have set the appropriate API key:")
        print(f"  - For OpenAI: export OPENAI_API_KEY=your_key")
        print(f"  - For Groq: export GROQ_API_KEY=your_key")
        print(f"  - For Anthropic: export ANTHROPIC_API_KEY=your_key")
        sys.exit(1)
    
    # Run tests
    if args.test:
        # Run single test
        test_idx = args.test - 1
        if 0 <= test_idx < len(TEST_QUERIES):
            test_case = TEST_QUERIES[test_idx]
            success = run_test(agent, test_case, args.verbose)
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Invalid test number. Choose 1-{len(TEST_QUERIES)}")
            sys.exit(1)
    else:
        # Run all tests
        print(f"Running {len(TEST_QUERIES)} test queries...\n")
        
        results = []
        for i, test_case in enumerate(TEST_QUERIES, 1):
            print(f"\n[Test {i}/{len(TEST_QUERIES)}]")
            success = run_test(agent, test_case, args.verbose)
            results.append((test_case['category'], success))
            
            if i < len(TEST_QUERIES):
                input("\nPress Enter to continue to next test...")
        
        # Print summary
        print(f"\n{'='*70}")
        print("Test Summary")
        print(f"{'='*70}")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for category, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {category}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        print(f"{'='*70}\n")
        
        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
