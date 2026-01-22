"""
Streamlit Web Interface for Morocco Weather Agent.

A user-friendly web interface for interacting with the AI Weather Assistant.

Usage:
    streamlit run src/agent/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.agent.react_agent import create_agent, ReActAgent


# Page configuration
st.set_page_config(
    page_title="Morocco Weather Agent",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .safety-go {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .safety-caution {
        background-color: #FF9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .safety-nogo {
        background-color: #F44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
    }
    .tool-call {
        background-color: #FFF3E0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False


def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment or session state."""
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    env_key = env_keys.get(provider)
    
    # Check environment first
    api_key = os.getenv(env_key)
    if api_key:
        return api_key
    
    # Check session state
    session_key = f"{provider}_api_key"
    if session_key in st.session_state:
        return st.session_state[session_key]
    
    return None


def initialize_agent(provider: str, model: Optional[str], api_key: str, verbose: bool) -> Optional[ReActAgent]:
    """Initialize the ReAct agent."""
    try:
        agent = create_agent(
            provider=provider,
            model=model if model else None,
            api_key=api_key,
            verbose=verbose
        )
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None


def display_message(role: str, content: str):
    """Display a chat message."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)


def extract_safety_assessment(response: str) -> Optional[str]:
    """Extract safety assessment from response for highlighting."""
    response_lower = response.lower()
    if "no-go" in response_lower or "not safe" in response_lower or "not recommended" in response_lower:
        return "NO-GO"
    elif "caution" in response_lower or "marginal" in response_lower:
        return "CAUTION"
    elif "safe" in response_lower or "favorable" in response_lower or "go" in response_lower:
        return "GO"
    return None


def display_safety_badge(assessment: str):
    """Display a safety assessment badge."""
    if assessment == "GO":
        st.markdown('<div class="safety-go">✅ SAFE FOR FLIGHT</div>', unsafe_allow_html=True)
    elif assessment == "CAUTION":
        st.markdown('<div class="safety-caution">⚠️ PROCEED WITH CAUTION</div>', unsafe_allow_html=True)
    elif assessment == "NO-GO":
        st.markdown('<div class="safety-nogo">❌ NOT SAFE - DO NOT FLY</div>', unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌤️ Morocco Weather Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">AI-Powered Weather Assistant for Aviation Safety</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            ["groq", "openai", "anthropic"],
            help="Select your LLM provider. Groq is recommended for free usage."
        )
        
        # Model selection
        model_options = {
            "groq": ["llama-3.3-70b-versatile (default)", "llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
            "openai": ["gpt-3.5-turbo (default)", "gpt-4", "gpt-4-turbo"],
            "anthropic": ["claude-3-haiku-20240307 (default)", "claude-3-sonnet-20240229"]
        }
        
        model_choice = st.selectbox(
            "Model",
            model_options.get(provider, ["default"]),
            help="Select the model to use. Default is recommended."
        )
        
        # Extract actual model name
        model = None if "(default)" in model_choice else model_choice
        
        # API Key input
        st.subheader("🔑 API Key")
        api_key_input = st.text_input(
            f"{provider.upper()} API Key",
            type="password",
            value=get_api_key(provider) or "",
            help=f"Enter your {provider.upper()} API key or set it as an environment variable."
        )
        
        # Store in session state
        if api_key_input:
            st.session_state[f"{provider}_api_key"] = api_key_input
        
        # Verbose mode
        verbose = st.checkbox(
            "Show Reasoning Steps",
            value=False,
            help="Display agent's thinking process and tool calls"
        )
        
        st.divider()
        
        # Initialize/Reinitialize button
        if st.button("🚀 Initialize Agent", type="primary"):
            if not api_key_input:
                st.error(f"Please enter your {provider.upper()} API key!")
            else:
                with st.spinner("Initializing agent..."):
                    agent = initialize_agent(provider, model, api_key_input, verbose)
                    if agent:
                        st.session_state.agent = agent
                        st.session_state.agent_initialized = True
                        st.success("Agent initialized successfully!")
                    else:
                        st.session_state.agent_initialized = False
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.reset_conversation()
            st.rerun()
        
        st.divider()
        
        # Quick links
        st.subheader("📚 Resources")
        st.markdown("""
        - [Get Groq API Key](https://console.groq.com) (Free)
        - [OpenAI API](https://platform.openai.com)
        - [Anthropic API](https://console.anthropic.com)
        """)
        
        st.divider()
        
        # Supported locations
        st.subheader("📍 Supported Locations")
        locations = ["Ceuta", "Tangier", "Casablanca", "Rabat", "Marrakech", "Fes", 
                     "Agadir", "Tetouan", "Nador", "Oujda", "Meknes", "Kenitra", "Safi"]
        st.markdown(", ".join(locations))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("💡 Example Questions")
        
        example_questions = [
            "Is it safe for a helicopter flight to Ceuta right now?",
            "What are the wind conditions in Tangier?",
            "Can I fly a small plane to Casablanca today?",
            "What's the visibility like in Agadir?",
            "Are there any weather hazards in Fes?",
            "What's the temperature in Marrakech?",
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"📝 {question}", key=f"example_{i}"):
                if st.session_state.agent_initialized:
                    # Add to chat
                    st.session_state.current_question = question
                    st.rerun()
                else:
                    st.warning("Please initialize the agent first!")
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message["role"], message["content"])
                
                # Show safety badge if applicable
                if message["role"] == "assistant":
                    assessment = extract_safety_assessment(message["content"])
                    if assessment:
                        display_safety_badge(assessment)
        
        # Chat input
        if not st.session_state.agent_initialized:
            st.info("👈 Please configure and initialize the agent in the sidebar to start chatting!")
        else:
            # Check if there's a question from example button
            if "current_question" in st.session_state:
                user_input = st.session_state.current_question
                del st.session_state.current_question
            else:
                user_input = st.chat_input("Ask about weather conditions...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Get agent response
                with st.spinner("🤔 Agent is thinking..."):
                    try:
                        # Capture verbose output if enabled
                        if verbose:
                            import io
                            from contextlib import redirect_stdout
                            
                            output_buffer = io.StringIO()
                            with redirect_stdout(output_buffer):
                                response = st.session_state.agent.query(user_input)
                            
                            verbose_output = output_buffer.getvalue()
                            if verbose_output:
                                with st.expander("🔍 View Agent Reasoning"):
                                    st.code(verbose_output, language="text")
                        else:
                            response = st.session_state.agent.query(user_input)
                        
                        # Add assistant message
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        response = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to display new messages
                st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🚁 Morocco Weather Agent - AI-Powered Aviation Safety Assistant</p>
        <p>Powered by Groq/OpenAI/Anthropic | Real-time weather data from Open-Meteo</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
