"""
Quick launcher for the Morocco Weather Agent Streamlit interface.

This script sets up environment and launches the Streamlit app.

Usage:
    python launch_streamlit.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get project root
    project_root = Path(__file__).resolve().parent
    
    # Path to streamlit app
    app_path = project_root / "src" / "agent" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching Morocco Weather Agent...")
    print(f"📂 Project root: {project_root}")
    print(f"🌐 Opening in browser...")
    print("\n" + "="*70)
    print("Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.headless", "false"
        ], cwd=str(project_root))
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Morocco Weather Agent...")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
