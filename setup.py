#!/usr/bin/env python3
"""
Tool Compass - Setup Script
Run this to install dependencies and build the index.
"""

import subprocess
import sys
import os


def run(cmd, check=True):
    """Run a command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def main():
    print("=" * 60)
    print("TOOL COMPASS SETUP")
    print("=" * 60)

    # Check Python version
    print(f"\n✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Install dependencies
    print("\n[1/4] Installing dependencies...")
    run("pip install hnswlib numpy httpx --break-system-packages -q")
    print("✓ Dependencies installed")

    # Check Ollama
    print("\n[2/4] Checking Ollama...")
    result = run("curl -s http://localhost:11434/api/tags", check=False)
    if result.returncode != 0 or "nomic-embed-text" not in result.stdout:
        print("⚠ Ollama not running or nomic-embed-text not available")
        print("  Please run: ollama pull nomic-embed-text")
        print("  Then re-run this script")
        sys.exit(1)
    print("✓ Ollama ready with nomic-embed-text")

    # Build index
    print("\n[3/4] Building Tool Compass index...")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run("python indexer.py")
    print("✓ Index built")

    # Run tests
    print("\n[4/4] Running tests...")
    run("python gateway.py --test")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo start the server:")
    print("  python gateway.py")
    print("\nTo use with Claude Desktop, add to config:")
    print("""
{
  "mcpServers": {
    "tool-compass": {
      "command": "python",
      "args": ["/path/to/tool-compass/gateway.py"]
    }
  }
}
""")


if __name__ == "__main__":
    main()
