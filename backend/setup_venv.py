#!/usr/bin/env python3
"""Setup script for creating Python virtual environment."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🐍 Setting up Python virtual environment for RAG Application")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Create virtual environment
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() == 'y':
            print("🗑️  Removing existing virtual environment...")
            if os.name == 'nt':  # Windows
                run_command("rmdir /s /q venv", check=False)
            else:  # Unix/Linux/macOS
                run_command("rm -rf venv", check=False)
        else:
            print("Using existing virtual environment")
            return
    
    print("📦 Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv venv"):
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    print("✅ Virtual environment created successfully!")
    print()
    print("🚀 Next steps:")
    print("1. Activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("3. Run the application:")
    print("   python main.py")
    print()
    print("💡 Or use the startup script from the project root:")
    print("   python start_application.py backend")


if __name__ == "__main__":
    main() 