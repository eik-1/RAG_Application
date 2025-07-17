#!/usr/bin/env python3
"""
Startup script for the RAG Application.

This script provides convenient commands to start the backend, install dependencies,
run evaluations, and manage the application lifecycle.

Usage:
    python start_application.py [command]

Commands:
    backend     - Start the FastAPI backend server
    install     - Install Python dependencies
    evaluate    - Run the evaluation framework
    quick-test  - Run a quick test of the system
    help        - Show this help message
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, cwd=None, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False


def install_backend():
    """Install backend dependencies."""
    print("📦 Installing backend dependencies...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    success = run_command("pip install -r requirements.txt", cwd=backend_dir)
    
    if success:
        print("✅ Backend dependencies installed successfully!")
    else:
        print("❌ Failed to install backend dependencies")
    
    return success


def install_frontend():
    """Install frontend dependencies."""
    print("📦 Installing frontend dependencies...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    success = run_command("npm install", cwd=frontend_dir)
    
    if success:
        print("✅ Frontend dependencies installed successfully!")
    else:
        print("❌ Failed to install frontend dependencies")
    
    return success


def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting RAG Application backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    # Check if dependencies are installed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("⚠️  Backend dependencies not installed. Installing now...")
        if not install_backend():
            return False
    
    print("Starting FastAPI server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    return run_command("python main.py", cwd=backend_dir, check=False)


def start_frontend():
    """Start the Next.js frontend development server."""
    print("🚀 Starting Next.js frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("⚠️  Frontend dependencies not installed. Installing now...")
        if not install_frontend():
            return False
    
    print("Starting Next.js development server on http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    
    return run_command("npm run dev", cwd=frontend_dir, check=False)


def run_evaluation():
    """Run the evaluation framework."""
    print("📊 Running RAG system evaluation...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    return run_command("python evaluate.py", cwd=backend_dir, check=False)


def run_quick_test():
    """Run a quick test of the system."""
    print("🔧 Running quick system test...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    return run_command("python evaluate.py quick", cwd=backend_dir, check=False)


def show_help():
    """Show help message."""
    print(__doc__)


def check_system_requirements():
    """Check if system requirements are met."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check pip
    try:
        import pip
        print("✅ pip available")
    except ImportError:
        print("❌ pip not available")
        return False
    
    # Check Node.js (for frontend)
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()}")
        else:
            print("⚠️  Node.js not found (required for frontend)")
    except FileNotFoundError:
        print("⚠️  Node.js not found (required for frontend)")
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm {result.stdout.strip()}")
        else:
            print("⚠️  npm not found (required for frontend)")
    except FileNotFoundError:
        print("⚠️  npm not found (required for frontend)")
    
    return True


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        command = "help"
    else:
        command = sys.argv[1].lower()
    
    print("🤖 RAG Application Startup Script")
    print("=" * 40)
    
    # Check system requirements first
    if command != "help" and not check_system_requirements():
        print("\n❌ System requirements not met!")
        sys.exit(1)
    
    print()
    
    if command == "backend":
        start_backend()
    elif command == "frontend":
        start_frontend()
    elif command == "install":
        print("Installing all dependencies...")
        backend_success = install_backend()
        frontend_success = install_frontend()
        
        if backend_success and frontend_success:
            print("\n✅ All dependencies installed successfully!")
        elif backend_success:
            print("\n⚠️  Backend installed, but frontend installation failed")
        elif frontend_success:
            print("\n⚠️  Frontend installed, but backend installation failed")
        else:
            print("\n❌ Failed to install dependencies")
            sys.exit(1)
    elif command == "evaluate":
        run_evaluation()
    elif command == "quick-test":
        run_quick_test()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python start_application.py help' for available commands")
        sys.exit(1)


if __name__ == "__main__":
    main() 