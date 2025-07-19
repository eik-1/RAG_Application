#!/usr/bin/env python3
"""
Gradual package installation script for RAG Application.
Handles network issues by installing packages in stages.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_pip_install(packages, retry_count=3, timeout=300):
    """Install packages with retries and timeout handling."""
    for package in packages:
        for attempt in range(retry_count):
            try:
                print(f"📦 Installing {package} (attempt {attempt + 1}/{retry_count})...")
                
                # Use pip with increased timeout and retries
                cmd = [
                    sys.executable, "-m", "pip", "install",
                    "--timeout", str(timeout),
                    "--retries", "5",
                    "--default-timeout", "300",
                    package
                ]
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 60  # Extra buffer
                )
                
                print(f"✅ Successfully installed {package}")
                break
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                if attempt == retry_count - 1:
                    print(f"⚠️  Skipping {package} after {retry_count} attempts")
                else:
                    print(f"🔄 Retrying in 5 seconds...")
                    time.sleep(5)
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout installing {package}")
                if attempt == retry_count - 1:
                    print(f"⚠️  Skipping {package} due to timeout")
                else:
                    print(f"🔄 Retrying in 10 seconds...")
                    time.sleep(10)


def main():
    """Main installation function."""
    print("🚀 RAG Application - Gradual Package Installation")
    print("=" * 50)
    
    # Stage 1: Core packages (small, essential)
    print("\n📋 Stage 1: Core packages...")
    core_packages = [
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.25.0"
    ]
    run_pip_install(core_packages)
    
    # Stage 2: Data processing (medium size)
    print("\n📋 Stage 2: Data processing...")
    data_packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyPDF2>=3.0.0",
        "nltk>=3.8.0"
    ]
    run_pip_install(data_packages)
    
    # Stage 3: Vector database
    print("\n📋 Stage 3: Vector database...")
    vector_packages = [
        "faiss-cpu>=1.7.0"
    ]
    run_pip_install(vector_packages)
    
    # Stage 4: ML packages (largest, install individually)
    print("\n📋 Stage 4: Machine Learning packages...")
    print("⚠️  These are large packages that may take time...")
    
    ml_packages = [
        "torch>=2.0.0,<3.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0"
    ]
    
    for package in ml_packages:
        print(f"\n🤖 Installing {package}...")
        print("This may take several minutes...")
        run_pip_install([package], retry_count=2, timeout=600)  # Longer timeout
    
    # Stage 5: Evaluation packages
    print("\n📋 Stage 5: Evaluation packages...")
    eval_packages = [
        "rouge-score>=0.1.2",
        "pytest>=7.0.0"
    ]
    run_pip_install(eval_packages)
    
    print("\n✅ Installation completed!")
    print("\n🧪 Testing imports...")
    
    # Test critical imports
    test_imports = [
        "fastapi",
        "numpy",
        "pandas",
        "faiss",
        "nltk"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
    
    print("\n🎉 Setup complete! You can now run:")
    print("   python main.py")


if __name__ == "__main__":
    main() 