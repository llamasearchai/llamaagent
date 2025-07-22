#!/usr/bin/env python3
"""
Quick Start Script for LlamaAgent Advanced
Run the cutting-edge AI system with one command
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def log_info(msg):
    print(f"[INFO] {msg}")

def log_success(msg):
    print(f"[SUCCESS] {msg}")

def log_error(msg):
    print(f"[ERROR] {msg}")

def log_warning(msg):
    print(f"[WARNING] {msg}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 11):
        log_error("Python 3.11+ is required")
        return False
    return True

def install_dependencies():
    """Install required dependencies"""
    log_info("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "fastapi", "uvicorn[standard]", "pydantic", "httpx",
            "structlog", "python-dotenv", "asyncio-throttle",
            "psutil", "prometheus-client"
        ])
        log_success("Basic dependencies installed")
        
        # Try to install cutting-edge dependencies
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "litellm[proxy]", "pillow", "transformers", 
                "torch", "torchvision", "--index-url", 
                "https://download.pytorch.org/whl/cpu"
            ])
            log_success("Cutting-edge dependencies installed")
        except subprocess.CalledProcessError:
            log_warning("Some cutting-edge dependencies failed to install. Basic functionality will work.")
        
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create basic .env file if it doesn't exist"""
    env_path = Path(".env")
    if not env_path.exists():
        log_info("Creating .env file...")
        env_content = """# LlamaAgent Advanced Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
MISTRAL_API_KEY=your_mistral_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Features
ENABLE_MULTIMODAL=true
ENABLE_REASONING=true
ENABLE_VISION=true
BUDGET_LIMIT=10.0

# Security
JWT_SECRET_KEY=your_secret_key_here
"""
        with open(env_path, "w") as f:
            f.write(env_content)
        log_success(".env file created")

def start_server():
    """Start the LlamaAgent server"""
    log_info("Starting LlamaAgent Advanced server...")
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    try:
        # Try to start with uvicorn
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.llamaagent.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        log_info("Command: " + " ".join(cmd))
        process = subprocess.Popen(cmd)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                log_success("Starting LlamaAgent Advanced is running!")
                print("\n" + "="*60)
                print("Featured LlamaAgent Advanced - Ready!")
                print("="*60)
                print(" API Documentation: http://localhost:8000/docs")
                print("Scanning Health Check: http://localhost:8000/health")
                print("TARGET Advanced Models: http://localhost:8000/models/advanced")
                print("\nBUILD: Cutting-Edge Endpoints:")
                print("   • POST /multimodal/analyze - Multimodal reasoning")
                print("   • POST /reasoning/advanced - O1-style thinking")
                print("   • POST /litellm/universal - Universal LLM interface")
                print("   • POST /vision/analyze - Vision analysis")
                print("\nINSIGHT Remember to set your API keys in .env file!")
                print("="*60)
                
                # Keep server running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    log_info("Shutting down server...")
                    process.terminate()
                    
            else:
                log_error("Server started but health check failed")
                return False
        except requests.RequestException:
            log_error("Could not connect to server")
            return False
            
    except FileNotFoundError:
        log_error("Could not start uvicorn. Make sure it's installed.")
        return False
    except Exception as e:
        log_error(f"Failed to start server: {e}")
        return False

def main():
    """Main function"""
    print("Starting LlamaAgent Advanced - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        log_error("Dependency installation failed")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main() 