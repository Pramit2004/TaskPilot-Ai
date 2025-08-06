#!/usr/bin/env python3
"""
TaskPilot AI Setup and Installation Script
This script helps you set up TaskPilot AI properly
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_banner():
    print("""
    🚀 TaskPilot AI Setup
    =====================
    The True AI Data Scientist Setup Script
    """)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    requirements = [
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "pydantic",
        "python-multipart",
        "google-generativeai",
        "langchain-google-genai",
        "plotly",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "reportlab",
        "Pillow",
        "librosa",
        "nltk",
        "textblob"
    ]
    
    try:
        for package in requirements:
            print(f"   Installing {package.split('==')[0]}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "reports",
        "uploads", 
        "static",
        "data",
        "models",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def create_config_file():
    """Create configuration file"""
    print("\n⚙️ Creating configuration file...")
    
    config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False
        },
        "data": {
            "max_file_size_mb": 100,
            "supported_formats": [".csv", ".xlsx", ".xls", ".json"],
            "upload_dir": "uploads"
        },
        "analysis": {
            "default_time_budget": 600,
            "max_time_budget": 1800,
            "enable_agents": True
        },
        "api": {
            "gemini_api_key": "",
            "enable_advanced_features": True
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("   ✅ config.json created")
    return True

def create_environment_file():
    """Create .env file template"""
    print("\n🔐 Creating environment file template...")
    
    env_content = """# TaskPilot AI Environment Configuration
# Copy this to .env and fill in your values

# Gemini API Key (optional but recommended for enhanced features)
# Get your free key at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Logging Level
LOG_LEVEL=INFO
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("   ✅ .env.example created")
    print("   📝 Copy .env.example to .env and add your Gemini API key")
    return True

def check_file_structure():
    """Check if all required files exist"""
    print("\n📋 Checking file structure...")
    
    required_files = [
        "enhanced_main_pipeline.py",
        "index.html"
    ]
    
    optional_agent_files = [
        "data_detective_agent.py",
        "feature_alchemist_agent.py", 
        "master_strategist_agent.py",
        "model_maestro_agent.py",
        "report_artisan_agent.py"
    ]
    
    missing_required = []
    for file in required_files:
        if not os.path.exists(file):
            missing_required.append(file)
        else:
            print(f"   ✅ {file}")
    
    if missing_required:
        print(f"\n❌ Missing required files: {missing_required}")
        return False
    
    missing_agents = []
    for file in optional_agent_files:
        if not os.path.exists(file):
            missing_agents.append(file)
        else:
            print(f"   ✅ {file}")
    
    if missing_agents:
        print(f"\n⚠️ Missing agent files (will run in simplified mode): {missing_agents}")
        print("   For full agent capabilities, ensure all agent files are present")
    
    return True

def test_imports():
    """Test if all imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        print("   ✅ Core packages imported successfully")
        
        # Test optional imports
        try:
            import google.generativeai
            import langchain_google_genai
            print("   ✅ AI packages imported successfully")
        except ImportError:
            print("   ⚠️ AI packages not available (will run in simplified mode)")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def create_run_script():
    """Create run scripts for different platforms"""
    print("\n📜 Creating run scripts...")
    
    # Windows batch script
    windows_script = """@echo off
echo Starting TaskPilot AI...
python enhanced_main_pipeline.py --mode api --host 0.0.0.0 --port 8000
pause
"""
    with open("run_windows.bat", "w") as f:
        f.write(windows_script)
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting TaskPilot AI..."
python3 enhanced_main_pipeline.py --mode api --host 0.0.0.0 --port 8000
"""
    with open("run_unix.sh", "w") as f:
        f.write(unix_script)
    
    # Make Unix script executable
    try:
        os.chmod("run_unix.sh", 0o755)
    except:
        pass
    
    print("   ✅ run_windows.bat created")
    print("   ✅ run_unix.sh created")
    
    return True

def main():
    """Main setup function"""
    print_banner()
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing requirements", install_requirements),
        ("Creating directories", create_directory_structure),
        ("Creating configuration", create_config_file),
        ("Creating environment template", create_environment_file),
        ("Checking file structure", check_file_structure),
        ("Testing imports", test_imports),
        ("Creating run scripts", create_run_script)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
            else:
                print(f"❌ {step_name} failed")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("\n🎉 TaskPilot AI setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Gemini API key to .env (optional but recommended)")
        print("   3. Run: python enhanced_main_pipeline.py")
        print("   4. Open browser to: http://localhost:8000")
        print("\n🚀 Ready to launch!")
    else:
        print("\n⚠️ Setup completed with some issues")
        print("   Check the errors above and resolve them")
        print("   You can still try running in simplified mode")
    
    # Create a simple test script
    test_script = '''#!/usr/bin/env python3
"""Quick test script for TaskPilot AI"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_taskpilot():
    try:
        # Test import
        from enhanced_main_pipeline import TaskPilotAI
        print("✅ TaskPilot AI imported successfully")
        
        # Test initialization
        pilot = TaskPilotAI()
        print("✅ TaskPilot AI initialized successfully")
        
        # Test basic functionality
        if hasattr(pilot, 'analyze_data'):
            print("✅ Core analysis functionality available")
        
        if pilot.agents_initialized:
            print("✅ Full agent army available")
        else:
            print("⚠️ Running in simplified mode (agents not available)")
        
        print("\\n🎉 TaskPilot AI is ready!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing TaskPilot AI...")
    result = asyncio.run(test_taskpilot())
    sys.exit(0 if result else 1)
'''
    
    with open("test_taskpilot.py", "w") as f:
        f.write(test_script)
    
    print("\n🧪 Created test_taskpilot.py - run this to test your installation")

if __name__ == "__main__":
    main()