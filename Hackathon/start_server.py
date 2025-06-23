#!/usr/bin/env python3
"""
CrewAI SQL Assistant - FastAPI Server Startup Script
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_directories():
    """Create necessary directories"""
    directories = ['static', 'uploads', 'exports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' ready")

def check_environment_file():
    """Check and create .env file if needed"""
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  No .env file found. Creating template...")
        
        env_template = """# CrewAI SQL Assistant Environment Variables
# Add your API keys here

# Groq API Key (Recommended - Fast and Free)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Database Settings
DATABASE_PATH=./databases/
"""
        
        with open('.env', 'w') as f:
            f.write(env_template)
        
        print("📝 Created .env template file")
        print("🔑 Please add your API keys to the .env file for full functionality")
        print("💡 You can also run without API keys using the Mock LLM option")
    else:
        print("✅ Environment file found")

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        print("💡 Try running: pip install -r requirements.txt manually")
        return False
    return True

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        'main_api.py',
        'main.py',
        'static/index.html',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def start_server():
    """Start the FastAPI server"""
    try:
        print("\n🚀 Starting CrewAI SQL Assistant Server...")
        print("📡 Server will be available at: http://localhost:8000")
        print("🔗 Chat Interface: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("\n⏰ Server starting in 3 seconds...")
        time.sleep(3)
        
        # Open browser automatically
        webbrowser.open('http://localhost:8000')
        
        # Start the server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'main_api:app', 
            '--host', '0.0.0.0', 
            '--port', '8000', 
            '--reload'
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║    🧠 CrewAI SQL Assistant - Professional Chat UI       ║
║                                                          ║
║    • Multi-Agent SQL Analysis                           ║
║    • Role-Based Access Control                          ║
║    • Real-time Chat Interface                           ║
║    • Multiple LLM Provider Support                      ║
║    • Human-in-the-Loop Feedback                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_usage_tips():
    """Print usage tips"""
    tips = """
🎯 Quick Start Tips:

1. 🔑 API Keys (Optional but Recommended):
   • Add your Groq API key to .env for best performance
   • Groq offers free API access: https://console.groq.com/
   • You can also use OpenAI, Anthropic, or Mock LLM

2. 💬 Chat Interface Features:
   • Natural language queries: "Show me top 10 employees by salary"
   • Chart generation: Check "Generate Chart" option
   • File uploads: Drag & drop CSV/Excel files
   • Feedback system: Click "Provide Feedback" to improve results

3. 👤 User Roles:
   • Admin: Full access to all data
   • Analyst: Limited access, no sensitive data
   • Viewer: Read-only, aggregated data only

4. 🗄️ Database Support:
   • Multiple databases simultaneously
   • File uploads (CSV, Excel, JSON)
   • Cross-database queries

5. 🔧 Configuration:
   • Change LLM provider in real-time
   • Switch user roles
   • Select multiple databases

📝 Need help? Check the API docs at http://localhost:8000/docs
"""
    print(tips)

def main():
    """Main startup function"""
    print_banner()
    
    print("🔍 System Check...")
    check_python_version()
    
    if not check_required_files():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        sys.exit(1)
    
    create_directories()
    check_environment_file()
    
    print("\n📦 Dependency Check...")
    if not install_requirements():
        response = input("\n❓ Continue without installing requirements? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print_usage_tips()
    
    input("\n🚀 Press Enter to start the server...")
    start_server()

if __name__ == "__main__":
    main()