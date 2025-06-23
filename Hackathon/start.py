#!/usr/bin/env python3
"""
Enhanced CrewAI SQL Assistant - Startup Script
Filename: start.py

Simple startup script to launch the enhanced CrewAI SQL Assistant
with proper error handling and setup verification.
"""

import os
import sys
import subprocess
import importlib.util

def check_file_exists(filename, description):
    """Check if required file exists"""
    if os.path.exists(filename):
        print(f"✅ {description}: {filename}")
        return True
    else:
        print(f"❌ Missing {description}: {filename}")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        ('fastapi', 'FastAPI web framework'),
        ('uvicorn', 'ASGI server'),
        ('pandas', 'Data processing'),
        ('crewai', 'CrewAI framework'),
        ('langchain', 'LangChain framework')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        if importlib.util.find_spec(package) is None:
            print(f"❌ Missing package: {package} ({description})")
            missing_packages.append(package)
        else:
            print(f"✅ Package installed: {package}")
    
    return missing_packages

def main():
    """Main startup function"""
    print("🚀 Enhanced CrewAI SQL Assistant - Startup Check")
    print("=" * 60)
    
    # Check required files
    files_ok = True
    files_ok &= check_file_exists("main.py", "Core system")
    files_ok &= check_file_exists("main_api.py", "API server")
    files_ok &= check_file_exists("index.html", "Web interface")
    
    print("\n📦 Checking Python Dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} required packages")
        print("📥 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n💡 Or install all dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    
    if not files_ok:
        print("\n❌ Required files are missing")
        print("📋 Please ensure you have:")
        print("   - main.py (core system)")
        print("   - main_api.py (API server)")  
        print("   - index.html (web interface)")
        return False
    
    print("\n✅ All checks passed!")
    print("🌐 Starting Enhanced CrewAI SQL Assistant...")
    print("🔗 Interface will be available at: http://localhost:8000")
    print("=" * 60)
    
    # Start the server
    try:
        subprocess.run([sys.executable, "main_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Python not found in PATH")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)