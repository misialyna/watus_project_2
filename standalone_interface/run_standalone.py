#!/usr/bin/env python3
"""
Standalone Watus Interface Launcher
Uruchamia niezależny HTML interface dla watus_project
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def check_watus_project():
    """Sprawdź czy watus_project jest dostępny"""
    print("🔍 Checking watus_project availability...")
    
    # Check if we're in watus_project directory
    current_dir = Path.cwd()
    
    # Look for watus.py in current or parent directories
    watus_locations = [
        current_dir / "watus.py",
        current_dir.parent / "watus.py", 
        Path.home() / "watus_project" / "watus.py",
        Path("/home") / "watus_project" / "watus.py"
    ]
    
    watus_found = None
    for location in watus_locations:
        if location.exists():
            watus_found = location.parent
            print(f"✅ Found watus.py at: {location}")
            break
    
    if not watus_found:
        print("❌ watus.py not found in common locations")
        print("💡 Please run from watus_project directory or specify path")
        return False
    
    # Check for required ZMQ topics (check if watus is running)
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 7780))
        sock.close()
        
        if result == 0:
            print("✅ ZMQ publisher port 7780 is accessible")
            zmq_available = True
        else:
            print("⚠️  ZMQ publisher port 7780 not accessible")
            print("💡 Make sure watus.py is running with ZMQ publisher")
            zmq_available = False
    except Exception as e:
        print(f"⚠️  Could not check ZMQ connection: {e}")
        zmq_available = False
    
    # Check for log files
    log_files = ['watus.log', 'dialog.jsonl', 'camera.jsonl']
    existing_logs = []
    
    for log_file in log_files:
        log_path = watus_found / log_file
        if log_path.exists():
            existing_logs.append(log_file)
            print(f"✅ Found log file: {log_file}")
    
    if existing_logs:
        print(f"📄 Found {len(existing_logs)} log files for monitoring")
    else:
        print("📄 No log files found yet (they will be created when watus runs)")
    
    return True, watus_found, zmq_available

def install_dependencies():
    """Zainstaluj wymagane dependencies"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'flask',
        'flask-socketio', 
        'pyzmq',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pyzmq':
                import zmq
            elif package == 'flask':
                import flask
            elif package == 'flask-socketio':
                import flask_socketio
            elif package == 'python-dotenv':
                import dotenv
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    else:
        print("✅ All dependencies available")
        return True

def run_interface():
    """Uruchom standalone interface"""
    print("\n🚀 Starting Standalone Watus Interface...")
    
    # Import and run the interface
    try:
        import standalone_watus_interface
        interface = standalone_watus_interface.StandaloneWatusInterface()
        
        # Get configuration
        host = os.getenv('STANDALONE_INTERFACE_HOST', '127.0.0.1')
        port = int(os.getenv('STANDALONE_INTERFACE_PORT', '5001'))
        
        print(f"🌐 Interface will be available at: http://{host}:{port}")
        print("📱 Open this URL in your browser to see Watus emoticons")
        print("\n🎮 Controls:")
        print("  - Buttons: Click to manually set emotions")
        print("  - Keyboard: 1-7 for different states")
        print("  - Auto-sync: Automatically follows watus states")
        print("\n📊 Status endpoint: http://{host}:{port}/status")
        print("\n🛑 Press Ctrl+C to stop")
        print("=" * 60)
        
        interface.run_interface(host, port)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure standalone_watus_interface.py is in the same directory")
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user")
    except Exception as e:
        print(f"❌ Interface error: {e}")

def main():
    """Main launcher function"""
    print("🤖 STANDALONE WATUS HTML INTERFACE LAUNCHER")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check':
            # Just check watus_project availability
            result = check_watus_project()
            if result:
                print("\n✅ watus_project found and ready!")
            else:
                print("\n❌ watus_project not ready")
            return 0
            
        elif sys.argv[1] == '--install':
            # Just install dependencies
            if install_dependencies():
                print("\n✅ Dependencies ready!")
            else:
                print("\n❌ Dependency installation failed")
            return 0
            
        elif sys.argv[1] == '--help':
            print("Standalone Watus Interface Launcher")
            print()
            print("Usage:")
            print("  python run_standalone.py              # Run interface")
            print("  python run_standalone.py --check     # Check watus_project")
            print("  python run_standalone.py --install   # Install dependencies")
            print("  python run_standalone.py --help      # Show this help")
            print()
            print("Environment Variables:")
            print("  STANDALONE_INTERFACE_HOST=127.0.0.1")
            print("  STANDALONE_INTERFACE_PORT=5001")
            return 0
    
    # Check dependencies
    if not install_dependencies():
        print("❌ Cannot continue without dependencies")
        return 1
    
    # Check watus_project
    check_result = check_watus_project()
    
    # Run interface (even if watus_project not found - interface will show demo mode)
    if check_result:
        print("\n✅ watus_project detected - ready for real-time sync")
    else:
        print("\n⚠️  watus_project not found - running in demo mode")
    
    print("\n🎭 Starting interface in 3 seconds...")
    time.sleep(3)
    
    run_interface()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())