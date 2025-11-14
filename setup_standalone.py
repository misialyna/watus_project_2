#!/usr/bin/env python3
"""
Quick Setup Script for Standalone Watus Interface
Automatycznie konfiguruje standalone interface w katalogu watus_project
"""

import os
import shutil
import sys
from pathlib import Path

def setup_standalone_interface():
    """Setup standalone interface w aktualnym katalogu"""
    
    print("🤖 STANDALONE WATUS INTERFACE - QUICK SETUP")
    print("=" * 50)
    
    current_dir = Path.cwd()
    
    # Sprawdź czy jesteśmy w watus_project
    if not (current_dir / "watus.py").exists():
        print("❌ Nie znaleziono watus.py w aktualnym katalogu")
        print("💡 Uruchom z katalogu watus_project")
        print(f"   cd /path/to/your/watus_project")
        print(f"   python {__file__}")
        return False
    
    print(f"✅ Znaleziono watus_project w: {current_dir}")
    
    # Utwórz katalog dla standalone interface
    standalone_dir = current_dir / "standalone_interface"
    standalone_dir.mkdir(exist_ok=True)
    
    # Skopiuj pliki
    files_to_copy = [
        ("standalone_watus_interface.py", standalone_dir / "standalone_watus_interface.py"),
        ("run_standalone.py", standalone_dir / "run_standalone.py"),
        ("templates/watus_face.html", standalone_dir / "watus_face.html")
    ]
    
    print("\n📁 Kopiowanie plików...")
    for src, dst in files_to_copy:
        if Path(src).exists():
            shutil.copy2(src, dst)
            print(f"✅ {src} → {dst}")
        else:
            print(f"❌ {src} nie istnieje")
    
    # Utwórz requirements.txt
    requirements_content = """# Standalone Watus Interface Requirements
Flask>=3.0.0
Flask-SocketIO>=5.0.0
pyzmq>=25.0.0
python-dotenv>=1.0.0
eventlet>=0.35.0
"""
    
    requirements_file = standalone_dir / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)
    print(f"✅ {requirements_file} utworzony")
    
    # Utwórz start script
    start_script_content = f"""#!/usr/bin/env python3
\"\"\"
Standalone Watus Interface Starter
\"\"\"

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run interface
from standalone_watus_interface import StandaloneWatusInterface

if __name__ == "__main__":
    interface = StandaloneWatusInterface()
    interface.run_interface(
        host=os.getenv('STANDALONE_INTERFACE_HOST', '127.0.0.1'),
        port=int(os.getenv('STANDALONE_INTERFACE_PORT', '5001'))
    )
"""
    
    start_file = standalone_dir / "start_interface.py"
    with open(start_file, 'w') as f:
        f.write(start_script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(start_file, 0o755)
    
    print(f"✅ {start_file} utworzony")
    
    print("\n📋 NASTĘPNE KROKI:")
    print("1. Zainstaluj dependencies:")
    print(f"   cd {standalone_dir}")
    print("   pip install -r requirements.txt")
    print()
    print("2. Uruchom interface:")
    print(f"   cd {standalone_dir}")
    print("   python start_interface.py")
    print()
    print("3. Otwórz w przeglądarce:")
    print("   http://127.0.0.1:5001")
    print()
    print("4. Uruchom watus (w osobnym terminalu):")
    print("   python watus.py")
    
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Standalone Watus Interface - Quick Setup")
            print()
            print("Usage:")
            print(f"  python {__file__}           # Setup standalone interface")
            print(f"  python {__file__} --help    # Show this help")
            print()
            print("This script will:")
            print("✅ Copy standalone interface files to your watus_project")
            print("✅ Create requirements.txt with dependencies")
            print("✅ Create start script for easy launching")
            print("✅ Provide step-by-step instructions")
            return 0
    
    try:
        if setup_standalone_interface():
            print("\n🎉 SETUP ZAKOŃCZONY POMYŚLNIE!")
            print("\n📖 Więcej informacji w pliku: STANDALONE_INTERFACE_GUIDE.md")
            return 0
        else:
            print("\n❌ SETUP FAILED")
            return 1
    except KeyboardInterrupt:
        print("\n🛑 Setup przerwany przez użytkownika")
        return 1
    except Exception as e:
        print(f"\n❌ Błąd podczas setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())