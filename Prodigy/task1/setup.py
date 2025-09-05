#!/usr/bin/env python3
"""
Setup script for GPT-2 Text Generation Task
This script helps with installation and verification.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All dependencies are available")
        return True

def test_model_loading():
    """Test if the model can be loaded."""
    print("\n🧪 Testing model loading...")
    try:
        from gpt2_text_generator import GPT2TextGenerator
        generator = GPT2TextGenerator()
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "fine_tuned_model",
        "processed_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")

def main():
    """Main setup function."""
    print("🚀 GPT-2 Text Generation Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n💡 Try installing manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Try reinstalling requirements:")
        print("   pip install -r requirements.txt --force-reinstall")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test model loading
    if not test_model_loading():
        print("\n💡 Model loading failed. This might be due to:")
        print("   - Network connectivity issues")
        print("   - Insufficient disk space")
        print("   - Memory constraints")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Run the demo: python demo.py")
    print("   2. Try interactive mode: python interactive_generator.py")
    print("   3. Fine-tune the model: python gpt2_text_generator.py")
    print("   4. Read the README.md for more information")

if __name__ == "__main__":
    main()
