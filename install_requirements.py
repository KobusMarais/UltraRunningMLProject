#!/usr/bin/env python3
"""
Requirements Installation Script

This script installs the required dependencies for the ultramarathon pace prediction project.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install the required Python packages."""
    
    print("=" * 60)
    print("INSTALLING REQUIREMENTS")
    print("=" * 60)
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        print("Please ensure requirements.txt is in the project root directory")
        return False
    
    print(f"📁 Found requirements file: {requirements_file}")
    
    # Read requirements
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📦 Found {len(requirements)} packages to install:")
    for req in requirements:
        print(f"  - {req}")
    
    # Install packages
    print("\n🚀 Installing packages...")
    try:
        for package in requirements:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"⚠️  {package} installation had warnings")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during installation: {e}")
        return False
    
    # Verify installation
    print("\n🔍 Verifying installations...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import lightgbm as lgb
        
        print("✅ All core packages imported successfully!")
        print(f"   - pandas: {pd.__version__}")
        print(f"   - numpy: {np.__version__}")
        print(f"   - matplotlib: {matplotlib.__version__ if hasattr(matplotlib, '__version__') else 'installed'}")
        print(f"   - seaborn: {sns.__version__ if hasattr(sns, '__version__') else 'installed'}")
        print(f"   - scikit-learn: {sklearn.__version__}")
        print(f"   - lightgbm: {lgb.__version__}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some packages may not have installed correctly")
        return False
    
    print("\n🎉 Requirements installation completed successfully!")
    return True


def check_environment():
    """Check the current Python environment."""
    
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not running in a virtual environment")
        print("   Consider creating a virtual environment for this project:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    
    return True


def main():
    """Main installation function."""
    
    # Check environment first
    check_environment()
    
    print("\n")
    
    # Install requirements
    success = install_requirements()
    
    if success:
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print("You can now run the pipeline with:")
        print("python run_pipeline.py")
        print("\nOr run individual components:")
        print("python notebooks/example_usage.ipynb")
    else:
        print("\n" + "=" * 60)
        print("SETUP FAILED!")
        print("=" * 60)
        print("Please check the error messages above and try again.")
        print("You may need to:")
        print("1. Install missing system dependencies")
        print("2. Use a virtual environment")
        print("3. Install packages manually with: pip install <package_name>")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
