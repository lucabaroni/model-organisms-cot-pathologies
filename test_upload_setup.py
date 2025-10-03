#!/usr/bin/env python3
"""
Test script to verify the upload setup before running the main upload script.
"""

import subprocess
import sys
from pathlib import Path

def test_dependencies():
    """Test if required dependencies are available."""
    print("Testing dependencies...")
    
    # Test huggingface_hub
    try:
        result = subprocess.run(["python", "-c", "import huggingface_hub; print('huggingface_hub available')"], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✓ huggingface_hub is available")
        else:
            print("✗ huggingface_hub not available")
            return False
    except Exception as e:
        print(f"✗ Error testing huggingface_hub: {e}")
        return False
    
    # Test git-lfs
    try:
        result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✓ git-lfs is available")
        else:
            print("✗ git-lfs not available")
            return False
    except Exception as e:
        print(f"✗ Error testing git-lfs: {e}")
        return False
    
    return True

def test_authentication():
    """Test Hugging Face authentication."""
    print("\nTesting Hugging Face authentication...")
    
    try:
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"✓ Authenticated as: {result.stdout.strip()}")
            return True
        else:
            print("✗ Not authenticated with Hugging Face")
            print("  Run: huggingface-cli login")
            return False
    except Exception as e:
        print(f"✗ Error testing authentication: {e}")
        return False

def test_checkpoints_directory():
    """Test if checkpoints directory exists and has content."""
    print("\nTesting checkpoints directory...")
    
    checkpoints_path = Path("/root/model-organisms-cot-pathologies/counts_kl")
    
    if not checkpoints_path.exists():
        print(f"✗ Checkpoints directory not found: {checkpoints_path}")
        return False
    
    print(f"✓ Checkpoints directory exists: {checkpoints_path}")
    
    # Count checkpoint directories
    checkpoint_dirs = [d for d in checkpoints_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    print(f"✓ Found {len(checkpoint_dirs)} checkpoint directories")
    
    if len(checkpoint_dirs) == 0:
        print("✗ No checkpoint directories found")
        return False
    
    # Check if README exists
    readme_path = checkpoints_path / "README.md"
    if readme_path.exists():
        print("✓ README.md found")
    else:
        print("⚠ README.md not found (optional)")
    
    return True

def main():
    print("=" * 50)
    print("Upload Setup Test")
    print("=" * 50)
    
    all_good = True
    
    # Test dependencies
    if not test_dependencies():
        all_good = False
    
    # Test authentication
    if not test_authentication():
        all_good = False
    
    # Test checkpoints directory
    if not test_checkpoints_directory():
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✓ All tests passed! Ready to upload.")
        print("Run: python push_to_huggingface.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 50)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
