#!/usr/bin/env python3
"""
Script to push the counts_kl checkpoints directory to Hugging Face Hub.

This script will:
1. Install required dependencies
2. Authenticate with Hugging Face
3. Create or update a repository on Hugging Face Hub
4. Upload all checkpoints from the counts_kl directory
5. Provide progress updates and error handling

Usage:
    python push_to_huggingface.py [--repo-name REPO_NAME] [--private]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result

def install_dependencies():
    """Install required dependencies for Hugging Face Hub."""
    print("Installing required dependencies...")
    
    # Install huggingface_hub if not already installed
    try:
        __import__("huggingface_hub")
        print("huggingface_hub already installed")
    except ImportError:
        print("Installing huggingface_hub...")
        run_command("pip install huggingface_hub[cli]")
    
    # Install git-lfs if not available
    result = run_command("git lfs version", check=False)
    if result.returncode != 0:
        print("Installing git-lfs...")
        run_command("apt-get update && apt-get install -y git-lfs")
        run_command("git lfs install")

def authenticate_huggingface():
    """Authenticate with Hugging Face Hub."""
    print("Setting up Hugging Face authentication...")
    
    # Check if already authenticated
    result = run_command("huggingface-cli whoami", check=False)
    if result.returncode == 0:
        print(f"Already authenticated as: {result.stdout.strip()}")
        return
    
    print("Please authenticate with Hugging Face:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'write' permissions")
    print("3. Run: huggingface-cli login")
    print("4. Enter your token when prompted")
    
    # Try to run the login command
    run_command("huggingface-cli login")

def create_repository(repo_name: str, private: bool = False):
    """Create a new repository on Hugging Face Hub."""
    print(f"Creating repository: {repo_name}")
    
    visibility = "private" if private else "public"
    cmd = f"huggingface-cli repo create {repo_name} --type model --{visibility}"
    
    result = run_command(cmd, check=False)
    if result.returncode == 0:
        print(f"Repository created successfully: https://huggingface.co/{repo_name}")
    else:
        # Repository might already exist
        if "already exists" in result.stderr.lower():
            print(f"Repository {repo_name} already exists, continuing...")
        else:
            print(f"Error creating repository: {result.stderr}")
            sys.exit(1)

def get_checkpoint_dirs(checkpoints_path: Path) -> list:
    """Get all checkpoint directories sorted by step number."""
    checkpoint_dirs = []
    for item in checkpoints_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step_num = int(item.name.split("-")[1])
                checkpoint_dirs.append((step_num, item))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse checkpoint name: {item.name}")
                continue
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    return [str(item[1]) for item in checkpoint_dirs]

def upload_checkpoints(repo_name: str, checkpoints_path: Path, max_checkpoints: Optional[int] = None):
    """Upload all checkpoints to the Hugging Face repository."""
    print(f"Uploading checkpoints from: {checkpoints_path}")
    
    # Get all checkpoint directories
    checkpoint_dirs = get_checkpoint_dirs(checkpoints_path)
    
    if not checkpoint_dirs:
        print("No checkpoint directories found!")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories")
    
    if max_checkpoints:
        checkpoint_dirs = checkpoint_dirs[:max_checkpoints]
        print(f"Limiting to first {max_checkpoints} checkpoints")
    
    # Upload each checkpoint
    for i, checkpoint_dir in enumerate(checkpoint_dirs, 1):
        checkpoint_name = Path(checkpoint_dir).name
        print(f"\n[{i}/{len(checkpoint_dirs)}] Uploading {checkpoint_name}...")
        
        # Upload the checkpoint directory
        cmd = f"huggingface-cli upload {repo_name} {checkpoint_dir} {checkpoint_name} --repo-type model"
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✓ Successfully uploaded {checkpoint_name}")
        else:
            print(f"✗ Failed to upload {checkpoint_name}: {result.stderr}")
            # Continue with other checkpoints even if one fails
    
    print(f"\nUpload completed! Repository: https://huggingface.co/{repo_name}")

def upload_readme(repo_name: str, readme_path: Path):
    """Upload the README.md file to the repository."""
    if readme_path.exists():
        print("Uploading README.md...")
        cmd = f"huggingface-cli upload {repo_name} {readme_path} README.md --repo-type model"
        result = run_command(cmd, check=False)
        if result.returncode == 0:
            print("✓ README.md uploaded successfully")
        else:
            print(f"✗ Failed to upload README.md: {result.stderr}")
    else:
        print("No README.md found in checkpoints directory")

def main():
    parser = argparse.ArgumentParser(description="Push counts_kl checkpoints to Hugging Face Hub")
    parser.add_argument("--repo-name", default="model-organisms-cot-pathologies/counts-kl-checkpoints", 
                       help="Repository name (default: model-organisms-cot-pathologies/counts-kl-checkpoints)")
    parser.add_argument("--private", action="store_true", 
                       help="Create a private repository")
    parser.add_argument("--max-checkpoints", type=int, 
                       help="Maximum number of checkpoints to upload (for testing)")
    parser.add_argument("--checkpoints-dir", default="/root/model-organisms-cot-pathologies/counts_kl",
                       help="Path to checkpoints directory")
    
    args = parser.parse_args()
    
    # Validate checkpoints directory
    checkpoints_path = Path(args.checkpoints_dir)
    if not checkpoints_path.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Hugging Face Checkpoints Upload Script")
    print("=" * 60)
    print(f"Repository: {args.repo_name}")
    print(f"Checkpoints directory: {checkpoints_path}")
    print(f"Private repository: {args.private}")
    if args.max_checkpoints:
        print(f"Max checkpoints: {args.max_checkpoints}")
    print("=" * 60)
    
    try:
        # Step 1: Install dependencies
        install_dependencies()
        
        # Step 2: Authenticate with Hugging Face
        authenticate_huggingface()
        
        # Step 3: Create repository
        create_repository(args.repo_name, args.private)
        
        # Step 4: Upload README
        readme_path = checkpoints_path / "README.md"
        upload_readme(args.repo_name, readme_path)
        
        # Step 5: Upload checkpoints
        upload_checkpoints(args.repo_name, checkpoints_path, args.max_checkpoints)
        
        print("\n" + "=" * 60)
        print("Upload completed successfully!")
        print(f"Repository: https://huggingface.co/{args.repo_name}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nUpload interrupted by user")
        sys.exit(1)
    except (OSError, subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"\nError during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
