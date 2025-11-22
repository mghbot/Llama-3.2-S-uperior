#!/usr/bin/env python3
"""
Validate project structure without requiring dependencies
"""

import os
import sys

def validate_structure():
    """Check that all required files exist"""
    print("Validating project structure...")
    print()

    required_files = [
        "config.py",
        "modules.py",
        "surgery.py",
        "train.py",
        "benchmark.py",
        "main.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "test_imports.py",
        "quick_start.sh",
    ]

    missing = []
    for filename in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename:25s} ({size:,} bytes)")
        else:
            missing.append(filename)
            print(f"✗ {filename:25s} MISSING")

    print()
    print("="*60)

    if missing:
        print(f"FAILED: {len(missing)} files missing")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print("SUCCESS: All required files present")
        print()
        print("Project is ready!")
        print()
        print("To get started:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Test imports: python3 test_imports.py")
        print("  3. See README.md for usage instructions")
        return True

    print("="*60)


if __name__ == "__main__":
    success = validate_structure()
    sys.exit(0 if success else 1)
