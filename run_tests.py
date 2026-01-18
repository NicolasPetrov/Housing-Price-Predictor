"""
Script to run all tests
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Running tests with verbose output"""
    
    print("🧪 Running Housing Price Predictor Tests")
    print("=" * 50)
    
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return 1
    
    print(f"📁 Running tests from: {test_dir}")
    print()
    
    pytest_args = [
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes",
        "-x",
    ]
    
    exit_code = pytest.main(pytest_args)
    
    print()
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())