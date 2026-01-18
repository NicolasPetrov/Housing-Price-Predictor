"""
Configuration for pytest
"""

import pytest
import sys
import os
import warnings

project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setting up a test environment"""
    os.environ['LOG_LEVEL'] = 'ERROR'  
    os.environ['LOG_FILE_ENABLED'] = 'false'
    
    yield
    
    test_files = [
        'test_model.pkl',
        'test_preprocessor.pkl'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)