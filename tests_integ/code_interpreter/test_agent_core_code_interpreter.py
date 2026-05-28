"""
Integration tests for the Bedrock AgentCore code interpreter platform.

These tests use the actual Bedrock AgentCore client to make real requests to the sandbox,
testing the complete end-to-end flow through the Agent interface.
"""

import logging

import pytest
from strands import Agent
from strands.agent import AgentResult
from strands_tools.code_interpreter import AgentCoreCodeInterpreter

from tests_integ.ci_environments import skip_if_github_action

logger = logging.getLogger(__name__)

@pytest.fixture
def bedrock_agent_core_code_interpreter() -> AgentCoreCodeInterpreter:
    """Create a real BedrockAgentCore code interpreter tool."""
    return AgentCoreCodeInterpreter(
        region="us-west-2",
        persist_sessions=False  # Don't persist for integration tests
    )


@pytest.fixture
def agent(bedrock_agent_core_code_interpreter: AgentCoreCodeInterpreter) -> Agent:
    """Create an agent with the BedrockAgentCore code interpreter tool."""
    return Agent(tools=[bedrock_agent_core_code_interpreter.code_interpreter])


@skip_if_github_action.mark
def test_direct_tool_call(agent):
    """Test code interpreter direct tool call."""

    result = agent.tool.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "initSession",
                "description": "Data analysis session",
                "session_name": "analysissession"  # Cleaned name (no dashes)
            }
        }
    )

    assert result['status'] == 'success'


@skip_if_github_action.mark
def test_complex_natural_language_workflow(agent):
    """Test complex multistep workflow through natural language instructions."""
    
    result: AgentResult = agent("""
You have code execution capabilities. Complete this workflow efficiently by combining operations where possible:

1. Create a Python script 'data_generator.py' that generates a CSV file 'sales_data.csv' with 20 rows (date, product, quantity, price, customer_id columns) and executes it

2. Verify the CSV was created by reading 'sales_data.csv'

3. Create and run a shell script 'cleanup.sh' that lists all files

After ALL steps complete successfully, respond with ONLY the word "PASS". If any step fails, respond with "FAIL".
    """)

    assert "PASS" in result.message["content"][0]["text"]

@skip_if_github_action.mark
def test_auto_session_creation(bedrock_agent_core_code_interpreter):
    """Test automatic session creation on first code execution."""
    # Execute code directly without initializing session
    result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode",
                "code": "import platform\nprint(f'Running on Python {platform.python_version()}')",
                "language": "python"
            }
        }
    )
    
    assert result['status'] == 'success'
    
    # Verify a session was auto-created (will be random UUID, not "default")
    assert len(bedrock_agent_core_code_interpreter._sessions) == 1
    auto_created_session = list(bedrock_agent_core_code_interpreter._sessions.keys())[0]
    assert auto_created_session.startswith("session")  # Check pattern instead of exact name
    
    # Execute a second command in the same auto-created session
    result2 = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode",
                "code": "print('Second execution in auto-created session')",
                "language": "python"
            }
        }
    )
    
    assert result2['status'] == 'success'
    
    # Verify still only one session exists (reused the auto-created one)
    assert len(bedrock_agent_core_code_interpreter._sessions) == 1


@skip_if_github_action.mark 
def test_download_files_integration(bedrock_agent_core_code_interpreter):
    """Test end-to-end file download from sandbox to local filesystem."""
    import json
    import tempfile
    import os
    from pathlib import Path
    
    # Initialize session
    init_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "initSession", 
                "description": "File download test session",
                "session_name": "downloadtest"
            }
        }
    )
    assert init_result['status'] == 'success'
    
    # Create test files in the sandbox
    create_files_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode",
                "session_name": "downloadtest",
                "code": """
import json
import csv
from pathlib import Path

# Create a text file
with open('test_data.txt', 'w') as f:
    f.write('Hello from AgentCore Code Interpreter!\\nThis is a test file for download functionality.')

# Create a JSON file
data = {
    'message': 'Download test successful',
    'numbers': [1, 2, 3, 4, 5],
    'nested': {'key': 'value', 'bool': True}
}
with open('test_data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Create a CSV file
csv_data = [
    ['Name', 'Age', 'City'],
    ['Alice', 30, 'New York'],
    ['Bob', 25, 'London'], 
    ['Charlie', 35, 'Tokyo']
]
with open('test_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print('Created test files: test_data.txt, test_data.json, test_data.csv')
""",
                "language": "python"
            }
        }
    )
    assert create_files_result['status'] == 'success'
    
    # List files to verify they were created
    list_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "listFiles",
                "session_name": "downloadtest",
                "path": "."
            }
        }
    )
    assert list_result['status'] == 'success'
    
    # Download the files to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        download_result = bedrock_agent_core_code_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "downloadFiles",
                    "session_name": "downloadtest",
                    "source_paths": ["test_data.txt", "test_data.json", "test_data.csv"],
                    "destination_dir": temp_dir
                }
            }
        )
        
        # Verify download was successful
        if download_result['status'] != 'success':
            print(f"Download failed: {download_result}")
        assert download_result['status'] == 'success', f"Download failed with: {download_result}"
        assert download_result['content'][0]['json']['totalFiles'] == 3
        
        downloaded_files = download_result['content'][0]['json']['downloadedFiles']
        assert len(downloaded_files) == 3
        
        # Verify all files were downloaded with correct paths and sizes
        local_files = {}
        for file_info in downloaded_files:
            source_path = file_info['sourcePath'] 
            local_path = file_info['localPath']
            file_size = file_info['size']
            
            local_files[source_path] = {
                'path': local_path,
                'size': file_size
            }
            
            # Verify file exists locally
            assert Path(local_path).exists(), f"Downloaded file {local_path} does not exist"
            assert Path(local_path).stat().st_size == file_size, f"File size mismatch for {local_path}"
        
        # Verify specific file contents
        txt_file_path = local_files['test_data.txt']['path']
        with open(txt_file_path, 'r') as f:
            txt_content = f.read()
        assert 'Hello from AgentCore Code Interpreter!' in txt_content
        assert 'This is a test file for download functionality.' in txt_content
        
        json_file_path = local_files['test_data.json']['path']
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        assert json_data['message'] == 'Download test successful'
        assert json_data['numbers'] == [1, 2, 3, 4, 5]
        assert json_data['nested']['key'] == 'value'
        
        csv_file_path = local_files['test_data.csv']['path']
        with open(csv_file_path, 'r') as f:
            csv_content = f.read()
        assert 'Name,Age,City' in csv_content
        assert 'Alice,30,New York' in csv_content
        assert 'Charlie,35,Tokyo' in csv_content


@skip_if_github_action.mark
def test_download_files_with_binary_data(bedrock_agent_core_code_interpreter):
    """Test downloading binary files (images) from sandbox."""
    import tempfile
    import os
    from pathlib import Path
    
    # Initialize session
    init_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "initSession",
                "description": "Binary file download test", 
                "session_name": "binarydownloadtest"
            }
        }
    )
    assert init_result['status'] == 'success'
    
    # Create a simple binary file (PNG-like structure) in the sandbox
    create_binary_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode", 
                "code": """
import struct

# Create a minimal binary file with PNG-like header
png_header = b'\\x89PNG\\r\\n\\x1a\\n'  # PNG file signature
additional_data = b'\\x00\\x00\\x00\\rIHDR' + b'\\x00' * 20  # Mock IHDR chunk

binary_data = png_header + additional_data

with open('test_image.png', 'wb') as f:
    f.write(binary_data)

print(f'Created binary file: test_image.png ({len(binary_data)} bytes)')

# Also create a simple binary data file
with open('binary_data.bin', 'wb') as f:
    # Write some binary patterns
    for i in range(256):
        f.write(bytes([i]))

print('Created binary_data.bin (256 bytes)')
""",
                "language": "python"
            }
        }
    )
    assert create_binary_result['status'] == 'success'
    
    # Download binary files
    with tempfile.TemporaryDirectory() as temp_dir:
        download_result = bedrock_agent_core_code_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "downloadFiles",
                    "session_name": "binarydownloadtest", 
                    "source_paths": ["test_image.png", "binary_data.bin"],
                    "destination_dir": temp_dir
                }
            }
        )
        
        assert download_result['status'] == 'success'
        assert download_result['content'][0]['json']['totalFiles'] == 2
        
        downloaded_files = download_result['content'][0]['json']['downloadedFiles']
        
        # Verify PNG file
        png_file = next(f for f in downloaded_files if f['sourcePath'] == 'test_image.png')
        png_path = Path(png_file['localPath'])
        assert png_path.exists()
        
        with open(png_path, 'rb') as f:
            png_content = f.read()
        assert png_content.startswith(b'\\x89PNG\\r\\n\\x1a\\n'), "PNG header not preserved"
        
        # Verify binary data file
        bin_file = next(f for f in downloaded_files if f['sourcePath'] == 'binary_data.bin')
        bin_path = Path(bin_file['localPath'])
        assert bin_path.exists()
        assert bin_file['size'] == 256
        
        with open(bin_path, 'rb') as f:
            bin_content = f.read()
        assert len(bin_content) == 256
        # Check that we have all bytes from 0-255
        assert bin_content == bytes(range(256)), "Binary data integrity check failed"


@skip_if_github_action.mark  
def test_download_files_error_handling(bedrock_agent_core_code_interpreter):
    """Test download error handling for non-existent files."""
    import tempfile
    
    # Initialize session
    init_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "initSession",
                "description": "Error handling test",
                "session_name": "errortest"
            }
        }
    )
    assert init_result['status'] == 'success'
    
    # Try to download non-existent files
    with tempfile.TemporaryDirectory() as temp_dir:
        download_result = bedrock_agent_core_code_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "downloadFiles",
                    "session_name": "errortest",
                    "source_paths": ["nonexistent1.txt", "nonexistent2.csv"],
                    "destination_dir": temp_dir
                }
            }
        )
        
        # Should return error status when all files fail
        assert download_result['status'] == 'error'
        assert 'All downloads failed' in download_result['content'][0]['text']
        assert 'File not found' in download_result['content'][0]['text']


@skip_if_github_action.mark
def test_download_files_mixed_success_failure(bedrock_agent_core_code_interpreter):
    """Test download with mix of existing and non-existent files."""
    import tempfile
    import json
    from pathlib import Path
    
    # Initialize session
    init_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "initSession",
                "description": "Mixed results test", 
                "session_name": "mixedtest"
            }
        }
    )
    assert init_result['status'] == 'success'
    
    # Create only one file
    create_result = bedrock_agent_core_code_interpreter.code_interpreter(
        code_interpreter_input={
            "action": {
                "type": "executeCode",
                "code": """
with open('exists.txt', 'w') as f:
    f.write('This file exists and should be downloaded successfully.')
print('Created exists.txt')
""",
                "language": "python"
            }
        }
    )
    assert create_result['status'] == 'success'
    
    # Try to download mix of existing and non-existing files
    with tempfile.TemporaryDirectory() as temp_dir:
        download_result = bedrock_agent_core_code_interpreter.code_interpreter(
            code_interpreter_input={
                "action": {
                    "type": "downloadFiles",
                    "session_name": "mixedtest", 
                    "source_paths": ["exists.txt", "missing.txt", "also_missing.csv"],
                    "destination_dir": temp_dir
                }
            }
        )
        
        # Should succeed partially
        assert download_result['status'] == 'success'
        
        result_data = download_result['content'][0]['json']
        assert result_data['totalFiles'] == 1  # Only one successful download
        assert 'errors' in result_data
        assert len(result_data['errors']) == 2  # Two failed files
        
        # Verify the successful download
        downloaded_files = result_data['downloadedFiles']
        assert len(downloaded_files) == 1
        assert downloaded_files[0]['sourcePath'] == 'exists.txt'
        
        # Verify the file actually exists and has correct content
        local_path = Path(downloaded_files[0]['localPath'])
        assert local_path.exists()
        
        with open(local_path, 'r') as f:
            content = f.read()
        assert 'This file exists and should be downloaded successfully.' in content
        
        # Check error messages
        error_messages = ' '.join(result_data['errors'])
        assert 'missing.txt' in error_messages
        assert 'also_missing.csv' in error_messages
