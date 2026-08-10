#!/usr/bin/env python3
"""
Setup Verification Script
Verifies all imports and dependencies are correctly configured
"""

import sys
import os

print("=" * 70)
print("🔍 VERIFYING PROJECT SETUP")
print("=" * 70)
print()

# Test 1: Verify folder structure
print("TEST 1: Folder Structure")
print("-" * 70)

folders_to_check = ['app', 'src', 'notebooks']
for folder in folders_to_check:
    exists = os.path.exists(folder) and os.path.isdir(folder)
    status = "✅" if exists else "❌"
    print(f"{status} {folder}/ {'exists' if exists else 'MISSING'}")

files_to_check = ['app/app.py', 'src/mcp_server.py', 'app.yaml', 'requirements.txt']
for file in files_to_check:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"{status} {file} {'exists' if exists else 'MISSING'}")

print()

# Test 2: Verify imports
print("TEST 2: Import Resolution")
print("-" * 70)

# Add src to path
sys.path.insert(0, 'src')

try:
    from mcp_server import MCPToolServer
    print("✅ MCPToolServer imported successfully")
    
    # Check methods exist
    server = MCPToolServer()
    tools = server.list_tools()
    print(f"✅ MCP Server initialized with {len(tools['tools'])} tools")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Check dependencies
print("TEST 3: Required Dependencies")
print("-" * 70)

required_packages = [
    'streamlit', 'pandas', 'plotly', 'databricks', 
    'mlflow', 'numpy', 'requests'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - NOT INSTALLED")

print()

# Test 4: App.yaml validation
print("TEST 4: Configuration Files")
print("-" * 70)

try:
    with open('app.yaml', 'r') as f:
        content = f.read()
        has_app_py = 'app/app.py' in content
        has_streamlit = 'streamlit' in content
        
        if has_app_py:
            print("✅ app.yaml points to app/app.py")
        else:
            print("❌ app.yaml entry point incorrect")
            
        if has_streamlit:
            print("✅ app.yaml uses streamlit command")
        else:
            print("❌ app.yaml missing streamlit command")
except Exception as e:
    print(f"❌ Failed to read app.yaml: {e}")

print()
print("=" * 70)
print("✨ VERIFICATION COMPLETE")
print("=" * 70)
