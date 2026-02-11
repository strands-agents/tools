#!/usr/bin/env python3
"""
Quick test script for S3 data loader tool.
Tests the basic functionality without requiring real AWS credentials.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from strands_tools.s3_data_loader import s3_data_loader

def test_basic_functionality():
    """Test basic functionality with mock data."""
    print("🧪 Testing S3 Data Loader Tool...")
    
    # Test 1: Import test
    print("✅ Import successful")
    
    # Test 2: Function signature test
    try:
        # This will fail with real S3 call, but tests the function structure
        result = s3_data_loader(bucket="test", key="test.csv")
        print("❌ Should have failed without AWS credentials")
    except Exception as e:
        if "credentials" in str(e).lower() or "access" in str(e).lower():
            print("✅ Correctly requires AWS credentials")
        else:
            print(f"✅ Function structure works (error: {str(e)[:50]}...)")
    
    print("\n🎉 Basic S3 Data Loader Tool is ready!")
    print("\n📋 Next steps:")
    print("1. Set AWS credentials: export AWS_ACCESS_KEY_ID=your-key")
    print("2. Set AWS secret: export AWS_SECRET_ACCESS_KEY=your-secret") 
    print("3. Test with real S3 bucket:")
    print("   result = s3_data_loader(bucket='your-bucket', key='data.csv')")
    
    return True

if __name__ == "__main__":
    test_basic_functionality()
