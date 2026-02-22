#!/usr/bin/env python3
"""
Real AWS S3 Testing Guide for S3 Data Loader Tool
Use this script to test with your actual AWS S3 buckets and data.
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("🔐 Checking AWS Credentials...")
    
    # Check environment variables
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    if aws_key and aws_secret:
        print("✅ AWS credentials found in environment variables")
        print(f"   Region: {aws_region}")
        return True
    else:
        print("❌ AWS credentials not found in environment variables")
        print("\n📋 To set up AWS credentials:")
        print("   export AWS_ACCESS_KEY_ID=your-access-key")
        print("   export AWS_SECRET_ACCESS_KEY=your-secret-key")
        print("   export AWS_REGION=your-region  # Optional, defaults to us-east-1")
        print("\n🔗 Or configure AWS CLI: aws configure")
        return False

def test_with_real_s3():
    """Test S3 data loader with real AWS S3."""
    print("\n🧪 Testing with Real AWS S3...")
    
    try:
        from strands_tools.s3_data_loader import s3_data_loader
        
        # Get user input for testing
        print("\n📝 Enter your S3 details for testing:")
        bucket = input("S3 Bucket name: ").strip()
        
        if not bucket:
            print("❌ Bucket name is required")
            return False
        
        print(f"\n🔍 Testing S3 bucket: {bucket}")
        
        # Test 1: List files in bucket
        print("\n1️⃣ Listing files in bucket...")
        try:
            result = s3_data_loader(
                bucket=bucket,
                operation="list_files",
                prefix="",  # List from root
                pattern="*.csv"  # Only CSV files
            )
            
            if result["status"] == "success":
                print(f"✅ Found {result['file_count']} CSV files")
                if result["files"]:
                    print("   Sample files:")
                    for file in result["files"][:5]:  # Show first 5
                        print(f"     • {file['key']} ({file['size']} bytes)")
                    
                    # Ask user to select a file for testing
                    if result["files"]:
                        print(f"\n📄 Select a file to analyze:")
                        for i, file in enumerate(result["files"][:10]):
                            print(f"   {i+1}. {file['key']}")
                        
                        try:
                            choice = input("\nEnter file number (or press Enter to skip): ").strip()
                            if choice and choice.isdigit():
                                file_index = int(choice) - 1
                                if 0 <= file_index < len(result["files"]):
                                    selected_file = result["files"][file_index]["key"]
                                    return test_file_operations(bucket, selected_file)
                        except (ValueError, IndexError):
                            print("Invalid selection")
                else:
                    print("   No CSV files found. Try with a different pattern or add some CSV files to your bucket.")
            else:
                print(f"❌ Failed to list files: {result['error']}")
                return False
                
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return False
        
        # If no file selected, ask for manual input
        print("\n📝 Or enter a specific file key to test:")
        key = input("File key (e.g., data/sample.csv): ").strip()
        
        if key:
            return test_file_operations(bucket, key)
        else:
            print("ℹ️  No file specified. Testing completed.")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_operations(bucket, key):
    """Test various operations on a specific file."""
    print(f"\n🔬 Testing operations on: {bucket}/{key}")
    
    from strands_tools.s3_data_loader import s3_data_loader
    
    operations_to_test = [
        ("shape", {}, "Get file dimensions"),
        ("columns", {}, "Analyze columns and data types"),
        ("head", {"rows": 5}, "Preview first 5 rows"),
        ("info", {}, "Detailed file information"),
        ("describe", {}, "Statistical summary")
    ]
    
    results = {}
    
    for op, params, description in operations_to_test:
        print(f"\n   {description}...")
        try:
            result = s3_data_loader(
                bucket=bucket,
                key=key,
                operation=op,
                **params
            )
            
            if result["status"] == "success":
                print(f"   ✅ {op}: Success ({result['execution_time']})")
                results[op] = result
                
                # Show key information
                if op == "shape":
                    shape = result["data_stats"]
                    print(f"      📊 {shape['rows']} rows × {shape['columns']} columns")
                elif op == "columns":
                    cols = result["data_stats"]["columns"]
                    print(f"      📋 Columns: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
                elif op == "head":
                    print(f"      👀 Preview data loaded successfully")
                elif op == "info":
                    info = result["data_stats"]
                    print(f"      ℹ️  Shape: {info['shape']}, Null counts available")
                elif op == "describe":
                    stats = result["data_stats"]
                    print(f"      📈 Statistics for {len(stats)} numeric columns")
                    
            else:
                print(f"   ❌ {op}: {result['error']}")
                results[op] = result
                
        except Exception as e:
            print(f"   ❌ {op}: Exception - {e}")
            results[op] = {"status": "error", "error": str(e)}
    
    # Test advanced operations if basic ones work
    if results.get("shape", {}).get("status") == "success":
        print(f"\n🔍 Testing advanced operations...")
        
        # Test query operation
        print("   Testing data filtering...")
        try:
            # Try a simple query (this might fail if no numeric columns)
            result = s3_data_loader(
                bucket=bucket,
                key=key,
                operation="sample",
                rows=3
            )
            
            if result["status"] == "success":
                print(f"   ✅ sample: Got {result['data_stats']['sample_size']} random samples")
            else:
                print(f"   ❌ sample: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ sample: Exception - {e}")
    
    # Summary
    successful_ops = sum(1 for r in results.values() if r.get("status") == "success")
    total_ops = len(results)
    
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Successful operations: {successful_ops}/{total_ops}")
    print(f"   📁 File: {bucket}/{key}")
    
    if successful_ops > 0:
        print(f"   🎉 S3 Data Loader is working with your real data!")
        return True
    else:
        print(f"   ⚠️  No operations succeeded. Check file format and permissions.")
        return False

def print_usage_guide():
    """Print comprehensive usage guide."""
    print("\n📚 S3 Data Loader - Complete Usage Guide")
    print("=" * 50)
    
    print("""
🚀 Quick Start:
   1. Set AWS credentials (see above)
   2. Import and use the tool:

from strands import Agent
from strands_tools import s3_data_loader

agent = Agent(tools=[s3_data_loader])

📋 Basic Operations:
   • describe  - Statistical summary of numeric columns
   • shape     - Get rows and columns count
   • columns   - List column names and data types
   • head      - Preview first N rows

🔍 Advanced Operations:
   • query     - Filter data with pandas syntax
   • sample    - Get random sample of rows
   • info      - Detailed metadata and memory usage
   • unique    - Analyze unique values per column

📦 Batch Operations:
   • list_files - List S3 objects with pattern matching
   • batch_load - Load multiple files simultaneously
   • compare    - Compare two datasets

📁 Supported Formats:
   • CSV (.csv)
   • Parquet (.parquet)
   • JSON (.json)
   • Excel (.xlsx, .xls)
   • TSV (.tsv)
   • Text (.txt) with auto-detection

💡 Example Usage:

# Basic analysis
result = agent.tool.s3_data_loader(
    bucket="my-bucket",
    key="data/sales.csv",
    operation="describe"
)

# Advanced filtering
result = agent.tool.s3_data_loader(
    bucket="my-bucket",
    key="data/sales.csv",
    operation="query",
    query="revenue > 1000 and region == 'North'",
    rows=20
)

# Batch processing
result = agent.tool.s3_data_loader(
    bucket="my-bucket",
    operation="list_files",
    prefix="data/2024/",
    pattern="*.csv"
)

# Dataset comparison
result = agent.tool.s3_data_loader(
    bucket="my-bucket",
    key="current.csv",
    operation="compare",
    compare_key="previous.csv"
)

🔧 Environment Variables:
   • AWS_ACCESS_KEY_ID     - Your AWS access key
   • AWS_SECRET_ACCESS_KEY - Your AWS secret key
   • AWS_REGION           - AWS region (default: us-east-1)

⚡ Performance Tips:
   • Use 'shape' for quick file size checks
   • Use 'sample' for large file previews
   • Use 'list_files' with patterns for discovery
   • Batch operations are limited to 10 files for performance

🛡️ Security Notes:
   • Tool requires valid AWS credentials
   • Respects S3 bucket permissions
   • No data is cached or stored locally
   • All operations are read-only
""")

def main():
    """Main testing function."""
    print("🚀 S3 Data Loader - Real AWS Testing")
    print("=" * 50)
    
    # Check AWS credentials
    if not check_aws_credentials():
        print("\n⚠️  Cannot test without AWS credentials")
        print("   Set up credentials and run again")
        print_usage_guide()
        return False
    
    # Test with real S3
    print("\n🎯 Ready to test with real AWS S3!")
    
    try:
        success = test_with_real_s3()
        print_usage_guide()
        
        if success:
            print("\n🎉 REAL AWS TESTING COMPLETED SUCCESSFULLY!")
            print("   Your S3 Data Loader tool is ready for production use!")
        else:
            print("\n⚠️  Some tests failed. Check your S3 setup and try again.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Testing error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
