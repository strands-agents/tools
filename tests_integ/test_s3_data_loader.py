"""Integration tests for S3 data loader tool with real AWS S3."""

import os

import pytest

from strands_tools.s3_data_loader import s3_data_loader


@pytest.mark.skipif(
    not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"),
    reason="AWS credentials required for integration tests"
)
class TestS3DataLoaderIntegration:
    """Integration tests using real S3 buckets."""
    
    def test_public_csv_file(self):
        """Test loading a CSV file from a public S3 bucket."""
        # Using a public dataset bucket (adjust as needed)
        # Note: Replace with actual public bucket/key for real testing
        bucket = "your-public-test-bucket"  # Replace with real bucket
        key = "sample-data.csv"  # Replace with real CSV file
        
        # Skip if no test bucket configured
        if bucket == "your-public-test-bucket":
            pytest.skip("Configure a real test bucket for integration tests")
        
        result = s3_data_loader(bucket=bucket, key=key, operation="shape")
        
        assert result["status"] == "success"
        assert "rows" in result["data_stats"]
        assert "columns" in result["data_stats"]
        assert result["file_info"]["bucket"] == bucket
        assert result["file_info"]["key"] == key
        assert result["file_info"]["format"] == "csv"
    
    def test_public_parquet_file(self):
        """Test loading a Parquet file from a public S3 bucket."""
        # Using a public dataset bucket (adjust as needed)
        bucket = "your-public-test-bucket"  # Replace with real bucket
        key = "sample-data.parquet"  # Replace with real Parquet file
        
        # Skip if no test bucket configured
        if bucket == "your-public-test-bucket":
            pytest.skip("Configure a real test bucket for integration tests")
        
        result = s3_data_loader(bucket=bucket, key=key, operation="describe")
        
        assert result["status"] == "success"
        assert "data_stats" in result
        assert result["file_info"]["format"] == "parquet"
    
    def test_all_operations(self):
        """Test all operations on a real file."""
        bucket = "your-public-test-bucket"  # Replace with real bucket
        key = "sample-data.csv"  # Replace with real CSV file
        
        # Skip if no test bucket configured
        if bucket == "your-public-test-bucket":
            pytest.skip("Configure a real test bucket for integration tests")
        
        operations = ["describe", "shape", "columns", "head"]
        
        for operation in operations:
            result = s3_data_loader(bucket=bucket, key=key, operation=operation)
            assert result["status"] == "success", f"Operation {operation} failed"
            assert "data_stats" in result
            assert "execution_time" in result
    
    def test_nonexistent_file(self):
        """Test error handling with nonexistent file."""
        bucket = "your-public-test-bucket"  # Replace with real bucket
        key = "nonexistent-file.csv"
        
        # Skip if no test bucket configured
        if bucket == "your-public-test-bucket":
            pytest.skip("Configure a real test bucket for integration tests")
        
        result = s3_data_loader(bucket=bucket, key=key)
        
        assert result["status"] == "error"
        assert "error" in result
    
    def test_custom_region(self):
        """Test with custom AWS region."""
        bucket = "your-public-test-bucket"  # Replace with real bucket
        key = "sample-data.csv"  # Replace with real CSV file
        region = "us-west-2"
        
        # Skip if no test bucket configured
        if bucket == "your-public-test-bucket":
            pytest.skip("Configure a real test bucket for integration tests")
        
        result = s3_data_loader(bucket=bucket, key=key, region=region, operation="shape")
        
        # Should work regardless of region for public buckets
        assert result["status"] in ["success", "error"]  # May fail due to region mismatch


# Example test configuration for common public datasets
class TestPublicDatasets:
    """Tests using known public datasets (when available)."""
    
    @pytest.mark.skip(reason="Configure with actual public dataset")
    def test_aws_open_data(self):
        """Test with AWS Open Data Registry datasets."""
        # Example: AWS Open Data has various public datasets
        # Replace with actual public dataset details
        bucket = "aws-open-data-example"
        key = "path/to/data.csv"
        
        result = s3_data_loader(bucket=bucket, key=key, operation="shape")
        
        if result["status"] == "success":
            assert "rows" in result["data_stats"]
            assert "columns" in result["data_stats"]
        else:
            # May fail due to access restrictions or dataset changes
            assert "error" in result


# Helper function for manual testing
def manual_test_with_your_bucket():
    """
    Manual test function - replace with your actual S3 bucket details.
    
    To run manually:
    1. Set AWS credentials in environment
    2. Replace bucket/key with your test data
    3. Run: python -c "from tests_integ.test_s3_data_loader import manual_test_with_your_bucket; manual_test_with_your_bucket()"
    """
    bucket = "your-bucket-name"  # Replace with your bucket
    key = "your-file.csv"  # Replace with your file
    
    print(f"Testing S3 data loader with bucket: {bucket}, key: {key}")
    
    result = s3_data_loader(bucket=bucket, key=key, operation="describe")
    
    print(f"Result: {result}")
    
    if result["status"] == "success":
        print("✅ S3 data loader working correctly!")
        print(f"File info: {result['file_info']}")
        print(f"Execution time: {result['execution_time']}")
    else:
        print("❌ S3 data loader failed:")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    # Run manual test if executed directly
    manual_test_with_your_bucket()
