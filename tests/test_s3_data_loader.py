"""Unit tests for S3 data loader tool."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strands_tools.s3_data_loader import s3_data_loader


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"


@pytest.fixture
def sample_parquet_data():
    """Sample Parquet data for testing."""
    df = pd.DataFrame({"name": ["John", "Jane", "Bob"], "age": [25, 30, 35], "city": ["NYC", "LA", "Chicago"]})
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return '{"name": ["John", "Jane", "Bob"], "age": [25, 30, 35], "city": ["NYC", "LA", "Chicago"]}'


@pytest.fixture
def sample_tsv_data():
    """Sample TSV data for testing."""
    return "name\tage\tcity\nJohn\t25\tNYC\nJane\t30\tLA\nBob\t35\tChicago"


@pytest.fixture
def sample_excel_data():
    """Sample Excel data for testing."""
    df = pd.DataFrame({"name": ["John", "Jane", "Bob"], "age": [25, 30, 35], "city": ["NYC", "LA", "Chicago"]})
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_csv_describe(mock_boto3_client, sample_csv_data):
    """Test CSV file loading with describe operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock head_object response
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}

    # Mock get_object response
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="describe")

    # Assertions
    assert result["status"] == "success"
    assert result["file_info"]["bucket"] == "test-bucket"
    assert result["file_info"]["key"] == "test.csv"
    assert result["file_info"]["format"] == "csv"
    assert "data_stats" in result
    assert "execution_time" in result

    # Verify S3 calls
    mock_s3.head_object.assert_called_once_with(Bucket="test-bucket", Key="test.csv")
    mock_s3.get_object.assert_called_once_with(Bucket="test-bucket", Key="test.csv")


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_csv_shape(mock_boto3_client, sample_csv_data):
    """Test CSV file loading with shape operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="shape")

    # Assertions
    assert result["status"] == "success"
    assert result["data_stats"]["rows"] == 3
    assert result["data_stats"]["columns"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_csv_columns(mock_boto3_client, sample_csv_data):
    """Test CSV file loading with columns operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="columns")

    # Assertions
    assert result["status"] == "success"
    assert result["data_stats"]["columns"] == ["name", "age", "city"]
    assert "dtypes" in result["data_stats"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_csv_head(mock_boto3_client, sample_csv_data):
    """Test CSV file loading with head operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="head", rows=2)

    # Assertions
    assert result["status"] == "success"
    assert "data_stats" in result
    # Should have 2 rows as requested
    assert len(result["data_stats"]["name"]) == 2


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_parquet(mock_boto3_client, sample_parquet_data):
    """Test Parquet file loading."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_parquet_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_parquet_data
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.parquet", operation="shape")

    # Assertions
    assert result["status"] == "success"
    assert result["file_info"]["format"] == "parquet"
    assert result["data_stats"]["rows"] == 3
    assert result["data_stats"]["columns"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_unsupported_format(mock_boto3_client):
    """Test unsupported file format handling."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": 100}
    mock_body = MagicMock()
    mock_body.read.return_value = b"some data"
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function with truly unsupported format
    result = s3_data_loader(bucket="test-bucket", key="test.pdf", operation="shape")

    # Assertions
    assert result["status"] == "error"
    assert "Unsupported file format" in result["error"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_invalid_operation(mock_boto3_client, sample_csv_data):
    """Test invalid operation handling."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="invalid")

    # Assertions
    assert result["status"] == "error"
    assert "Unsupported operation" in result["error"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_s3_error(mock_boto3_client):
    """Test S3 access error handling."""
    # Mock S3 client to raise exception
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3
    mock_s3.head_object.side_effect = Exception("Access denied")

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv")

    # Assertions
    assert result["status"] == "error"
    assert "Access denied" in result["error"]


def test_s3_data_loader_custom_region():
    """Test custom region parameter."""
    with patch("strands_tools.s3_data_loader.boto3.client") as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        mock_s3.head_object.side_effect = Exception("Test exception")

        # Test with custom region
        s3_data_loader(bucket="test-bucket", key="test.csv", region="us-west-2")

        # Verify region was passed to boto3
        mock_boto3_client.assert_called_once_with("s3", region_name="us-west-2")


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_json(mock_boto3_client, sample_json_data):
    """Test JSON file loading."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_json_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_json_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.json", operation="shape")

    # Assertions
    assert result["status"] == "success"
    assert result["file_info"]["format"] == "json"
    assert result["data_stats"]["rows"] == 3
    assert result["data_stats"]["columns"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_tsv(mock_boto3_client, sample_tsv_data):
    """Test TSV file loading."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_tsv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_tsv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.tsv", operation="columns")

    # Assertions
    assert result["status"] == "success"
    assert result["file_info"]["format"] == "tsv"
    assert result["data_stats"]["columns"] == ["name", "age", "city"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_excel(mock_boto3_client, sample_excel_data):
    """Test Excel file loading."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_excel_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_excel_data
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.xlsx", operation="shape")

    # Assertions
    assert result["status"] == "success"
    assert result["file_info"]["format"] == "xlsx"
    assert result["data_stats"]["rows"] == 3
    assert result["data_stats"]["columns"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_txt_auto_detect(mock_boto3_client):
    """Test TXT file with auto-detection."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Test tab-separated data
    tsv_data = "name\tage\tcity\nJohn\t25\tNYC"
    mock_s3.head_object.return_value = {"ContentLength": len(tsv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = tsv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.txt", operation="shape")

    # Assertions
    assert result["status"] == "success"
    assert result["file_info"]["format"] == "txt"
    assert result["data_stats"]["rows"] == 1
    assert result["data_stats"]["columns"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_unsupported_format_enhanced(mock_boto3_client):
    """Test enhanced error message for unsupported formats."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": 100}
    mock_body = MagicMock()
    mock_body.read.return_value = b"some data"
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.pdf", operation="shape")

    # Assertions
    assert result["status"] == "error"
    assert "Unsupported file format: pdf" in result["error"]
    assert "csv, parquet, json, xlsx, xls, tsv, txt" in result["error"]


# Advanced Operations Tests (Phase 2B)


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_query_operation(mock_boto3_client, sample_csv_data):
    """Test query operation with pandas query syntax."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="query", query="age > 25")

    # Assertions
    assert result["status"] == "success"
    assert result["data_stats"]["query"] == "age > 25"
    assert result["data_stats"]["matched_rows"] == 2  # Jane and Bob
    assert result["data_stats"]["total_rows"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_query_no_query_param(mock_boto3_client, sample_csv_data):
    """Test query operation without query parameter."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="query")

    # Assertions
    assert result["status"] == "error"
    assert "Query parameter required" in result["error"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_sample_operation(mock_boto3_client, sample_csv_data):
    """Test sample operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="sample", rows=2)

    # Assertions
    assert result["status"] == "success"
    assert result["data_stats"]["sample_size"] == 2
    assert result["data_stats"]["total_rows"] == 3
    assert "results" in result["data_stats"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_info_operation(mock_boto3_client, sample_csv_data):
    """Test info operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="info")

    # Assertions
    assert result["status"] == "success"
    assert "shape" in result["data_stats"]
    assert "columns" in result["data_stats"]
    assert "dtypes" in result["data_stats"]
    assert "memory_usage" in result["data_stats"]
    assert "null_counts" in result["data_stats"]
    assert "non_null_counts" in result["data_stats"]
    assert result["data_stats"]["shape"]["rows"] == 3
    assert result["data_stats"]["shape"]["columns"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_unique_all_columns(mock_boto3_client, sample_csv_data):
    """Test unique operation for all columns."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="unique")

    # Assertions
    assert result["status"] == "success"
    assert "name" in result["data_stats"]
    assert "age" in result["data_stats"]
    assert "city" in result["data_stats"]
    assert result["data_stats"]["name"]["unique_count"] == 3
    assert result["data_stats"]["age"]["unique_count"] == 3
    assert result["data_stats"]["city"]["unique_count"] == 3


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_unique_specific_column(mock_boto3_client, sample_csv_data):
    """Test unique operation for specific column."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="unique", column="city")

    # Assertions
    assert result["status"] == "success"
    assert result["data_stats"]["column"] == "city"
    assert result["data_stats"]["unique_count"] == 3
    assert result["data_stats"]["total_count"] == 3
    assert "unique_values" in result["data_stats"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_unique_invalid_column(mock_boto3_client, sample_csv_data):
    """Test unique operation with invalid column."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="unique", column="invalid_column")

    # Assertions
    assert result["status"] == "error"
    assert "Column 'invalid_column' not found" in result["error"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_invalid_query(mock_boto3_client, sample_csv_data):
    """Test query operation with invalid query syntax."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock responses
    mock_s3.head_object.return_value = {"ContentLength": len(sample_csv_data)}
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="test.csv", operation="query", query="invalid_column > 25")

    # Assertions
    assert result["status"] == "error"
    assert "Invalid query" in result["error"]


# Batch Operations Tests (Phase 2C)


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_list_files(mock_boto3_client):
    """Test list_files operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock paginator
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator

    # Mock page iterator - S3 paginate already filters by prefix
    mock_page_iterator = [
        {
            "Contents": [
                {"Key": "data/file1.csv", "Size": 1024, "LastModified": pd.Timestamp("2023-01-01")},
                {"Key": "data/file2.json", "Size": 2048, "LastModified": pd.Timestamp("2023-01-02")},
            ]
        }
    ]
    mock_paginator.paginate.return_value = mock_page_iterator

    # Test the function
    result = s3_data_loader(bucket="test-bucket", operation="list_files", prefix="data/")

    # Assertions
    assert result["status"] == "success"
    assert result["operation"] == "list_files"
    assert result["file_count"] == 2  # Files with 'data/' prefix
    assert len(result["files"]) == 2
    assert result["files"][0]["key"] == "data/file1.csv"
    assert result["files"][0]["format"] == "csv"


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_list_files_with_pattern(mock_boto3_client):
    """Test list_files operation with pattern filtering."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock paginator
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator

    # Mock page iterator
    mock_page_iterator = [
        {
            "Contents": [
                {"Key": "data/file1.csv", "Size": 1024, "LastModified": pd.Timestamp("2023-01-01")},
                {"Key": "data/file2.json", "Size": 2048, "LastModified": pd.Timestamp("2023-01-02")},
                {"Key": "data/file3.csv", "Size": 1536, "LastModified": pd.Timestamp("2023-01-03")},
            ]
        }
    ]
    mock_paginator.paginate.return_value = mock_page_iterator

    # Test the function
    result = s3_data_loader(bucket="test-bucket", operation="list_files", pattern="*.csv")

    # Assertions
    assert result["status"] == "success"
    assert result["file_count"] == 2  # Only CSV files
    assert all(f["format"] == "csv" for f in result["files"])


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_batch_load(mock_boto3_client, sample_csv_data):
    """Test batch_load operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock get_object responses
    mock_body = MagicMock()
    mock_body.read.return_value = sample_csv_data.encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Test the function
    keys = ["file1.csv", "file2.csv"]
    result = s3_data_loader(bucket="test-bucket", operation="batch_load", keys=keys, rows=2)

    # Assertions
    assert result["status"] == "success"
    assert result["operation"] == "batch_load"
    assert result["files_processed"] == 2
    assert result["total_rows"] == 6  # 3 rows per file * 2 files
    assert result["max_columns"] == 3
    assert "file1.csv" in result["results"]
    assert "file2.csv" in result["results"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_batch_load_no_keys(mock_boto3_client):
    """Test batch_load operation without keys parameter."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Test the function
    result = s3_data_loader(bucket="test-bucket", operation="batch_load")

    # Assertions
    assert result["status"] == "error"
    assert "Keys parameter required" in result["error"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_compare(mock_boto3_client, sample_csv_data, sample_json_data):
    """Test compare operation."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock get_object responses
    def mock_get_object(Bucket, Key):
        mock_body = MagicMock()
        if Key == "file1.csv":
            mock_body.read.return_value = sample_csv_data.encode()
        else:  # file2.json
            mock_body.read.return_value = sample_json_data.encode()
        return {"Body": mock_body}

    mock_s3.get_object.side_effect = mock_get_object

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="file1.csv", operation="compare", compare_key="file2.json")

    # Assertions
    assert result["status"] == "success"
    assert result["operation"] == "compare"
    assert result["comparison"]["file1"]["key"] == "file1.csv"
    assert result["comparison"]["file2"]["key"] == "file2.json"
    assert result["comparison"]["comparison"]["same_shape"]
    assert result["comparison"]["comparison"]["same_columns"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_compare_no_compare_key(mock_boto3_client):
    """Test compare operation without compare_key parameter."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Test the function
    result = s3_data_loader(bucket="test-bucket", key="file1.csv", operation="compare")

    # Assertions
    assert result["status"] == "error"
    assert "compare_key parameter required" in result["error"]


@patch("strands_tools.s3_data_loader.boto3.client")
def test_s3_data_loader_single_file_no_key(mock_boto3_client):
    """Test single-file operation without key parameter."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Test the function
    result = s3_data_loader(bucket="test-bucket", operation="describe")

    # Assertions
    assert result["status"] == "error"
    assert "Key parameter is required for single-file operations" in result["error"]
