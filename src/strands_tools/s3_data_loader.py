"""
S3 Data Loader tool for Strands Agents.

Simplest entry point—loads data files directly from S3 buckets into memory for agents to analyze.
Uses boto3 to fetch Parquet/CSV objects, converts to pandas DataFrame, and returns stats like
shape, columns, or describe(). Ideal for data workflows where agents need quick access to
cloud datasets without manual downloads.
"""

import os
import re
import time
from io import BytesIO
from typing import Any, Dict

import boto3
import pandas as pd


def s3_data_loader(
    bucket: str,
    key: str = None,
    operation: str = "describe",
    rows: int = 5,
    region: str = None,
    query: str = None,
    column: str = None,
    prefix: str = None,
    pattern: str = None,
    keys: list = None,
    compare_key: str = None,
) -> Dict[str, Any]:
    """
    Load data files from S3 buckets and return analysis statistics.

    Args:
        bucket: S3 bucket name
        key: S3 object key (file path) - optional for list_files operation
        operation: Analysis operation - 'describe', 'shape', 'columns', 'head', 'query', 'sample', 'info', 'unique',
                  'list_files', 'batch_load', 'compare'
        rows: Number of rows for 'head' and 'sample' operations
        region: AWS region (defaults to AWS_REGION env var or us-east-1)
        query: Pandas query string for 'query' operation (e.g., "age > 25")
        column: Column name for 'unique' operation
        prefix: S3 prefix for 'list_files' operation
        pattern: File pattern for 'list_files' operation (e.g., "*.csv")
        keys: List of S3 keys for 'batch_load' operation
        compare_key: Second file key for 'compare' operation

    Returns:
        Dictionary with file info, data statistics, and execution time
    """
    start_time = time.time()

    # Setup AWS client
    region = region or os.getenv("AWS_REGION", "us-east-1")
    s3_client = boto3.client("s3", region_name=region)

    try:
        # Handle batch operations first
        if operation == "list_files":
            return _list_s3_files(s3_client, bucket, start_time, prefix, pattern)
        elif operation == "batch_load":
            return _batch_load_files(s3_client, bucket, start_time, keys, rows)
        elif operation == "compare":
            return _compare_files(s3_client, bucket, key, compare_key, start_time)

        # Validate key parameter for single-file operations
        if not key:
            raise ValueError("Key parameter is required for single-file operations")

        # Get object metadata
        response = s3_client.head_object(Bucket=bucket, Key=key)
        file_size = response["ContentLength"]

        # Download object
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()

        # Determine file format and load into DataFrame
        file_ext = key.lower().split(".")[-1]

        if file_ext == "csv":
            df = pd.read_csv(BytesIO(data))
        elif file_ext == "parquet":
            df = pd.read_parquet(BytesIO(data))
        elif file_ext == "json":
            df = pd.read_json(BytesIO(data))
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(BytesIO(data))
        elif file_ext == "tsv":
            df = pd.read_csv(BytesIO(data), sep="\t")
        elif file_ext == "txt":
            # Try to auto-detect delimiter for text files
            sample_data = data[:1024].decode("utf-8", errors="ignore")
            if "\t" in sample_data:
                df = pd.read_csv(BytesIO(data), sep="\t")
            else:
                df = pd.read_csv(BytesIO(data))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: csv, parquet, json, xlsx, xls, tsv, txt")

        # Perform requested operation
        if operation == "describe":
            data_stats = df.describe().to_dict()
        elif operation == "shape":
            data_stats = {"rows": df.shape[0], "columns": df.shape[1]}
        elif operation == "columns":
            data_stats = {"columns": list(df.columns), "dtypes": df.dtypes.to_dict()}
        elif operation == "head":
            data_stats = df.head(rows).to_dict()
        elif operation == "query":
            if not query:
                raise ValueError("Query parameter required for 'query' operation")
            try:
                filtered_df = df.query(query)
                data_stats = {
                    "query": query,
                    "matched_rows": len(filtered_df),
                    "total_rows": len(df),
                    "results": filtered_df.head(rows).to_dict() if len(filtered_df) > 0 else {},
                }
            except Exception as e:
                raise ValueError(f"Invalid query '{query}': {str(e)}") from e
        elif operation == "sample":
            sample_size = min(rows, len(df))
            sampled_df = df.sample(n=sample_size) if len(df) > 0 else df
            data_stats = {"sample_size": sample_size, "total_rows": len(df), "results": sampled_df.to_dict()}
        elif operation == "info":
            data_stats = {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "non_null_counts": df.count().to_dict(),
            }
        elif operation == "unique":
            if not column:
                # Return unique counts for all columns
                data_stats = {}
                for col in df.columns:
                    try:
                        unique_vals = df[col].value_counts().head(10).to_dict()
                        data_stats[col] = {
                            "unique_count": df[col].nunique(),
                            "total_count": len(df[col]),
                            "top_values": unique_vals,
                        }
                    except Exception:
                        # Handle columns that can't be counted (e.g., complex objects)
                        data_stats[col] = {"unique_count": "N/A", "error": "Cannot count unique values"}
            else:
                # Return unique values for specific column
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
                unique_vals = df[column].value_counts().head(20).to_dict()
                data_stats = {
                    "column": column,
                    "unique_count": df[column].nunique(),
                    "total_count": len(df[column]),
                    "unique_values": unique_vals,
                }
        else:
            raise ValueError(
                f"Unsupported operation: {operation}. "
                f"Supported: describe, shape, columns, head, query, sample, info, unique, "
                f"list_files, batch_load, compare"
            )

        execution_time = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "file_info": {"bucket": bucket, "key": key, "size_bytes": file_size, "format": key.split(".")[-1].lower()},
            "data_stats": data_stats,
            "execution_time": f"{execution_time}s",
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "file_info": {"bucket": bucket, "key": key},
            "execution_time": f"{round(time.time() - start_time, 2)}s",
        }


def _list_s3_files(
    s3_client, bucket: str, start_time: float, prefix: str = None, pattern: str = None
) -> Dict[str, Any]:
    """List S3 files with optional prefix and pattern filtering."""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix or "")

        files = []
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Apply pattern filtering if specified
                    if pattern:
                        # Convert shell pattern to regex
                        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
                        if not re.search(regex_pattern, key):
                            continue

                    files.append(
                        {
                            "key": key,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                            "format": key.split(".")[-1].lower() if "." in key else "unknown",
                        }
                    )

        execution_time = round(time.time() - start_time, 2)
        return {
            "status": "success",
            "operation": "list_files",
            "bucket": bucket,
            "prefix": prefix,
            "pattern": pattern,
            "file_count": len(files),
            "files": files[:100],  # Limit to first 100 files
            "truncated": len(files) > 100,
            "execution_time": f"{execution_time}s",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "operation": "list_files",
            "execution_time": f"{round(time.time() - start_time, 2)}s",
        }


def _load_single_file(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """Load a single file from S3 into DataFrame."""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    file_ext = key.lower().split(".")[-1]

    if file_ext == "csv":
        return pd.read_csv(BytesIO(data))
    elif file_ext == "parquet":
        return pd.read_parquet(BytesIO(data))
    elif file_ext == "json":
        return pd.read_json(BytesIO(data))
    elif file_ext in ["xlsx", "xls"]:
        return pd.read_excel(BytesIO(data))
    elif file_ext == "tsv":
        return pd.read_csv(BytesIO(data), sep="\t")
    elif file_ext == "txt":
        sample_data = data[:1024].decode("utf-8", errors="ignore")
        if "\t" in sample_data:
            return pd.read_csv(BytesIO(data), sep="\t")
        else:
            return pd.read_csv(BytesIO(data))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def _batch_load_files(s3_client, bucket: str, start_time: float, keys: list = None, rows: int = 5) -> Dict[str, Any]:
    """Load multiple files from S3 and return combined statistics."""
    if not keys:
        raise ValueError("Keys parameter required for batch_load operation")

    try:
        results = {}
        total_rows = 0
        total_columns = 0

        for key in keys[:10]:  # Limit to 10 files for performance
            try:
                df = _load_single_file(s3_client, bucket, key)
                results[key] = {
                    "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                    "columns": list(df.columns),
                    "sample": df.head(min(rows, 3)).to_dict() if len(df) > 0 else {},
                }
                total_rows += df.shape[0]
                total_columns = max(total_columns, df.shape[1])
            except Exception as e:
                results[key] = {"error": str(e)}

        execution_time = round(time.time() - start_time, 2)
        return {
            "status": "success",
            "operation": "batch_load",
            "bucket": bucket,
            "files_processed": len(results),
            "total_rows": total_rows,
            "max_columns": total_columns,
            "results": results,
            "execution_time": f"{execution_time}s",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "operation": "batch_load",
            "execution_time": f"{round(time.time() - start_time, 2)}s",
        }


def _compare_files(s3_client, bucket: str, key1: str, key2: str, start_time: float) -> Dict[str, Any]:
    """Compare two files from S3."""
    if not key2:
        raise ValueError("compare_key parameter required for compare operation")

    try:
        df1 = _load_single_file(s3_client, bucket, key1)
        df2 = _load_single_file(s3_client, bucket, key2)

        comparison = {
            "file1": {
                "key": key1,
                "shape": {"rows": df1.shape[0], "columns": df1.shape[1]},
                "columns": list(df1.columns),
            },
            "file2": {
                "key": key2,
                "shape": {"rows": df2.shape[0], "columns": df2.shape[1]},
                "columns": list(df2.columns),
            },
            "comparison": {
                "same_shape": df1.shape == df2.shape,
                "same_columns": list(df1.columns) == list(df2.columns),
                "common_columns": list(set(df1.columns) & set(df2.columns)),
                "unique_to_file1": list(set(df1.columns) - set(df2.columns)),
                "unique_to_file2": list(set(df2.columns) - set(df1.columns)),
            },
        }

        execution_time = round(time.time() - start_time, 2)
        return {
            "status": "success",
            "operation": "compare",
            "bucket": bucket,
            "comparison": comparison,
            "execution_time": f"{execution_time}s",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "operation": "compare",
            "execution_time": f"{round(time.time() - start_time, 2)}s",
        }


# Export the tool
__all__ = ["s3_data_loader"]
