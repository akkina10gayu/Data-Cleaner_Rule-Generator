"""
Utility functions for the data cleaning pipeline.
Common helper functions used across multiple modules.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def create_output_directories(base_dir: str) -> None:
    """
    Create necessary output directories.
    
    Args:
        base_dir: Base output directory path
    """
    directories = [
        base_dir,
        f"{base_dir}/logs",
        f"{base_dir}/reports",
        f"{base_dir}/visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directories under {base_dir}")


def save_applied_rules_json(cleaning_log: List[Dict], output_path: str) -> str:
    """
    Save applied rules in the required JSON format for the assessment.
    
    Args:
        cleaning_log: List of cleaning operations from DataCleaner
        output_path: Path to save the JSON file
        
    Returns:
        Path to saved file
    """
    try:
        # Transform cleaning log into required format
        rules_output = []
        seen_rules = set()
        
        for log_entry in cleaning_log:
            rule_id = log_entry.get('rule_id')
            field = log_entry.get('field', 'unknown')
            
            # Create unique key to avoid duplicates
            rule_key = f"{rule_id}_{field}"
            
            if rule_key not in seen_rules:
                rule_output = {
                    "rule_id": rule_id,
                    "description": log_entry.get('result', ''),
                    "confidence": log_entry.get('confidence', 1.0),
                    "applied_to": field,
                    "records_affected": log_entry.get('records_affected', 0)
                }
                
                # Add specific patterns for certain rule types
                if rule_id == 'normalize_phone':
                    rule_output.update({
                        "pattern": "(\\d{3})[. -]?(\\d{3})[. -]?(\\d{4})",
                        "replacement": "+1-\\1-\\2-\\3"
                    })
                elif 'outlier' in rule_id:
                    outlier_info = log_entry.get('outlier_info', {})
                    rule_output['method'] = outlier_info.get('method', 'unknown')
                elif 'validation' in rule_id:
                    validation_issues = log_entry.get('validation_issues', [])
                    if validation_issues:
                        rule_output['issues_found'] = validation_issues
                
                rules_output.append(rule_output)
                seen_rules.add(rule_key)
        
        # Create final output structure
        output = {
            "generated_at": datetime.now().isoformat(),
            "total_rules_applied": len(rules_output),
            "confidence_threshold_used": 0.7,  # This would come from pipeline config
            "rules": rules_output
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(rules_output)} applied rules to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving applied rules JSON: {e}")
        raise


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def calculate_improvement_percentage(original: float, improved: float) -> float:
    """
    Calculate improvement percentage between two values.
    
    Args:
        original: Original value
        improved: Improved value
        
    Returns:
        Improvement percentage
    """
    if original == 0:
        return 0.0
    return ((improved - original) / original) * 100


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def load_json_config(config_path: str, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load JSON configuration file with fallback to default.
    
    Args:
        config_path: Path to configuration file
        default_config: Default configuration if file not found
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
        if default_config:
            logger.info("Using default configuration")
            return default_config
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
        raise


def create_backup_file(file_path: str) -> str:
    """
    Create a backup of an existing file.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        Path to backup file
    """
    if not Path(file_path).exists():
        return ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    
    import shutil
    shutil.copy2(file_path, backup_path)
    
    logger.info(f"Created backup: {backup_path}")
    return backup_path


def validate_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Validate CSV file and return basic information.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Validation results
    """
    try:
        import polars as pl
        
        # Try to read the file
        df = pl.read_csv(file_path, n_rows=10)  # Just read first 10 rows for validation
        
        return {
            "valid": True,
            "columns": df.columns,
            "estimated_rows": None,  # Would need to count for large files
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024),
            "sample_data": df.head(3).to_dict()
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024) if Path(file_path).exists() else 0
        }


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries with nested key support.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dictionaries(result[key], value)
            else:
                result[key] = value
    
    return result


def generate_file_hash(file_path: str) -> str:
    """
    Generate SHA-256 hash of a file for integrity checking.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hex digest of file hash
    """
    import hashlib
    
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def extract_sample_data(data_list: List[Any], sample_size: int = 5) -> List[Any]:
    """
    Extract a representative sample from a list.
    
    Args:
        data_list: List to sample from
        sample_size: Number of samples to extract
        
    Returns:
        Sample list
    """
    if len(data_list) <= sample_size:
        return data_list
    
    # Try to get diverse samples
    step = len(data_list) // sample_size
    samples = []
    
    for i in range(0, len(data_list), step):
        if len(samples) < sample_size:
            samples.append(data_list[i])
    
    return samples


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def chunk_list(data_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        data_list: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(data_list), chunk_size):
        chunks.append(data_list[i:i + chunk_size])
    return chunks


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Memory usage statistics
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}