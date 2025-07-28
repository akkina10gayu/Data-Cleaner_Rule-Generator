"""
Utility functions for the data cleaning pipeline.
Common helper functions used across multiple modules.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import polars as pl

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


def load_data(input_file: str) -> pl.DataFrame:
    """
    Load and validate input data with comprehensive error handling.
    
    Args:
        input_file: Path to input CSV file
        
    Returns:
        Loaded DataFrame
    """
    try:
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Try to load with Polars
        df = pl.read_csv(input_file, ignore_errors=True)
        
        if df.is_empty():
            raise ValueError("Input file is empty or could not be read")
        
        logger.info(f"Successfully loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
        logger.info(f"Memory usage: {df.estimated_size() / (1024 * 1024):.2f} MB")
        logger.debug(f"Dataset structure: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from {input_file}: {e}")
        raise


def generate_executive_summary(original_df: pl.DataFrame, cleaned_df: pl.DataFrame,
                              monitoring_report: dict, execution_stats: dict, 
                              confidence_threshold: float, output_path: str) -> str:
    """Generate executive summary report."""
    
    quality_scores = monitoring_report.get('quality_scores', {})
    alerts = monitoring_report.get('alerts', [])
    comparison_stats = monitoring_report.get('comparison_stats', {})
    
    summary_content = f"""# Data Cleaning Executive Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pipeline Confidence Threshold: {confidence_threshold}

## Key Results

### Data Quality Score: {quality_scores.get('overall', 0):.1%} ({quality_scores.get('interpretation', 'Unknown')})

### Transformation Summary
- **Original Dataset**: {original_df.shape[0]:,} rows × {original_df.shape[1]} columns
- **Cleaned Dataset**: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns
- **Rows Removed**: {original_df.shape[0] - cleaned_df.shape[0]:,}
- **Processing Time**: {execution_stats.get('total_execution_time', 0):.2f} seconds

## Quality Metrics

| Metric | Score | Status |
|--------|--------|--------|
| Completeness | {quality_scores.get('completeness', 0):.1%} | {'Excellent' if quality_scores.get('completeness', 0) > 0.9 else 'Good' if quality_scores.get('completeness', 0) > 0.8 else 'Needs Attention'} |
| Consistency | {quality_scores.get('consistency', 0):.1%} | {'Excellent' if quality_scores.get('consistency', 0) > 0.9 else 'Good' if quality_scores.get('consistency', 0) > 0.8 else 'Needs Attention'} |
| Validity | {quality_scores.get('validity', 0):.1%} | {'Excellent' if quality_scores.get('validity', 0) > 0.9 else 'Good' if quality_scores.get('validity', 0) > 0.8 else 'Needs Attention'} |
| Uniqueness | {quality_scores.get('uniqueness', 0):.1%} | {'Excellent' if quality_scores.get('uniqueness', 0) > 0.9 else 'Good' if quality_scores.get('uniqueness', 0) > 0.8 else 'Needs Attention'} |

## Operations Performed

- **Total Operations**: {execution_stats.get('data_cleaning', {}).get('operations_performed', 0)}
- **Records Affected**: {execution_stats.get('data_cleaning', {}).get('records_affected', 0):,}
- **Rule Coverage**: {execution_stats.get('rule_matching', {}).get('coverage_percentage', 0):.1f}%

## Alerts & Issues

"""
    
    if alerts:
        summary_content += f"**{len(alerts)} alerts generated:**\n\n"
        for alert in alerts:
            severity_level = {'high': 'HIGH', 'medium': 'MEDIUM', 'low': 'LOW'}.get(alert.get('severity', 'low'), 'UNKNOWN')
            summary_content += f"- **{severity_level}**: {alert.get('type', 'Alert').replace('_', ' ').title()}: {alert.get('message', 'No message')}\n"
    else:
        summary_content += "**No critical alerts** - All metrics within acceptable thresholds.\n"
    
    anomalies_count = monitoring_report.get('anomalies_detected', {}).get('summary', {}).get('total_anomalies', 0)
    if anomalies_count > 0:
        summary_content += f"\n**{anomalies_count} anomalies detected** across multiple columns. See detailed monitoring report for analysis.\n"
    
    summary_content += f"""

## Memory & Performance

- **Memory Usage**: {comparison_stats.get('memory_usage', {}).get('original_mb', 0):.1f} MB → {comparison_stats.get('memory_usage', {}).get('cleaned_mb', 0):.1f} MB
- **Memory Change**: {comparison_stats.get('memory_usage', {}).get('reduction_mb', 0):+.1f} MB
- **Processing Speed**: {original_df.shape[0] / execution_stats.get('total_execution_time', 1):.0f} rows/second

## Generated Files

1. **cleaned_data.csv** - The cleaned dataset ready for analysis
2. **applied_rules.json** - Detailed list of all cleaning rules applied
3. **monitoring_dashboard.html** - Interactive quality metrics dashboard
4. **data_profile.html** - Comprehensive data analysis report
5. **monitoring_report.json** - Detailed quality metrics and anomalies

## Next Steps

"""
    
    # Generate recommendations based on results
    if quality_scores.get('overall', 0) >= 0.9:
        summary_content += "**Excellent data quality achieved!** Dataset is ready for analysis and production use.\n"
    elif quality_scores.get('overall', 0) >= 0.8:
        summary_content += "**Good data quality achieved.** Minor improvements may be beneficial:\n"
    else:
        summary_content += "**Data quality needs attention.** Recommended actions:\n"
    
    if quality_scores.get('completeness', 0) < 0.9:
        summary_content += "- Review missing value handling strategies\n"
    if quality_scores.get('consistency', 0) < 0.9:
        summary_content += "- Address remaining format inconsistencies\n"
    if anomalies_count > 0:
        summary_content += "- Investigate detected anomalies for potential data issues\n"
    if alerts:
        summary_content += "- Address flagged quality alerts\n"
    
    summary_content += "\nFor detailed analysis, open the monitoring dashboard: `monitoring_dashboard.html`\n"
    
    # Save summary
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    logger.info(f"Executive summary saved to {output_path}")
    return output_path