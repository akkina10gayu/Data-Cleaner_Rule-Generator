"""
Centralized logging configuration for the data cleaning pipeline.
Provides consistent logging across all modules with appropriate formatting and handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog


def setup_logging(log_level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Set up centralized logging with console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set the root logger level
    root_logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG+ in files
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return root_logger


class PerformanceLogger:
    """Context manager for timing and logging performance metrics."""
    
    def __init__(self, operation_name: str, logger: logging.Logger = None):
        """
        Initialize performance logger.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Logger instance to use
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        import time
        end_time = time.time()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"✓ {self.operation_name} completed in {duration:.2f} seconds")
        else:
            self.logger.error(f"✗ {self.operation_name} failed after {duration:.2f} seconds: {exc_val}")


class DataCleaningLogger:
    """Specialized logger for data cleaning operations with structured logging."""
    
    def __init__(self, name: str = "data_cleaning"):
        """Initialize the data cleaning logger."""
        self.logger = logging.getLogger(name)
        self.operations_log = []
        
    def log_rule_application(self, rule_id: str, field: str, result: str, 
                           confidence: float, records_affected: int = 0, **kwargs):
        """
        Log rule application with structured data.
        
        Args:
            rule_id: ID of the applied rule
            field: Field the rule was applied to
            result: Description of the result
            confidence: Confidence score of the rule
            records_affected: Number of records affected
            **kwargs: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'rule_id': rule_id,
            'field': field,
            'result': result,
            'confidence': confidence,
            'records_affected': records_affected,
            'type': 'rule_application',
            **kwargs
        }
        
        self.operations_log.append(log_entry)
        
        # Log to standard logger
        self.logger.info(
            f"Applied rule '{rule_id}' to field '{field}': {result} "
            f"(confidence: {confidence:.2f}, affected: {records_affected})"
        )
    
    def log_data_quality_metric(self, metric_name: str, value: float, 
                               threshold: float = None, status: str = None):
        """
        Log data quality metrics.
        
        Args:
            metric_name: Name of the quality metric
            value: Metric value
            threshold: Optional threshold for comparison
            status: Optional status (good, warning, critical)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'threshold': threshold,
            'status': status,
            'type': 'quality_metric'
        }
        
        self.operations_log.append(log_entry)
        
        # Log with appropriate level
        if status == 'critical':
            self.logger.error(f"Quality metric '{metric_name}': {value:.2%} (CRITICAL)")
        elif status == 'warning':
            self.logger.warning(f"Quality metric '{metric_name}': {value:.2%} (WARNING)")
        else:
            self.logger.info(f"Quality metric '{metric_name}': {value:.2%}")
    
    def log_anomaly_detection(self, field: str, anomaly_type: str, count: int, 
                             details: dict = None):
        """
        Log anomaly detection results.
        
        Args:
            field: Field where anomalies were detected
            anomaly_type: Type of anomaly (outlier, pattern, etc.)
            count: Number of anomalies detected
            details: Additional details about the anomalies
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'field': field,
            'anomaly_type': anomaly_type,
            'count': count,
            'details': details or {},
            'type': 'anomaly_detection'
        }
        
        self.operations_log.append(log_entry)
        
        if count > 0:
            self.logger.warning(
                f"Detected {count} {anomaly_type} anomalies in field '{field}'"
            )
        else:
            self.logger.info(f"No {anomaly_type} anomalies detected in field '{field}'")
    
    def log_error(self, operation: str, error_message: str, context: dict = None):
        """
        Log errors with context.
        
        Args:
            operation: Operation that failed
            error_message: Error description
            context: Additional context information
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error_message': error_message,
            'context': context or {},
            'type': 'error'
        }
        
        self.operations_log.append(log_entry)
        self.logger.error(f"Error in {operation}: {error_message}")
    
    def get_structured_log(self) -> list:
        """Get all structured log entries."""
        return self.operations_log.copy()
    
    def save_structured_log(self, output_path: str):
        """Save structured log to JSON file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                import json
                json.dump({
                    'generated_at': datetime.now().isoformat(),
                    'total_entries': len(self.operations_log),
                    'log_entries': self.operations_log
                }, f, indent=2)
                
            self.logger.info(f"Structured log saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save structured log: {e}")


# Convenience function for getting a configured logger
def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the module name.
    
    Args:
        name: Logger name (defaults to caller's module name)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    setup_logging(logging.DEBUG, "test_log.log")
    
    logger = get_logger("test_module")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logger
    with PerformanceLogger("test_operation", logger):
        import time
        time.sleep(1)  # Simulate work
    
    # Test data cleaning logger
    cleaning_logger = DataCleaningLogger("test_cleaning")
    cleaning_logger.log_rule_application(
        "normalize_phone", "customer_phone", 
        "Normalized 150 phone numbers", 0.95, 150
    )
    cleaning_logger.log_data_quality_metric("completeness", 0.98, 0.95, "good")
    cleaning_logger.log_anomaly_detection("price", "outlier", 5)
    
    print("Logging test completed")