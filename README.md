# Data Cleaning Pipeline - Automated Data Quality System

A comprehensive, production-ready data cleaning system that automatically learns and applies cleaning rules with intelligent field detection, comprehensive monitoring, and detailed reporting.

## Key Features

- **Intelligent Field Detection**: Automatically identifies field types (phone, email, currency, etc.) based on column names and data patterns
- **Rule-Based Cleaning**: Comprehensive set of predefined rules for common data quality issues
- **Confidence Scoring**: Each rule has confidence scores with configurable thresholds
- **Comprehensive Monitoring**: Real-time quality metrics, anomaly detection, and interactive dashboards
- **Production Ready**: Proper logging, error handling, and scalability considerations
- **Extensible Architecture**: Easy to add new rules and customize for specific datasets

## Requirements

- **Python 3.8+**
- **Memory**: 2GB RAM minimum (4GB+ recommended for large datasets)
- **Storage**: 100MB for dependencies, additional space for output files
- **OS**: Windows, macOS, or Linux

## Project Structure

```
Data-Cleaner_Rule-Generator/
├── main.py                    # Main orchestrator script
├── config/
│   └── rules.json            # Rule definitions and configuration
├── src/
│   ├── data_profiler.py      # Dataset analysis and profiling
│   ├── rule_engine.py        # Rule matching and selection logic
│   ├── data_cleaner.py       # Cleaning implementation
│   └── monitoring.py         # Quality metrics and dashboard
├── utils/
│   ├── logger.py             # Centralized logging
│   ├── helpers.py            # Utility functions
│   └── test_rules.py         # Unit test cases for the rules
├── output/                   # Generated files directory
├── logs/                     # Log files
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your data:**
   - Place your CSV file in the project directory or note its path
   - The system works with any CSV file

### Basic Usage

```bash
# Basic cleaning with default settings
python main.py your_dataset.csv

# Custom confidence threshold
python main.py your_dataset.csv --confidence 0.8

# Custom output directory
python main.py your_dataset.csv --output results/

# Full customization
python main.py your_dataset.csv --confidence 0.8 --output results/ --log-level DEBUG
```

### Command Line Options

- `input_file`: Path to input CSV file (required)
- `--output, -o`: Output directory (default: output)
- `--confidence, -c`: Confidence threshold 0.0-1.0 (default: 0.7)
- `--rules, -r`: Path to rules config file (default: config/rules.json)
- `--log-level, -l`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--quiet, -q`: Suppress progress output

### Expected Results

**Processing Time**: 30 seconds to 2 minutes for typical datasets (10K-100K rows)
**Success Indicators**: 
- Overall quality score of 85%+ 
- Interactive dashboard opens correctly
- No critical errors in logs

## Output Files

After running the pipeline, you'll find these files in your output directory:

### Essential Files
1. **`cleaned_data.csv`** - Your cleaned dataset ready for analysis
2. **`monitoring_dashboard.html`** - Interactive quality metrics dashboard (open this first!)
3. **`executive_summary.md`** - High-level summary for stakeholders

### Detailed Reports  
4. **`data_profile.html`** - Comprehensive data analysis report
5. **`applied_rules.json`** - Detailed list of all cleaning rules applied
6. **`monitoring_report.json`** - Detailed quality metrics and anomalies
7. **`rule_analysis.json`** - Rule matching and coverage analysis
8. **`cleaning_report.json`** - Detailed cleaning operation log

## Rule System

### Rule Categories

**Universal Rules** (Applied to All Data):
- Duplicate removal, whitespace trimming, missing value handling

**Field-Specific Rules** (Applied Based on Detection):
- Phone/email normalization, currency formatting, date parsing, address standardization, outlier detection

**Dataset-Specific Rules** (Business Logic):
- Transaction validation, range validation, business rule enforcement

### Confidence Scoring

Rules are applied based on confidence thresholds (default: 0.7):

- **Perfect Confidence (1.0)**: Core data operations (duplicate removal, whitespace trimming)
- **High Confidence (0.9-0.99)**: Reliable transformations (phone normalization, currency formatting, validation)
- **Medium Confidence (0.8-0.89)**: Moderate reliability (address standardization, missing value handling)
- **Low Confidence (<0.8)**: Skipped unless threshold is lowered (aggressive transformations)

**Threshold Control:**
```bash
# Conservative cleaning (only high-confidence rules)
python main.py data.csv --confidence 0.9

# Aggressive cleaning (include lower-confidence rules)  
python main.py data.csv --confidence 0.6
```

## Monitoring & Quality Metrics

### Quality Dimensions Tracked
1. **Completeness**: Percentage of non-null values
2. **Consistency**: Format uniformity across similar fields
3. **Validity**: Adherence to expected formats and patterns
4. **Uniqueness**: Duplicate detection and removal
5. **Accuracy**: Improvement through transformations

### Anomaly Detection
- **Statistical Outliers**: IQR and Z-score methods
- **Pattern Anomalies**: Unusual text patterns and formats
- **Value Anomalies**: High null rates, low uniqueness, single value dominance

### Alert System
Automatically generates alerts for:
- Quality scores below thresholds
- High numbers of anomalies detected
- Columns with excessive missing values
- Processing errors or failures

## Configuration

### Customizing Rules

Edit `config/rules.json` to:
- Enable/disable specific rules
- Adjust confidence scores
- Modify field detection patterns
- Add custom transformations

Example rule structure:
```json
{
  "rule_id": "normalize_phone",
  "rule_type": "normalization",
  "field_patterns": ["phone", "contact", "mobile"],
  "data_types": ["string", "object"],
  "description": "Normalize phone numbers to standard format",
  "confidence": 0.94,
  "enabled": true,
  "priority": 5
}
```

### Field Detection

The system automatically detects field types based on:
- **Column names**: Matches patterns like 'phone', 'email', 'price'
- **Data patterns**: Analyzes actual values for phone/email/date patterns
- **Data types**: Considers numeric vs text vs boolean types

### Unmatched Fields

Fields that don't match specific rules are:
- Flagged for manual review
- Analyzed for characteristics (length, patterns, value types)
- Given recommendations for potential custom rules

## Production Deployment

### Architecture Considerations

**Scalability**:
- Uses Polars for efficient large dataset processing
- Modular design allows horizontal scaling
- Configurable sampling for rule learning on large datasets

**Monitoring**:
- Structured logging for centralized log management
- Quality metrics export to CSV for time-series analysis
- Alert system integration with monitoring tools

**Error Handling**:
- Comprehensive exception handling with graceful degradation
- Failed operations logged without stopping the pipeline
- Data integrity preservation through backup mechanisms

**Versioning**:
- Rule configuration version control
- Data lineage tracking
- Rollback capabilities for cleaning operations

### Deployment Patterns

1. **Batch Processing**: Schedule regular runs via cron/Airflow
2. **API Service**: Wrap in FastAPI for on-demand cleaning
3. **Streaming**: Adapt for real-time data stream processing
4. **Microservice**: Deploy as containerized service

## Design Principles

### About the Approach

1. **Intelligence Over Hardcoding**: Rules are matched dynamically based on field characteristics rather than hardcoded field names
2. **Confidence-Based Processing**: Different confidence levels allow for careful automated processing while flagging uncertain cases
3. **Comprehensive Monitoring**: Production systems need observability - metrics, alerts, and dashboards provide this
4. **Extensible Architecture**: Adding new rules or field types is straightforward without modifying core logic
5. **Production Readiness**: Proper logging, error handling, and documentation make this suitable for real-world use

### Edge Cases Handled

- **Mixed Data Types**: Handles columns with mixed content types
- **Missing Rule Matches**: Gracefully handles fields that don't match any specific rules
- **Processing Failures**: Individual rule failures don't stop the entire pipeline
- **Large Datasets**: Memory-efficient processing with configurable sampling
- **Invalid Data**: Comprehensive validation and error reporting

```

## Testing

### Running Unit Tests

```bash
# Run all unit tests
python -m pytest utils/test_rules.py -v

# Run specific test categories
python -m pytest utils/test_rules.py::TestUniversalRules -v
python -m pytest utils/test_rules.py::TestFieldSpecificRules -v
python -m pytest utils/test_rules.py::TestDatasetSpecificRules -v
```

### Test Coverage

The test suite covers:
- Universal rules (duplicate removal, whitespace trimming, missing value handling)
- Field-specific rules (phone normalization, currency formatting, boolean standardization)
- Dataset-specific rules (transaction validation, discount validation)
- Edge cases and error conditions

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**"Permission denied" errors:**
- Ensure write permissions for output directory
- Check file locks on input CSV files

**Low quality scores:**
- Review confidence threshold (try lowering with `--confidence 0.6`)
- Check unmatched fields in rule analysis report
- Verify data format compatibility

**Memory issues with large datasets:**
- Use sampling for rule learning: modify confidence thresholds
- Process data in chunks (feature planned for future releases)
- Increase system memory or use cloud processing

## Contributing

This system is designed to be extensible. To add new cleaning rules:

1. Define the rule in `config/rules.json`
2. Add field detection logic in `rule_engine.py`
3. Implement cleaning logic in `data_cleaner.py`
4. Add tests for your new functionality

## Technical Details

### Dependencies
- **Polars**: High-performance DataFrame operations
- **Plotly**: Interactive visualizations and dashboards
- **Phonenumbers**: International phone number processing
- **Colorlog**: Enhanced logging with colors
- **Jinja2**: HTML template rendering

### Performance
- **Memory Efficient**: Uses Polars lazy evaluation
- **Fast Processing**: Optimized for datasets up to 100M+ rows
- **Incremental**: Can process data in chunks for very large datasets

### Alternatives Considered
- **Outlier Detection**: IQR (current), Z-score, Isolation Forest, DBSCAN
- **Fuzzy Matching**: RapidFuzz for category standardization (future enhancement)
- **ML-Based**: Embedding-based similarity for advanced deduplication
