"""
Main Pipeline - Data Cleaning System Orchestrator
Coordinates all components for end-to-end data cleaning workflow.
"""

import polars as pl
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import os

# Import our modules
from src.data_profiler import DataProfiler
from src.rule_engine import RuleEngine
from src.data_cleaner import DataCleaner
from src.monitoring import DataQualityMonitor
from utils.logger import setup_logging
from utils.helpers import create_output_directories, save_applied_rules_json, load_data, generate_executive_summary

logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """
    Main orchestrator for the data cleaning pipeline.
    Manages the entire workflow from data loading to final reporting.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, rules_config: str = "config/rules.json"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            confidence_threshold: Minimum confidence to apply rules
            rules_config: Path to rules configuration file
        """
        self.confidence_threshold = confidence_threshold
        self.rules_config = rules_config
        self.execution_stats = {}
        
        # Initialize components
        self.profiler = DataProfiler()
        self.rule_engine = RuleEngine(rules_config)
        self.data_cleaner = DataCleaner()
        self.monitor = DataQualityMonitor()
        
        logger.info(f"Pipeline initialized with confidence threshold: {confidence_threshold}")
    
    def run_pipeline(self, input_file = "data/retail_store_sales.csv", output_dir = "/output"):
        """
        Execute the complete data cleaning pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory for output files
            
        Returns:
            Dictionary of generated file paths
        """
        start_time = time.time()
        
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("-" * 50)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        try:
            # Create output directories
            create_output_directories(output_dir)
            
            # Step 1: Load and validate input data
            logger.info("STEP 1: Loading and validating input data")
            original_df = load_data(input_file)
            self.execution_stats['data_loading'] = {
                'input_file': input_file,
                'original_shape': original_df.shape,
                'memory_usage_mb': original_df.estimated_size() / (1024 * 1024)
            }
            
            # Step 2: Profile the dataset
            logger.info("STEP 2: Profiling dataset characteristics")
            profile_start = time.time()
            profile_results = self.profiler.profile_dataset(original_df)
            
            # Generate profile reports
            profile_html_path = self.profiler.generate_profile_report(f"{output_dir}/data_profile.html")
            profile_json_path = self.profiler.save_profile_json(f"{output_dir}/data_profile.json")
            
            self.execution_stats['profiling'] = {
                'duration_seconds': time.time() - profile_start,
                'columns_analyzed': len(profile_results.get('columns', {})),
                'quality_issues_found': len(profile_results.get('quality', {}).get('consistency', {}).get('issues', []))
            }
            
            logger.info(f"Profile completed in {self.execution_stats['profiling']['duration_seconds']:.2f} seconds")
            logger.info(f"Analyzed {self.execution_stats['profiling']['columns_analyzed']} columns")
            
            # Step 3: Match rules to fields
            logger.info("STEP 3: Matching cleaning rules to fields")
            rule_start = time.time()
            
            field_analysis = profile_results.get('columns', {})
            matched_rules = self.rule_engine.analyze_and_match_rules(original_df, field_analysis)
            
            # Generate rule analysis reports
            rule_analysis_path = self.rule_engine.save_rule_analysis(f"{output_dir}/rule_analysis.json")
            rule_coverage = self.rule_engine.generate_rule_coverage_report()
            
            self.execution_stats['rule_matching'] = {
                'duration_seconds': time.time() - rule_start,
                'total_rules_matched': sum(len(rules) for rules in matched_rules.values()),
                'fields_with_rules': len([f for f, rules in matched_rules.items() if rules and f != '_dataset_rules']),
                'coverage_percentage': rule_coverage.get('coverage_statistics', {}).get('coverage_percentage', 0)
            }
            
            logger.info(f"Rule matching completed in {self.execution_stats['rule_matching']['duration_seconds']:.2f} seconds")
            logger.info(f"Matched {self.execution_stats['rule_matching']['total_rules_matched']} rules across {self.execution_stats['rule_matching']['fields_with_rules']} fields")
            logger.info(f"Field coverage: {self.execution_stats['rule_matching']['coverage_percentage']:.1f}%")
            
            # Log unmatched fields
            unmatched_summary = self.rule_engine.get_unmatched_fields_summary()
            if unmatched_summary['count'] > 0:
                logger.warning(f"{unmatched_summary['count']} fields did not match specific rules")
                for field in list(unmatched_summary['fields'].keys())[:3]:
                    logger.warning(f"Unmatched field: {field}")
            
            # Step 4: Apply cleaning rules
            logger.info("STEP 4: Applying cleaning transformations")
            cleaning_start = time.time()
            
            cleaned_df = self.data_cleaner.clean_dataset(
                original_df, 
                matched_rules, 
                self.confidence_threshold
            )
            
            # Save cleaned dataset
            cleaned_data_path = f"{output_dir}/cleaned_data.csv"
            cleaned_df.write_csv(cleaned_data_path)
            
            # Generate cleaning reports
            cleaning_summary = self.data_cleaner.get_cleaning_summary()
            cleaning_report_path = self.data_cleaner.save_cleaning_report(f"{output_dir}/cleaning_report.json")
            
            self.execution_stats['data_cleaning'] = {
                'duration_seconds': time.time() - cleaning_start,
                'operations_performed': cleaning_summary.get('operations_performed', 0),
                'records_affected': cleaning_summary.get('total_records_affected', 0),
                'errors_encountered': cleaning_summary.get('errors_encountered', 0),
                'final_shape': cleaned_df.shape
            }
            
            logger.info(f"Cleaning completed in {self.execution_stats['data_cleaning']['duration_seconds']:.2f} seconds")
            logger.info(f"Performed {self.execution_stats['data_cleaning']['operations_performed']} operations")
            logger.info(f"Affected {self.execution_stats['data_cleaning']['records_affected']} records")
            
            if self.execution_stats['data_cleaning']['errors_encountered'] > 0:
                logger.warning(f"{self.execution_stats['data_cleaning']['errors_encountered']} errors encountered")
            
            # Step 5: Monitor and generate metrics
            logger.info("STEP 5: Generating quality metrics and monitoring")
            monitoring_start = time.time()
            
            monitoring_report = self.monitor.monitor_cleaning_process(
                original_df, 
                cleaned_df, 
                cleaning_summary, 
                rule_coverage
            )
            
            # Generate monitoring outputs
            monitoring_report_path = self.monitor.save_monitoring_report(
                monitoring_report, f"{output_dir}/monitoring_report.json"
            )
            # monitoring_report_path = os.path.join(output_dir, "monitoring_report.json")
            # with open(monitoring_report_path, "w") as f:
            #         json.dump(monitoring_report, f, indent=4)
            
            dashboard_path = self.monitor.generate_html_dashboard(
                monitoring_report, f"{output_dir}/monitoring_dashboard.html"
            )
            
            self.execution_stats['monitoring'] = {
                'duration_seconds': time.time() - monitoring_start,
                'quality_score': monitoring_report.get('quality_scores', {}).get('overall', 0),
                'alerts_generated': len(monitoring_report.get('alerts', [])),
                'anomalies_detected': monitoring_report.get('anomalies_detected', {}).get('summary', {}).get('total_anomalies', 0)
            }
            
            logger.info(f"Monitoring completed in {self.execution_stats['monitoring']['duration_seconds']:.2f} seconds")
            logger.info(f"Overall quality score: {self.execution_stats['monitoring']['quality_score']:.1%}")
            logger.info(f"Generated {self.execution_stats['monitoring']['alerts_generated']} alerts")
            logger.info(f"Detected {self.execution_stats['monitoring']['anomalies_detected']} anomalies")
            
            # Step 6: Generate final reports and summaries
            logger.info("STEP 6: Generating final reports")
            
            # Save applied rules in required format
            applied_rules_path = save_applied_rules_json(
                self.data_cleaner.cleaning_log, f"{output_dir}/applied_rules.json"
            )
            
            # Generate executive summary
            summary_path = generate_executive_summary(
                original_df, cleaned_df, monitoring_report, self.execution_stats, 
                self.confidence_threshold, f"{output_dir}/executive_summary.md"
            )
            
            # Calculate total execution time
            total_time = time.time() - start_time
            self.execution_stats['total_execution_time'] = total_time
            
            # Final success logging
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("-" * 50)
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            logger.info(f"Data shape: {original_df.shape} → {cleaned_df.shape}")
            logger.info(f"Quality improvement: {monitoring_report.get('quality_scores', {}).get('overall', 0):.1%}")
            
            # Print alerts summary if any
            alerts = monitoring_report.get('alerts', [])
            if alerts:
                logger.warning("ALERTS GENERATED:")
                alerts_summary = self.monitor.generate_alerts_summary(alerts)
                logger.warning(alerts_summary)
            
            logger.info(f"Output files saved to: {output_dir}/")
            
            # Return paths to generated files
            return {
                'cleaned_data': cleaned_data_path,
                'applied_rules': applied_rules_path,
                'data_profile_html': profile_html_path,
                'data_profile_json': profile_json_path,
                'rule_analysis': rule_analysis_path,
                'cleaning_report': cleaning_report_path,
                'monitoring_report': monitoring_report_path,
                'monitoring_dashboard': dashboard_path,
                'executive_summary': summary_path
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
    


def main():
    """Main entry point with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Data Cleaning Pipeline - Automated data quality improvement system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv                           # Basic cleaning with default settings
  %(prog)s data.csv --confidence 0.8          # Higher confidence threshold
  %(prog)s data.csv --output results/         # Custom output directory
  %(prog)s data.csv --rules config/rules.json # Custom rules configuration
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file to clean"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for cleaned data and reports (default: output)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for applying rules (0.0-1.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--rules", "-r",
        default="config/rules.json",
        help="Path to rules configuration file (default: config/rules.json)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output (errors and warnings still shown)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    if not Path(args.rules).exists():
        print(f"Error: Rules configuration file '{args.rules}' not found")
        sys.exit(1)
    
    if not 0 <= args.confidence <= 1:
        print(f"Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Setup logging
    log_level = logging.WARNING if args.quiet else getattr(logging, args.log_level)
    setup_logging(log_level, f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    try:
        # Create and run pipeline
        pipeline = DataCleaningPipeline(
            confidence_threshold=args.confidence,
            rules_config=args.rules
        )
        
        generated_files = pipeline.run_pipeline(args.input_file, args.output)
        
        # Print success message
        if not args.quiet:
            print("\n" + "="*60)
            print("DATA CLEANING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"All files saved to: {args.output}/")
            print(f"Open monitoring_dashboard.html for detailed analysis")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()