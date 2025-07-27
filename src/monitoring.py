"""
Monitoring Module
Comprehensive monitoring, metrics collection, and dashboard generation.
"""

import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Comprehensive monitoring system for data quality metrics,
    anomaly detection, and dashboard generation.
    """
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.anomalies = {}
        self.quality_scores = {}
        self.comparison_stats = {}
        
    def monitor_cleaning_process(self, original_df: pl.DataFrame, cleaned_df: pl.DataFrame,
                                cleaning_summary: Dict[str, Any], 
                                rule_coverage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive monitoring of the entire cleaning process.
        
        Args:
            original_df: Original dataset before cleaning
            cleaned_df: Dataset after cleaning
            cleaning_summary: Summary from data cleaner
            rule_coverage: Rule coverage analysis
            
        Returns:
            Complete monitoring report
        """
        logger.info("Starting comprehensive monitoring analysis...")
        
        # Calculate data quality metrics
        self.metrics = self._calculate_quality_metrics(original_df, cleaned_df)
        
        # Detect anomalies and outliers
        self.anomalies = self._detect_anomalies(cleaned_df)
        
        # Generate quality scores
        self.quality_scores = self._calculate_quality_scores()
        
        # Create comparison statistics
        self.comparison_stats = self._generate_comparison_stats(original_df, cleaned_df)
        
        # Generate alerts based on thresholds
        self.alerts = self._generate_alerts()
        
        # Combine all monitoring results
        monitoring_report = {
            'quality_metrics': self.metrics,
            'anomalies_detected': self.anomalies,
            'quality_scores': self.quality_scores,
            'comparison_stats': self.comparison_stats,
            'alerts': self.alerts,
            'cleaning_summary': cleaning_summary,
            'rule_coverage': rule_coverage,
            'monitoring_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Monitoring completed. Generated {len(self.alerts)} alerts.")
        
        return monitoring_report
    
    def _calculate_quality_metrics(self, original_df: pl.DataFrame, 
                                  cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        
        metrics = {
            'completeness': self._calculate_completeness(original_df, cleaned_df),
            'consistency': self._calculate_consistency(original_df, cleaned_df),
            'validity': self._calculate_validity(cleaned_df),
            'uniqueness': self._calculate_uniqueness(original_df, cleaned_df),
            'accuracy': self._calculate_accuracy(original_df, cleaned_df),
            'timeliness': self._calculate_timeliness(cleaned_df)
        }
        
        return metrics
    
    def _calculate_completeness(self, original_df: pl.DataFrame, 
                               cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate data completeness metrics."""
        
        original_nulls = sum(original_df[col].null_count() for col in original_df.columns)
        cleaned_nulls = sum(cleaned_df[col].null_count() for col in cleaned_df.columns)
        
        original_total = original_df.shape[0] * original_df.shape[1]
        cleaned_total = cleaned_df.shape[0] * cleaned_df.shape[1]
        
        return {
            'original': {
                'total_cells': original_total,
                'null_cells': original_nulls,
                'completeness_ratio': 1 - (original_nulls / original_total) if original_total > 0 else 0
            },
            'cleaned': {
                'total_cells': cleaned_total,
                'null_cells': cleaned_nulls,
                'completeness_ratio': 1 - (cleaned_nulls / cleaned_total) if cleaned_total > 0 else 0
            },
            'improvement': {
                'nulls_fixed': original_nulls - cleaned_nulls,
                'improvement_ratio': ((cleaned_nulls / cleaned_total) - (original_nulls / original_total)) if original_total > 0 and cleaned_total > 0 else 0
            },
            'by_column': {
                col: {
                    'original_nulls': original_df[col].null_count() if col in original_df.columns else 0,
                    'cleaned_nulls': cleaned_df[col].null_count() if col in cleaned_df.columns else 0,
                    'null_percentage': (cleaned_df[col].null_count() / len(cleaned_df)) * 100 if col in cleaned_df.columns else 0
                }
                for col in set(original_df.columns) | set(cleaned_df.columns)
            }
        }
    
    def _calculate_consistency(self, original_df: pl.DataFrame, 
                              cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate data consistency metrics."""
        
        consistency_issues = []
        consistency_scores = []
        
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == pl.Utf8:
                # Check case consistency
                valid_values = cleaned_df[col].drop_nulls()
                if len(valid_values) > 0:
                    sample = valid_values.head(min(100, len(valid_values))).to_list()
                    
                    # Check case patterns
                    case_patterns = {
                        'title': sum(1 for s in sample if isinstance(s, str) and s.istitle()),
                        'upper': sum(1 for s in sample if isinstance(s, str) and s.isupper()),
                        'lower': sum(1 for s in sample if isinstance(s, str) and s.islower()),
                        'mixed': sum(1 for s in sample if isinstance(s, str) and 
                                   any(c.isupper() for c in s) and any(c.islower() for c in s) and not s.istitle())
                    }
                    
                    # Calculate consistency score
                    max_pattern = max(case_patterns.values())
                    consistency_score = max_pattern / len(sample) if len(sample) > 0 else 1
                    consistency_scores.append(consistency_score)
                    
                    if consistency_score < 0.8:
                        consistency_issues.append({
                            'column': col,
                            'issue': 'case_inconsistency',
                            'score': consistency_score,
                            'patterns': case_patterns
                        })
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return {
            'overall_score': overall_consistency,
            'issues_found': len(consistency_issues),
            'issues': consistency_issues,
            'columns_analyzed': len(consistency_scores)
        }
    
    def _calculate_validity(self, cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate data validity metrics."""
        
        validity_checks = {}
        
        for col in cleaned_df.columns:
            validity_info = {'column': col, 'checks_performed': []}
            
            # Check for obviously invalid values based on data type
            if cleaned_df[col].dtype in [pl.Int64, pl.Float64]:
                # Check for infinite or extremely large values
                valid_values = cleaned_df[col].drop_nulls()
                if len(valid_values) > 0:
                    infinite_count = 0  # Polars handles infinities differently
                    try:
                        # Check for unreasonably large values (>1e10 or <-1e10)
                        extreme_values = valid_values.filter(
                            (valid_values > 1e10) | (valid_values < -1e10)
                        )
                        extreme_count = len(extreme_values)
                        
                        validity_info['checks_performed'].append({
                            'check': 'extreme_values',
                            'invalid_count': extreme_count,
                            'validity_ratio': 1 - (extreme_count / len(valid_values))
                        })
                    except Exception:
                        pass
            
            elif cleaned_df[col].dtype == pl.Utf8:
                # Check for obviously invalid text patterns
                valid_values = cleaned_df[col].drop_nulls()
                if len(valid_values) > 0:
                    sample = valid_values.head(min(100, len(valid_values))).to_list()
                    
                    # Check for excessively long strings (>1000 chars)
                    long_strings = sum(1 for s in sample if isinstance(s, str) and len(s) > 1000)
                    
                    # Check for strings with unusual character patterns
                    unusual_patterns = sum(1 for s in sample if isinstance(s, str) and 
                                         (len(set(s)) == 1 or  # All same character
                                          sum(c.isdigit() for c in s) > len(s) * 0.9))  # Mostly digits
                    
                    validity_info['checks_performed'].extend([
                        {
                            'check': 'excessive_length',
                            'invalid_count': long_strings,
                            'validity_ratio': 1 - (long_strings / len(sample))
                        },
                        {
                            'check': 'unusual_patterns',
                            'invalid_count': unusual_patterns,
                            'validity_ratio': 1 - (unusual_patterns / len(sample))
                        }
                    ])
            
            validity_checks[col] = validity_info
        
        # Calculate overall validity score
        all_ratios = []
        for col_info in validity_checks.values():
            for check in col_info['checks_performed']:
                all_ratios.append(check['validity_ratio'])
        
        overall_validity = np.mean(all_ratios) if all_ratios else 1.0
        
        return {
            'overall_score': overall_validity,
            'by_column': validity_checks,
            'columns_with_issues': sum(1 for info in validity_checks.values() 
                                     if any(check['validity_ratio'] < 0.95 for check in info['checks_performed']))
        }
    
    def _calculate_uniqueness(self, original_df: pl.DataFrame, 
                             cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate data uniqueness metrics."""
        
        original_total = len(original_df)
        original_unique = len(original_df.unique())
        
        cleaned_total = len(cleaned_df)
        cleaned_unique = len(cleaned_df.unique())
        
        return {
            'original': {
                'total_rows': original_total,
                'unique_rows': original_unique,
                'duplicate_rows': original_total - original_unique,
                'uniqueness_ratio': original_unique / original_total if original_total > 0 else 0
            },
            'cleaned': {
                'total_rows': cleaned_total,
                'unique_rows': cleaned_unique,
                'duplicate_rows': cleaned_total - cleaned_unique,
                'uniqueness_ratio': cleaned_unique / cleaned_total if cleaned_total > 0 else 0
            },
            'improvement': {
                'duplicates_removed': (original_total - original_unique) - (cleaned_total - cleaned_unique),
                'rows_removed': original_total - cleaned_total
            }
        }
    
    def _calculate_accuracy(self, original_df: pl.DataFrame, 
                           cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate data accuracy metrics based on transformations."""
        
        # This is a simplified accuracy calculation
        # In a real scenario, you'd have ground truth data to compare against
        
        changes_made = 0
        total_cells_compared = 0
        
        for col in original_df.columns:
            if col in cleaned_df.columns:
                # Count cells that changed (simplified approach)
                try:
                    original_values = original_df[col]
                    cleaned_values = cleaned_df[col]
                    
                    # For comparable length dataframes
                    min_len = min(len(original_values), len(cleaned_values))
                    if min_len > 0:
                        # This is a simplified comparison - in reality, you'd need more sophisticated matching
                        total_cells_compared += min_len
                        # Assume some percentage of changes were improvements
                        # This would be calculated differently in a real system
                        
                except Exception:
                    continue
        
        # Placeholder accuracy calculation
        # In production, this would be based on validation against known correct values
        estimated_accuracy = 0.95  # Assume 95% accuracy for successfully processed fields
        
        return {
            'estimated_accuracy': estimated_accuracy,
            'cells_processed': total_cells_compared,
            'method': 'estimated_based_on_transformations',
            'note': 'Accuracy calculation would require ground truth data in production'
        }
    
    def _calculate_timeliness(self, cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate data timeliness metrics."""
        
        timeliness_info = {
            'processing_timestamp': datetime.now().isoformat(),
            'data_freshness': 'current',  # Would be calculated based on data timestamps
            'note': 'Timeliness assessment would require data source timestamps'
        }
        
        # Look for date columns to assess data freshness
        date_columns = [col for col in cleaned_df.columns 
                       if cleaned_df[col].dtype in [pl.Date, pl.Datetime]]
        
        if date_columns:
            date_analysis = {}
            for col in date_columns:
                valid_dates = cleaned_df[col].drop_nulls()
                if len(valid_dates) > 0:
                    try:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        date_analysis[col] = {
                            'min_date': str(min_date),
                            'max_date': str(max_date),
                            'date_range': str(max_date - min_date) if hasattr((max_date - min_date), 'days') else 'N/A'
                        }
                    except Exception:
                        date_analysis[col] = {'error': 'Could not analyze date range'}
            
            timeliness_info['date_analysis'] = date_analysis
        
        return timeliness_info
    
    def _detect_anomalies(self, cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Detect anomalies and outliers in the cleaned dataset."""
        
        anomalies = {
            'statistical_outliers': {},
            'pattern_anomalies': {},
            'value_anomalies': {},
            'summary': {'total_anomalies': 0, 'affected_columns': []}
        }
        
        for col in cleaned_df.columns:
            col_anomalies = []
            
            if cleaned_df[col].dtype in [pl.Int64, pl.Float64]:
                # Statistical outlier detection
                outliers = self._detect_statistical_outliers(cleaned_df[col], col)
                if outliers['count'] > 0:
                    anomalies['statistical_outliers'][col] = outliers
                    col_anomalies.extend(['statistical_outliers'])
            
            elif cleaned_df[col].dtype == pl.Utf8:
                # Pattern anomaly detection
                pattern_anomalies = self._detect_pattern_anomalies(cleaned_df[col], col)
                if pattern_anomalies:
                    anomalies['pattern_anomalies'][col] = pattern_anomalies
                    col_anomalies.extend(['pattern_anomalies'])
            
            # Value anomaly detection (applies to all types)
            value_anomalies = self._detect_value_anomalies(cleaned_df[col], col)
            if value_anomalies:
                anomalies['value_anomalies'][col] = value_anomalies
                col_anomalies.extend(['value_anomalies'])
            
            if col_anomalies:
                anomalies['summary']['affected_columns'].append({
                    'column': col,
                    'anomaly_types': col_anomalies
                })
        
        anomalies['summary']['total_anomalies'] = (
            len(anomalies['statistical_outliers']) + 
            len(anomalies['pattern_anomalies']) + 
            len(anomalies['value_anomalies'])
        )
        
        return anomalies
    
    def _detect_statistical_outliers(self, series: pl.Series, col_name: str) -> Dict[str, Any]:
        """
        Detect statistical outliers using multiple methods.
        
        Current: IQR method
        Alternatives: Z-score, Modified Z-score (MAD), Isolation Forest
        """
        valid_values = series.drop_nulls()
        if len(valid_values) < 4:
            return {'count': 0, 'method': 'insufficient_data'}
        
        try:
            # IQR Method - robust, doesn't assume normal distribution
            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = valid_values.filter(
                (valid_values < lower_bound) | (valid_values > upper_bound)
            )
            
            outlier_percentage = (len(outliers) / len(valid_values)) * 100
            
            return {
                'method': 'iqr',
                'count': len(outliers),
                'percentage': outlier_percentage,
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
                'sample_outliers': outliers.head(5).to_list(),
                'alternative_methods': [
                    'z_score (assumes normal distribution)',
                    'modified_z_score (uses MAD, more robust)',
                    'isolation_forest (for multivariate outliers)'
                ]
            }
            
        except Exception as e:
            return {'count': 0, 'error': str(e)}
    
    def _detect_pattern_anomalies(self, series: pl.Series, col_name: str) -> Optional[Dict[str, Any]]:
        """
        Detect pattern anomalies in text data.
        
        Current: Length and character pattern analysis
        Alternatives: Edit distance clustering, N-gram analysis, Regex pattern mining
        """
        valid_values = series.drop_nulls()
        if len(valid_values) == 0:
            return None
        
        try:
            sample = valid_values.head(min(200, len(valid_values))).to_list()
            
            # Length analysis
            lengths = [len(str(val)) for val in sample]
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            # Find length outliers
            length_outliers = [
                val for val in sample 
                if abs(len(str(val)) - mean_length) > 2 * std_length
            ]
            
            # Character pattern analysis
            digit_ratios = [
                sum(c.isdigit() for c in str(val)) / len(str(val)) if len(str(val)) > 0 else 0
                for val in sample
            ]
            mean_digit_ratio = np.mean(digit_ratios)
            
            # Find values with unusual digit patterns
            digit_outliers = [
                val for i, val in enumerate(sample)
                if abs(digit_ratios[i] - mean_digit_ratio) > 0.5
            ]
            
            anomalies_found = len(length_outliers) + len(digit_outliers)
            
            if anomalies_found > 0:
                return {
                    'length_anomalies': {
                        'count': len(length_outliers),
                        'samples': length_outliers[:3],
                        'mean_length': mean_length,
                        'std_length': std_length
                    },
                    'pattern_anomalies': {
                        'count': len(digit_outliers),
                        'samples': digit_outliers[:3],
                        'mean_digit_ratio': mean_digit_ratio
                    },
                    'total_anomalies': anomalies_found,
                    'alternative_methods': [
                        'edit_distance_clustering (group similar patterns)',
                        'ngram_analysis (detect common subpatterns)',
                        'regex_pattern_mining (discover format rules)'
                    ]
                }
            
            return None
            
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_value_anomalies(self, series: pl.Series, col_name: str) -> Optional[Dict[str, Any]]:
        """
        Detect general value anomalies.
        
        Current: Null clustering, unique value analysis
        Alternatives: Frequency-based anomaly detection, Domain-specific validation
        """
        try:
            total_count = len(series)
            null_count = series.null_count()
            unique_count = series.n_unique()
            
            anomalies = []
            
            # High null percentage
            null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
            if null_percentage > 50:
                anomalies.append({
                    'type': 'high_null_percentage',
                    'value': null_percentage,
                    'threshold': 50,
                    'severity': 'high'
                })
            
            # Very low uniqueness (potential data quality issue)
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            if uniqueness_ratio < 0.01 and series.dtype != pl.Boolean:  # Less than 1% unique (excluding booleans)
                anomalies.append({
                    'type': 'low_uniqueness',
                    'value': uniqueness_ratio,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'severity': 'medium'
                })
            
            # Single value dominance (one value appears in >95% of records)
            if total_count > 0:
                value_counts = series.value_counts()
                if len(value_counts) > 0:
                    max_count = value_counts[0, 1]  # Get count of most frequent value
                    dominance_ratio = max_count / total_count
                    
                    if dominance_ratio > 0.95:
                        anomalies.append({
                            'type': 'single_value_dominance',
                            'value': dominance_ratio,
                            'dominant_value': value_counts[0, 0],
                            'severity': 'medium'
                        })
            
            if anomalies:
                return {
                    'anomalies': anomalies,
                    'count': len(anomalies),
                    'alternative_methods': [
                        'frequency_based_detection (identify rare value patterns)',
                        'domain_specific_validation (business rule validation)'
                    ]
                }
            
            return None
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_quality_scores(self) -> Dict[str, Any]:
        """Calculate overall quality scores."""
        
        scores = {}
        
        # Completeness score
        completeness = self.metrics.get('completeness', {})
        scores['completeness'] = completeness.get('cleaned', {}).get('completeness_ratio', 0)
        
        # Consistency score
        consistency = self.metrics.get('consistency', {})
        scores['consistency'] = consistency.get('overall_score', 0)
        
        # Validity score
        validity = self.metrics.get('validity', {})
        scores['validity'] = validity.get('overall_score', 0)
        
        # Uniqueness score
        uniqueness = self.metrics.get('uniqueness', {})
        scores['uniqueness'] = uniqueness.get('cleaned', {}).get('uniqueness_ratio', 0)
        
        # Calculate weighted overall score
        weights = {'completeness': 0.3, 'consistency': 0.25, 'validity': 0.25, 'uniqueness': 0.2}
        overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
        
        scores['overall'] = overall_score
        scores['weights_used'] = weights
        
        # Score interpretation
        if overall_score >= 0.9:
            scores['interpretation'] = 'Excellent'
        elif overall_score >= 0.8:
            scores['interpretation'] = 'Good'
        elif overall_score >= 0.7:
            scores['interpretation'] = 'Fair'
        else:
            scores['interpretation'] = 'Poor'
        
        return scores
    
    def _generate_comparison_stats(self, original_df: pl.DataFrame, 
                                  cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Generate before/after comparison statistics."""
        
        return {
            'shape_comparison': {
                'original': original_df.shape,
                'cleaned': cleaned_df.shape,
                'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
                'columns_changed': original_df.shape[1] - cleaned_df.shape[1]
            },
            'data_type_changes': self._analyze_data_type_changes(original_df, cleaned_df),
            'memory_usage': {
                'original_mb': original_df.estimated_size() / (1024 * 1024),
                'cleaned_mb': cleaned_df.estimated_size() / (1024 * 1024),
                'reduction_mb': (original_df.estimated_size() - cleaned_df.estimated_size()) / (1024 * 1024)
            },
            'null_comparison': self._compare_null_values(original_df, cleaned_df)
        }
    
    def _analyze_data_type_changes(self, original_df: pl.DataFrame, 
                                  cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze data type changes between original and cleaned data."""
        
        changes = {}
        
        for col in original_df.columns:
            if col in cleaned_df.columns:
                original_type = str(original_df[col].dtype)
                cleaned_type = str(cleaned_df[col].dtype)
                
                if original_type != cleaned_type:
                    changes[col] = {
                        'from': original_type,
                        'to': cleaned_type,
                        'change_type': self._categorize_type_change(original_type, cleaned_type)
                    }
        
        return {
            'columns_with_changes': len(changes),
            'changes': changes
        }
    
    def _categorize_type_change(self, from_type: str, to_type: str) -> str:
        """Categorize the type of data type change."""
        if 'Utf8' in from_type and 'Float64' in to_type:
            return 'string_to_numeric'
        elif 'Utf8' in from_type and 'Boolean' in to_type:
            return 'string_to_boolean'
        elif 'Utf8' in from_type and 'Date' in to_type:
            return 'string_to_date'
        elif 'Float64' in from_type and 'Int64' in to_type:
            return 'float_to_integer'
        else:
            return 'other'
    
    def _compare_null_values(self, original_df: pl.DataFrame, 
                            cleaned_df: pl.DataFrame) -> Dict[str, Any]:
        """Compare null values between original and cleaned datasets."""
        
        comparison = {}
        
        for col in original_df.columns:
            if col in cleaned_df.columns:
                original_nulls = original_df[col].null_count()
                cleaned_nulls = cleaned_df[col].null_count()
                
                comparison[col] = {
                    'original_nulls': original_nulls,
                    'cleaned_nulls': cleaned_nulls,
                    'nulls_fixed': original_nulls - cleaned_nulls,
                    'improvement_ratio': (original_nulls - cleaned_nulls) / original_nulls if original_nulls > 0 else 0
                }
        
        return comparison
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on quality thresholds and anomalies."""
        
        alerts = []
        
        # Quality score alerts
        quality_scores = self.quality_scores
        
        if quality_scores.get('overall', 1) < 0.7:
            alerts.append({
                'type': 'quality_warning',
                'severity': 'high',
                'message': f"Overall data quality score is low: {quality_scores.get('overall', 0):.2%}",
                'timestamp': datetime.now().isoformat(),
                'metric': 'overall_quality',
                'value': quality_scores.get('overall', 0),
                'threshold': 0.7
            })
        
        # Completeness alerts
        completeness = self.metrics.get('completeness', {})
        cleaned_completeness = completeness.get('cleaned', {}).get('completeness_ratio', 1)
        
        if cleaned_completeness < 0.95:
            alerts.append({
                'type': 'completeness_warning',
                'severity': 'medium',
                'message': f"Data completeness below threshold: {cleaned_completeness:.2%}",
                'timestamp': datetime.now().isoformat(),
                'metric': 'completeness',
                'value': cleaned_completeness,
                'threshold': 0.95
            })
        
        # Anomaly alerts
        anomalies_summary = self.anomalies.get('summary', {})
        total_anomalies = anomalies_summary.get('total_anomalies', 0)
        
        if total_anomalies > 0:
            alerts.append({
                'type': 'anomaly_detection',
                'severity': 'medium',
                'message': f"Detected {total_anomalies} anomalies across {len(anomalies_summary.get('affected_columns', []))} columns",
                'timestamp': datetime.now().isoformat(),
                'metric': 'anomalies',
                'value': total_anomalies,
                'affected_columns': [col['column'] for col in anomalies_summary.get('affected_columns', [])]
            })
        
        # High null percentage alerts
        completeness_by_col = completeness.get('by_column', {})
        high_null_columns = [
            col for col, info in completeness_by_col.items()
            if info.get('null_percentage', 0) > 20
        ]
        
        if high_null_columns:
            alerts.append({
                'type': 'high_null_values',
                'severity': 'medium',
                'message': f"Columns with >20% missing values: {', '.join(high_null_columns[:5])}",
                'timestamp': datetime.now().isoformat(),
                'metric': 'missing_values',
                'affected_columns': high_null_columns
            })
        
        return alerts
    
    def generate_html_dashboard(self, monitoring_report: Dict[str, Any], 
                               output_path: str = "output/monitoring_dashboard.html") -> str:
        """Generate comprehensive HTML dashboard."""
        
        try:
            # Create visualizations
            figures = self._create_dashboard_visualizations(monitoring_report)
            
            # Generate HTML dashboard
            html_content = self._generate_dashboard_html(monitoring_report, figures)
            
            # Save dashboard
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Monitoring dashboard saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            raise
    
    def _create_dashboard_visualizations(self, monitoring_report: Dict[str, Any]) -> Dict[str, str]:
        """Create visualizations for the monitoring dashboard."""
        
        figures = {}
        
        try:
            # 1. Quality Scores Overview
            quality_scores = monitoring_report.get('quality_scores', {})
            metrics = ['completeness', 'consistency', 'validity', 'uniqueness']
            scores = [quality_scores.get(metric, 0) for metric in metrics]
            
            fig_quality = go.Figure(go.Bar(
                x=metrics,
                y=scores,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                text=[f'{score:.2%}' for score in scores],
                textposition='auto'
            ))
            fig_quality.update_layout(
                title='Data Quality Scores',
                yaxis_title='Score',
                yaxis=dict(range=[0, 1]),
                showlegend=False
            )
            figures['quality_scores'] = fig_quality.to_html(include_plotlyjs=False)
            
            # 2. Before/After Comparison
            comparison_stats = monitoring_report.get('comparison_stats', {})
            shape_comparison = comparison_stats.get('shape_comparison', {})
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Original',
                x=['Rows', 'Columns'],
                y=[shape_comparison.get('original', [0, 0])[0], shape_comparison.get('original', [0, 0])[1]],
                marker_color='#ff7f0e'
            ))
            fig_comparison.add_trace(go.Bar(
                name='Cleaned',
                x=['Rows', 'Columns'],
                y=[shape_comparison.get('cleaned', [0, 0])[0], shape_comparison.get('cleaned', [0, 0])[1]],
                marker_color='#2ca02c'
            ))
            fig_comparison.update_layout(
                title='Dataset Size Comparison',
                yaxis_title='Count',
                barmode='group'
            )
            figures['comparison'] = fig_comparison.to_html(include_plotlyjs=False)
            
            # 3. Anomalies Summary
            anomalies = monitoring_report.get('anomalies_detected', {})
            anomaly_types = ['statistical_outliers', 'pattern_anomalies', 'value_anomalies']
            anomaly_counts = [len(anomalies.get(atype, {})) for atype in anomaly_types]
            
            if sum(anomaly_counts) > 0:
                fig_anomalies = px.pie(
                    values=anomaly_counts,
                    names=[atype.replace('_', ' ').title() for atype in anomaly_types],
                    title='Anomalies by Type'
                )
                figures['anomalies'] = fig_anomalies.to_html(include_plotlyjs=False)
            
            # 4. Null Values Improvement
            null_comparison = comparison_stats.get('null_comparison', {})
            columns_with_nulls = [col for col, info in null_comparison.items() 
                                if info.get('original_nulls', 0) > 0]
            
            if columns_with_nulls:
                original_nulls = [null_comparison[col]['original_nulls'] for col in columns_with_nulls]
                cleaned_nulls = [null_comparison[col]['cleaned_nulls'] for col in columns_with_nulls]
                
                fig_nulls = go.Figure()
                fig_nulls.add_trace(go.Bar(
                    name='Original',
                    x=columns_with_nulls,
                    y=original_nulls,
                    marker_color='#ff7f0e'
                ))
                fig_nulls.add_trace(go.Bar(
                    name='Cleaned',
                    x=columns_with_nulls,
                    y=cleaned_nulls,
                    marker_color='#2ca02c'
                ))
                fig_nulls.update_layout(
                    title='Null Values: Before vs After',
                    yaxis_title='Null Count',
                    xaxis_tickangle=-45,
                    barmode='group'
                )
                figures['null_comparison'] = fig_nulls.to_html(include_plotlyjs=False)
        
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return figures
    
    def _generate_dashboard_html(self, monitoring_report: Dict[str, Any], 
                                figures: Dict[str, str]) -> str:
        """Generate HTML dashboard content."""
        
        quality_scores = monitoring_report.get('quality_scores', {})
        alerts = monitoring_report.get('alerts', [])
        anomalies_summary = monitoring_report.get('anomalies_detected', {}).get('summary', {})
        comparison_stats = monitoring_report.get('comparison_stats', {})
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-card {{ text-align: center; }}
                .metric-value {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ color: #666; font-size: 0.9em; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid; }}
                .alert-high {{ background: #fff3cd; border-color: #ff6b6b; }}
                .alert-medium {{ background: #d1ecf1; border-color: #feca57; }}
                .alert-low {{ background: #d4edda; border-color: #48cab2; }}
                .chart-container {{ margin: 20px 0; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .stat-item {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .quality-excellent {{ color: #28a745; }}
                .quality-good {{ color: #17a2b8; }}
                .quality-fair {{ color: #ffc107; }}
                .quality-poor {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Monitoring Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Overall Quality Score: <span class="quality-{quality_scores.get('interpretation', 'poor').lower()}">{quality_scores.get('overall', 0):.1%}</span> ({quality_scores.get('interpretation', 'Unknown')})</p>
            </div>
            
            <div class="container">
                <!-- Quality Metrics Overview -->
                <div class="section">
                    <h2>Quality Metrics Overview</h2>
                    <div class="dashboard-grid">
                        <div class="card metric-card">
                            <div class="metric-value quality-{self._get_quality_color(quality_scores.get('completeness', 0))}">{quality_scores.get('completeness', 0):.1%}</div>
                            <div class="metric-label">Completeness</div>
                        </div>
                        <div class="card metric-card">
                            <div class="metric-value quality-{self._get_quality_color(quality_scores.get('consistency', 0))}">{quality_scores.get('consistency', 0):.1%}</div>
                            <div class="metric-label">Consistency</div>
                        </div>
                        <div class="card metric-card">
                            <div class="metric-value quality-{self._get_quality_color(quality_scores.get('validity', 0))}">{quality_scores.get('validity', 0):.1%}</div>
                            <div class="metric-label">Validity</div>
                        </div>
                        <div class="card metric-card">
                            <div class="metric-value quality-{self._get_quality_color(quality_scores.get('uniqueness', 0))}">{quality_scores.get('uniqueness', 0):.1%}</div>
                            <div class="metric-label">Uniqueness</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="chart-container">
                            {figures.get('quality_scores', '<p>Quality scores chart not available</p>')}
                        </div>
                    </div>
                </div>
                
                <!-- Alerts Section -->
                <div class="section">
                    <h2>Alerts & Warnings</h2>
                    <div class="card">
        """
        
        if alerts:
            for alert in alerts:
                severity_class = f"alert-{alert.get('severity', 'low')}"
                html_template += f"""
                        <div class="alert {severity_class}">
                            <strong>{alert.get('type', 'Alert').replace('_', ' ').title()}:</strong> {alert.get('message', 'No message')}
                        </div>
                """
        else:
            html_template += '<div class="alert alert-low"><strong>No Alerts:</strong> All quality metrics are within acceptable thresholds.</div>'
        
        html_template += """
                    </div>
                </div>
                
                <!-- Before/After Comparison -->
                <div class="section">
                    <h2>Data Transformation Summary</h2>
                    <div class="card">
                        <div class="stats-grid">
        """
        
        shape_comparison = comparison_stats.get('shape_comparison', {})
        memory_usage = comparison_stats.get('memory_usage', {})
        
        html_template += f"""
                            <div class="stat-item">
                                <div class="metric-value">{shape_comparison.get('rows_removed', 0):,}</div>
                                <div class="metric-label">Rows Removed</div>
                            </div>
                            <div class="stat-item">
                                <div class="metric-value">{memory_usage.get('reduction_mb', 0):.1f} MB</div>
                                <div class="metric-label">Memory Saved</div>
                            </div>
                            <div class="stat-item">
                                <div class="metric-value">{anomalies_summary.get('total_anomalies', 0)}</div>
                                <div class="metric-label">Anomalies Detected</div>
                            </div>
                            <div class="stat-item">
                                <div class="metric-value">{len(anomalies_summary.get('affected_columns', []))}</div>
                                <div class="metric-label">Columns with Anomalies</div>
                            </div>
        """
        
        html_template += f"""
                        </div>
                        
                        <div class="chart-container">
                            {figures.get('comparison', '<p>Comparison chart not available</p>')}
                        </div>
                    </div>
                </div>
                
                <!-- Anomalies Section -->
                <div class="section">
                    <h2>Anomaly Analysis</h2>
                    <div class="card">
                        {figures.get('anomalies', '<p>No anomalies detected</p>')}
                    </div>
                </div>
                
                <!-- Null Values Improvement -->
                <div class="section">
                    <h2>Missing Values Analysis</h2>
                    <div class="card">
                        {figures.get('null_comparison', '<p>No missing values found</p>')}
                    </div>
                </div>
                
                <!-- Detailed Statistics -->
                <div class="section">
                    <h2>Detailed Statistics</h2>
                    <div class="card">
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Original</th>
                                    <th>Cleaned</th>
                                    <th>Improvement</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Total Rows</td>
                                    <td>{shape_comparison.get('original', [0, 0])[0]:,}</td>
                                    <td>{shape_comparison.get('cleaned', [0, 0])[0]:,}</td>
                                    <td>{shape_comparison.get('rows_removed', 0):,} removed</td>
                                </tr>
                                <tr>
                                    <td>Total Columns</td>
                                    <td>{shape_comparison.get('original', [0, 0])[1]:,}</td>
                                    <td>{shape_comparison.get('cleaned', [0, 0])[1]:,}</td>
                                    <td>{shape_comparison.get('columns_changed', 0)} changed</td>
                                </tr>
                                <tr>
                                    <td>Memory Usage</td>
                                    <td>{memory_usage.get('original_mb', 0):.1f} MB</td>
                                    <td>{memory_usage.get('cleaned_mb', 0):.1f} MB</td>
                                    <td>{memory_usage.get('reduction_mb', 0):.1f} MB saved</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _get_quality_color(self, score: float) -> str:
        """Get color class based on quality score."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'fair'
        else:
            return 'poor'
    
    def save_monitoring_report(self, monitoring_report: Dict[str, Any], 
                              output_path: str = "output/monitoring_report.json") -> str:
        """Save complete monitoring report to JSON."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(monitoring_report, f, indent=2, default=str)
            
            logger.info(f"Monitoring report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving monitoring report: {e}")
            raise
    
    def generate_alerts_summary(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate a text summary of alerts for logging."""
        if not alerts:
            return "No alerts generated - all metrics within acceptable thresholds."
        
        summary_lines = [f"Generated {len(alerts)} alerts:"]
        
        # Group by severity
        by_severity = {}
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(alert)
        
        for severity, severity_alerts in by_severity.items():
            summary_lines.append(f"\n{severity.upper()} ({len(severity_alerts)}):")
            for alert in severity_alerts:
                summary_lines.append(f"  - {alert.get('message', 'No message')}")
        
        return "\n".join(summary_lines)