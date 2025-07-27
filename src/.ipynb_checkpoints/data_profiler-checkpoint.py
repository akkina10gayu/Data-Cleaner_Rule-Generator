"""
Data Profiler Module
Comprehensive analysis of dataset structure, quality, and patterns.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Comprehensive data profiler that analyzes dataset characteristics,
    quality issues, and generates detailed reports.
    """
    
    def __init__(self):
        self.profile_results = {}
        self.data_quality_issues = {}
        self.field_analysis = {}
        
    def profile_dataset(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Main profiling function that generates comprehensive dataset analysis.
        
        Args:
            df: Polars DataFrame to profile
            
        Returns:
            Dict containing complete profiling results
        """
        logger.info(f"Starting dataset profiling for shape: {df.shape}")
        
        # Basic dataset overview
        self.profile_results['overview'] = self._analyze_overview(df)
        
        # Column-wise detailed analysis
        self.profile_results['columns'] = self._analyze_columns(df)
        
        # Data quality assessment
        self.profile_results['quality'] = self._assess_data_quality(df)
        
        # Statistical summary
        self.profile_results['statistics'] = self._generate_statistics(df)
        
        # Pattern analysis
        self.profile_results['patterns'] = self._analyze_patterns(df)
        
        # Correlation analysis for numeric columns
        self.profile_results['correlations'] = self._analyze_correlations(df)
        
        logger.info("Dataset profiling completed successfully")
        return self.profile_results
    
    def _analyze_overview(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset overview."""
        return {
            'shape': df.shape,
            'columns': df.columns,
            'memory_usage_mb': df.estimated_size() / (1024 * 1024),
            'total_cells': df.shape[0] * df.shape[1],
            'data_types': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        }
    
    def _analyze_columns(self, df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detailed analysis for each column."""
        column_analysis = {}
        
        for col in df.columns:
            series = df[col]
            analysis = {
                'dtype': str(series.dtype),
                'null_count': series.null_count(),
                'null_percentage': (series.null_count() / len(series)) * 100,
                'unique_count': series.n_unique(),
                'unique_percentage': (series.n_unique() / len(series)) * 100,
            }
            
            # Type-specific analysis
            if series.dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                analysis.update(self._analyze_numeric_column(series))
            elif series.dtype == pl.Utf8:
                analysis.update(self._analyze_text_column(series))
            elif series.dtype == pl.Boolean:
                analysis.update(self._analyze_boolean_column(series))
            elif series.dtype in [pl.Date, pl.Datetime]:
                analysis.update(self._analyze_date_column(series))
            
            # Field type detection
            analysis['detected_field_type'] = self._detect_field_type(col, series)
            
            column_analysis[col] = analysis
            
        return column_analysis
    
    def _analyze_numeric_column(self, series: pl.Series) -> Dict[str, Any]:
        """Analyze numeric column characteristics."""
        try:
            valid_values = series.drop_nulls()
            if len(valid_values) == 0:
                return {'analysis_type': 'numeric', 'error': 'All values are null'}
            
            stats = {
                'analysis_type': 'numeric',
                'min': float(valid_values.min()),
                'max': float(valid_values.max()),
                'mean': float(valid_values.mean()),
                'median': float(valid_values.median()),
                'std': float(valid_values.std()) if valid_values.std() is not None else 0,
                'q25': float(valid_values.quantile(0.25)),
                'q75': float(valid_values.quantile(0.75)),
            }
            
            # Check for potential outliers using IQR
            iqr = stats['q75'] - stats['q25']
            lower_bound = stats['q25'] - 1.5 * iqr
            upper_bound = stats['q75'] + 1.5 * iqr
            outliers = valid_values.filter(
                (valid_values < lower_bound) | (valid_values > upper_bound)
            )
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = (len(outliers) / len(valid_values)) * 100
            
            # Check if values are integers (even if stored as float)
            stats['is_integer_like'] = all(v == int(v) for v in valid_values.to_list()[:100])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing numeric column: {e}")
            return {'analysis_type': 'numeric', 'error': str(e)}
    
    def _analyze_text_column(self, series: pl.Series) -> Dict[str, Any]:
        """Analyze text column characteristics."""
        try:
            valid_values = series.drop_nulls()
            if len(valid_values) == 0:
                return {'analysis_type': 'text', 'error': 'All values are null'}
            
            # Basic text statistics
            lengths = valid_values.str.len_chars()
            
            stats = {
                'analysis_type': 'text',
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'avg_length': float(lengths.mean()),
                'median_length': float(lengths.median()),
            }
            
            # Sample values for pattern analysis
            sample_values = valid_values.head(min(100, len(valid_values))).to_list()
            
            # Pattern detection
            stats['patterns'] = self._detect_text_patterns(sample_values)
            
            # Case analysis
            stats['case_analysis'] = self._analyze_text_case(sample_values)
            
            # Most common values (top 10)
            value_counts = valid_values.value_counts().head(10)
            stats['top_values'] = [
                {'value': row[0], 'count': row[1]} 
                for row in value_counts.iter_rows()
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing text column: {e}")
            return {'analysis_type': 'text', 'error': str(e)}
    
    def _analyze_boolean_column(self, series: pl.Series) -> Dict[str, Any]:
        """Analyze boolean column characteristics."""
        try:
            valid_values = series.drop_nulls()
            if len(valid_values) == 0:
                return {'analysis_type': 'boolean', 'error': 'All values are null'}
            
            value_counts = valid_values.value_counts()
            
            return {
                'analysis_type': 'boolean',
                'true_count': int(valid_values.sum()),
                'false_count': len(valid_values) - int(valid_values.sum()),
                'true_percentage': (int(valid_values.sum()) / len(valid_values)) * 100,
                'value_distribution': [
                    {'value': bool(row[0]), 'count': row[1]} 
                    for row in value_counts.iter_rows()
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing boolean column: {e}")
            return {'analysis_type': 'boolean', 'error': str(e)}
    
    def _analyze_date_column(self, series: pl.Series) -> Dict[str, Any]:
        """Analyze date column characteristics."""
        try:
            valid_values = series.drop_nulls()
            if len(valid_values) == 0:
                return {'analysis_type': 'date', 'error': 'All values are null'}
            
            return {
                'analysis_type': 'date',
                'min_date': str(valid_values.min()),
                'max_date': str(valid_values.max()),
                'date_range_days': (valid_values.max() - valid_values.min()).days if hasattr((valid_values.max() - valid_values.min()), 'days') else 'N/A'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing date column: {e}")
            return {'analysis_type': 'date', 'error': str(e)}
    
    def _detect_text_patterns(self, sample_values: List[str]) -> Dict[str, Any]:
        """Detect common patterns in text data."""
        patterns = {
            'phone_like': 0,
            'email_like': 0,
            'numeric_like': 0,
            'alphanumeric': 0,
            'alpha_only': 0,
            'contains_special_chars': 0,
            'mixed_case': 0,
            'all_caps': 0,
            'all_lower': 0
        }
        
        phone_pattern = re.compile(r'[\d\s\-\(\)\.]{10,}')
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for value in sample_values:
            if not isinstance(value, str):
                continue
                
            # Pattern matching
            if phone_pattern.search(value):
                patterns['phone_like'] += 1
            if email_pattern.match(value):
                patterns['email_like'] += 1
            if value.isdigit():
                patterns['numeric_like'] += 1
            if value.isalnum():
                patterns['alphanumeric'] += 1
            if value.isalpha():
                patterns['alpha_only'] += 1
            if any(c in value for c in '!@#$%^&*()_+-=[]{}|;:,.<>?'):
                patterns['contains_special_chars'] += 1
            
            # Case analysis
            if value.isupper():
                patterns['all_caps'] += 1
            elif value.islower():
                patterns['all_lower'] += 1
            elif any(c.isupper() for c in value) and any(c.islower() for c in value):
                patterns['mixed_case'] += 1
        
        # Convert to percentages
        total = len(sample_values)
        return {k: (v / total) * 100 if total > 0 else 0 for k, v in patterns.items()}
    
    def _analyze_text_case(self, sample_values: List[str]) -> Dict[str, int]:
        """Analyze case distribution in text values."""
        case_counts = {'title': 0, 'upper': 0, 'lower': 0, 'mixed': 0, 'other': 0}
        
        for value in sample_values:
            if not isinstance(value, str):
                continue
                
            if value.istitle():
                case_counts['title'] += 1
            elif value.isupper():
                case_counts['upper'] += 1
            elif value.islower():
                case_counts['lower'] += 1
            elif any(c.isupper() for c in value) and any(c.islower() for c in value):
                case_counts['mixed'] += 1
            else:
                case_counts['other'] += 1
                
        return case_counts
    
    def _detect_field_type(self, col_name: str, series: pl.Series) -> str:
        """Detect the likely field type based on name and data patterns."""
        col_lower = col_name.lower()
        
        # Field type patterns
        field_patterns = {
            'id': ['id', 'identifier', 'code', 'number'],
            'name': ['name', 'customer', 'first', 'last', 'full'],
            'address': ['address', 'street', 'location', 'city', 'state'],
            'phone': ['phone', 'mobile', 'contact', 'tel', 'cell'],
            'email': ['email', 'mail', 'e-mail'],
            'date': ['date', 'time', 'created', 'updated', 'timestamp'],
            'currency': ['price', 'cost', 'amount', 'total', 'spent', 'payment'],
            'quantity': ['quantity', 'qty', 'count'],
            'category': ['category', 'type', 'class', 'method', 'status'],
            'boolean': ['discount', 'applied', 'active', 'enabled', 'flag', 'is_']
        }
        
        # Check column name patterns
        for field_type, patterns in field_patterns.items():
            if any(pattern in col_lower for pattern in patterns):
                # Validate with data type
                if field_type == 'boolean' and series.dtype in [pl.Boolean, pl.Utf8]:
                    return 'boolean'
                elif field_type == 'currency' and series.dtype in [pl.Float64, pl.Int64]:
                    return 'currency'
                elif field_type == 'quantity' and series.dtype in [pl.Int64, pl.Float64]:
                    return 'quantity'
                elif field_type in ['name', 'address', 'phone', 'email', 'category'] and series.dtype == pl.Utf8:
                    return field_type
                elif field_type == 'date' and series.dtype in [pl.Date, pl.Datetime, pl.Utf8]:
                    return 'date'
                elif field_type == 'id':
                    return 'id'
        
        # Fallback to data type-based detection
        if series.dtype in [pl.Int64, pl.Float64]:
            return 'numeric'
        elif series.dtype == pl.Utf8:
            return 'text'
        elif series.dtype == pl.Boolean:
            return 'boolean'
        elif series.dtype in [pl.Date, pl.Datetime]:
            return 'date'
        else:
            return 'unknown'
    
    def _assess_data_quality(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality metrics."""
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = sum(df[col].null_count() for col in df.columns)
        
        # Duplicate analysis
        duplicate_rows = len(df) - len(df.unique())
        
        quality_metrics = {
            'completeness': {
                'total_cells': total_cells,
                'null_cells': total_nulls,
                'completeness_ratio': 1 - (total_nulls / total_cells) if total_cells > 0 else 0
            },
            'uniqueness': {
                'total_rows': len(df),
                'unique_rows': len(df.unique()),
                'duplicate_rows': duplicate_rows,
                'uniqueness_ratio': len(df.unique()) / len(df) if len(df) > 0 else 0
            },
            'consistency': self._assess_consistency(df)
        }
        
        return quality_metrics
    
    def _assess_consistency(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Assess data consistency across columns."""
        consistency_issues = []
        
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                # Check for case inconsistency
                valid_values = df[col].drop_nulls()
                if len(valid_values) > 0:
                    sample = valid_values.head(min(100, len(valid_values)))
                    case_variety = len(set(val.lower() for val in sample.to_list() if isinstance(val, str)))
                    unique_variety = len(set(sample.to_list()))
                    
                    if case_variety < unique_variety * 0.8:  # Potential case issues
                        consistency_issues.append({
                            'column': col,
                            'issue': 'case_inconsistency',
                            'severity': 'medium'
                        })
        
        return {
            'issues_found': len(consistency_issues),
            'issues': consistency_issues
        }
    
    def _generate_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
        text_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
        
        stats = {
            'numeric_columns': len(numeric_cols),
            'text_columns': len(text_cols),
            'boolean_columns': len([col for col in df.columns if df[col].dtype == pl.Boolean]),
            'date_columns': len([col for col in df.columns if df[col].dtype in [pl.Date, pl.Datetime]])
        }
        
        if numeric_cols:
            # Basic statistics for numeric columns
            numeric_df = df.select(numeric_cols)
            stats['numeric_summary'] = {
                'mean_values': {col: float(numeric_df[col].mean()) for col in numeric_cols},
                'std_values': {col: float(numeric_df[col].std()) if numeric_df[col].std() is not None else 0 for col in numeric_cols}
            }
        
        return stats
    
    def _analyze_patterns(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns across the dataset."""
        patterns = {
            'null_patterns': {},
            'duplicate_patterns': {},
            'value_distributions': {}
        }
        
        # Null patterns by column
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                patterns['null_patterns'][col] = {
                    'count': null_count,
                    'percentage': (null_count / len(df)) * 100
                }
        
        # Value distribution patterns for categorical columns
        for col in df.columns:
            if df[col].dtype == pl.Utf8 and df[col].n_unique() < 20:  # Categorical with reasonable cardinality
                value_counts = df[col].value_counts().head(10)
                patterns['value_distributions'][col] = [
                    {'value': row[0], 'count': row[1], 'percentage': (row[1] / len(df)) * 100}
                    for row in value_counts.iter_rows()
                ]
        
        return patterns
    
    def _analyze_correlations(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
        
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        try:
            # Convert to pandas for correlation calculation
            numeric_df = df.select(numeric_cols).to_pandas()
            correlation_matrix = numeric_df.corr()
            
            # Find high correlations (> 0.7 or < -0.7)
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.9 else 'moderate'
                        })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'numeric_columns_analyzed': numeric_cols
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {'error': str(e)}
    
    def generate_profile_report(self, output_path: str = "../output/data_profile.html") -> str:
        """
        Generate comprehensive HTML report with visualizations.
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            Path to generated report
        """
        try:
            # Create visualizations
            figures = self._create_profile_visualizations()
            
            # Generate HTML report
            html_content = self._generate_html_report(figures)
            
            # Save report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Profile report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating profile report: {e}")
            raise
    
    def _create_profile_visualizations(self) -> Dict[str, str]:
        """Create visualizations for the profile report."""
        figures = {}
        
        try:
            # 1. Data Quality Overview
            quality = self.profile_results.get('quality', {})
            completeness = quality.get('completeness', {}).get('completeness_ratio', 0)
            uniqueness = quality.get('uniqueness', {}).get('uniqueness_ratio', 0)
            
            fig_quality = go.Figure(go.Bar(
                x=['Completeness', 'Uniqueness'],
                y=[completeness, uniqueness],
                marker_color=['#2E86AB', '#A23B72'],
                text=[f'{completeness:.2%}', f'{uniqueness:.2%}'],
                textposition='auto'
            ))
            fig_quality.update_layout(
                title='Data Quality Overview',
                yaxis_title='Ratio',
                yaxis=dict(range=[0, 1])
            )
            figures['quality_overview'] = fig_quality.to_html(include_plotlyjs=False)
            
            # 2. Missing Values by Column
            columns_data = self.profile_results.get('columns', {})
            missing_data = {
                col: data.get('null_percentage', 0) 
                for col, data in columns_data.items()
            }
            
            if missing_data:
                fig_missing = px.bar(
                    x=list(missing_data.keys()),
                    y=list(missing_data.values()),
                    title='Missing Values by Column (%)',
                    labels={'x': 'Columns', 'y': 'Missing Percentage'}
                )
                fig_missing.update_layout(xaxis_tickangle=-45)
                figures['missing_values'] = fig_missing.to_html(include_plotlyjs=False)
            
            # 3. Data Type Distribution
            overview = self.profile_results.get('overview', {})
            data_types = overview.get('data_types', {})
            type_counts = Counter(data_types.values())
            
            fig_types = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title='Data Type Distribution'
            )
            figures['data_types'] = fig_types.to_html(include_plotlyjs=False)
            
            # 4. Field Type Detection Results
            field_types = {
                col: data.get('detected_field_type', 'unknown')
                for col, data in columns_data.items()
            }
            field_type_counts = Counter(field_types.values())
            
            fig_field_types = px.bar(
                x=list(field_type_counts.keys()),
                y=list(field_type_counts.values()),
                title='Detected Field Types',
                labels={'x': 'Field Type', 'y': 'Count'}
            )
            figures['field_types'] = fig_field_types.to_html(include_plotlyjs=False)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            
        return figures
    
    def _generate_html_report(self, figures: Dict[str, str]) -> str:
        """Generate HTML report content."""
        
        overview = self.profile_results.get('overview', {})
        quality = self.profile_results.get('quality', {})
        columns_data = self.profile_results.get('columns', {})
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                .chart {{ margin: 20px 0; }}
                .column-details {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .alert-warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                .alert-info {{ background-color: #d1ecf1; border-color: #bee5eb; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Profile Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric">Rows: <strong>{overview.get('shape', [0, 0])[0]:,}</strong></div>
                <div class="metric">Columns: <strong>{overview.get('shape', [0, 0])[1]:,}</strong></div>
                <div class="metric">Memory Usage: <strong>{overview.get('memory_usage_mb', 0):.2f} MB</strong></div>
                <div class="metric">Total Cells: <strong>{overview.get('total_cells', 0):,}</strong></div>
            </div>
            
            <div class="section">
                <h2>Data Quality Metrics</h2>
                <div class="metric">Completeness: <strong>{quality.get('completeness', {}).get('completeness_ratio', 0):.2%}</strong></div>
                <div class="metric">Uniqueness: <strong>{quality.get('uniqueness', {}).get('uniqueness_ratio', 0):.2%}</strong></div>
                <div class="metric">Missing Cells: <strong>{quality.get('completeness', {}).get('null_cells', 0):,}</strong></div>
                <div class="metric">Duplicate Rows: <strong>{quality.get('uniqueness', {}).get('duplicate_rows', 0):,}</strong></div>
                
                <div class="chart">
                    {figures.get('quality_overview', '<p>Quality overview chart not available</p>')}
                </div>
            </div>
            
            <div class="section">
                <h2>Column Analysis</h2>
                <div class="chart">
                    {figures.get('missing_values', '<p>Missing values chart not available</p>')}
                </div>
                <div class="chart">
                    {figures.get('data_types', '<p>Data types chart not available</p>')}
                </div>
                <div class="chart">
                    {figures.get('field_types', '<p>Field types chart not available</p>')}
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Column Information</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Data Type</th>
                            <th>Detected Field Type</th>
                            <th>Null %</th>
                            <th>Unique Count</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add column details
        for col, data in columns_data.items():
            notes = []
            if data.get('outlier_count', 0) > 0:
                notes.append(f"{data['outlier_count']} outliers")
            if data.get('null_percentage', 0) > 20:
                notes.append("High missing values")
            
            html_template += f"""
                        <tr>
                            <td>{col}</td>
                            <td>{data.get('dtype', 'Unknown')}</td>
                            <td>{data.get('detected_field_type', 'Unknown')}</td>
                            <td>{data.get('null_percentage', 0):.1f}%</td>
                            <td>{data.get('unique_count', 0):,}</td>
                            <td>{'; '.join(notes) if notes else '-'}</td>
                        </tr>
            """
        
        html_template += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Data Quality Issues</h2>
        """
        
        # Add quality issues
        consistency = quality.get('consistency', {})
        if consistency.get('issues_found', 0) > 0:
            html_template += '<div class="alert alert-warning"><strong>Consistency Issues Found:</strong><ul>'
            for issue in consistency.get('issues', []):
                html_template += f"<li>{issue['column']}: {issue['issue']} (Severity: {issue['severity']})</li>"
            html_template += '</ul></div>'
        else:
            html_template += '<div class="alert alert-info">No major consistency issues detected.</div>'
        
        # Check for high missing value columns
        high_missing = [
            col for col, data in columns_data.items() 
            if data.get('null_percentage', 0) > 20
        ]
        if high_missing:
            html_template += f'<div class="alert alert-warning"><strong>Columns with >20% missing values:</strong> {", ".join(high_missing)}</div>'
        
        html_template += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            html_template += f"<li>{rec}</li>"
        
        html_template += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality recommendations based on profiling results."""
        recommendations = []
        
        quality = self.profile_results.get('quality', {})
        columns_data = self.profile_results.get('columns', {})
        
        # Completeness recommendations
        completeness_ratio = quality.get('completeness', {}).get('completeness_ratio', 1)
        if completeness_ratio < 0.95:
            recommendations.append(f"Consider addressing missing values (Current completeness: {completeness_ratio:.1%})")
        
        # Duplicate recommendations
        duplicate_rows = quality.get('uniqueness', {}).get('duplicate_rows', 0)
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows to improve data quality")
        
        # Column-specific recommendations
        for col, data in columns_data.items():
            null_pct = data.get('null_percentage', 0)
            if null_pct > 50:
                recommendations.append(f"Column '{col}' has {null_pct:.1f}% missing values - consider dropping or imputing")
            elif null_pct > 20:
                recommendations.append(f"Column '{col}' has {null_pct:.1f}% missing values - review imputation strategy")
            
            outlier_count = data.get('outlier_count', 0)
            if outlier_count > 0:
                outlier_pct = data.get('outlier_percentage', 0)
                if outlier_pct > 5:
                    recommendations.append(f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%) - review for data quality issues")
        
        # Field type recommendations
        unknown_fields = [
            col for col, data in columns_data.items() 
            if data.get('detected_field_type') == 'unknown'
        ]
        if unknown_fields:
            recommendations.append(f"Review field types for: {', '.join(unknown_fields[:5])}")
        
        if not recommendations:
            recommendations.append("Data quality appears good! Consider running outlier detection and consistency checks.")
        
        return recommendations
    
    def save_profile_json(self, output_path: str = "../output/data_profile.json") -> str:
        """
        Save complete profiling results as JSON.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            Path to saved file
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert any non-serializable objects to strings
            serializable_results = self._make_json_serializable(self.profile_results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Profile results saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving profile JSON: {e}")
            raise
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj