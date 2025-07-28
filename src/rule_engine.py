"""
Rule Engine Module
Intelligent rule matching and selection system for data cleaning.
"""

import polars as pl
import json
import re
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Intelligent rule engine that matches fields to appropriate cleaning rules
    based on field names, data types, and data patterns.
    """
    
    def __init__(self, rules_config_path: str = "../config/rules.json"):
        """
        Initialize rule engine with configuration.
        
        Args:
            rules_config_path: Path to rules configuration JSON file
        """
        self.rules_config = self._load_rules_config(rules_config_path)
        self.matched_rules = {}
        self.unmatched_fields = {}
        self.rule_application_log = []
        
    def _load_rules_config(self, config_path: str) -> Dict[str, Any]:
        """Load rules configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded rules configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load rules config: {e}")
            raise
    
    def analyze_and_match_rules(self, df: pl.DataFrame, field_analysis: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Analyze dataset and match appropriate rules to each field.
        
        Args:
            df: DataFrame to analyze
            field_analysis: Field analysis results from data profiler
            
        Returns:
            Dictionary mapping field names to applicable rules
        """
        logger.info("Starting rule matching analysis...")
        
        self.matched_rules = {}
        self.unmatched_fields = {}
        
        # First, apply universal rules to all fields
        universal_rules = self.rules_config.get('universal_rules', [])
        for field in df.columns:
            self.matched_rules[field] = [rule for rule in universal_rules if rule.get('enabled', True)]
        
        # Then match field-specific rules
        field_specific_rules = self.rules_config.get('field_specific_rules', [])
        for field in df.columns:
            field_info = field_analysis.get(field, {})
            applicable_rules = self._match_field_specific_rules(
                field, df[field], field_info, field_specific_rules
            )
            self.matched_rules[field].extend(applicable_rules)
        
        # Identify unmatched fields for flagging
        self._identify_unmatched_fields(df, field_analysis)
        
        # Apply dataset-specific rules
        dataset_rules = self.rules_config.get('dataset_specific_rules', [])
        self._apply_dataset_specific_rules(df, dataset_rules)
        
        logger.info(f"Rule matching completed. Matched rules for {len(self.matched_rules)} fields.")
        
        return self.matched_rules
    
    def _match_field_specific_rules(self, field_name: str, field_data: pl.Series, 
                                   field_info: Dict[str, Any], 
                                   field_specific_rules: List[Dict]) -> List[Dict]:
        """
        Match field-specific rules based on field name, data type, and patterns.
        
        Args:
            field_name: Name of the field
            field_data: Field data series
            field_info: Analysis information about the field
            field_specific_rules: List of available field-specific rules
            
        Returns:
            List of applicable rules for this field
        """
        applicable_rules = []
        field_lower = field_name.lower()
        field_dtype = str(field_data.dtype).lower()
        detected_type = field_info.get('detected_field_type', 'unknown')
        
        for rule in field_specific_rules:
            if not rule.get('enabled', True):
                continue
                
            # Check field name patterns
            name_patterns = rule.get('field_patterns', [])
            name_match = any(pattern.lower() in field_lower for pattern in name_patterns)
            
            # Check data type compatibility
            rule_data_types = [dt.lower() for dt in rule.get('data_types', [])]
            dtype_match = any(
                dtype in field_dtype or dtype in detected_type 
                for dtype in rule_data_types
            )
            
            # Special handling for boolean fields
            if rule.get('rule_id') == 'standardize_boolean':
                boolean_match = (
                    field_data.dtype == pl.Boolean or
                    detected_type == 'boolean' or
                    self._check_boolean_patterns(field_data)
                )
                if boolean_match:
                    applicable_rules.append(rule)
                    logger.debug(f"Matched boolean rule to field '{field_name}'")
                continue
            
            # Regular matching logic
            if name_match and dtype_match:
                applicable_rules.append(rule)
                logger.debug(f"Matched rule '{rule['rule_id']}' to field '{field_name}' (name + dtype)")
            elif name_match and not rule_data_types:  # Rule doesn't specify data types
                applicable_rules.append(rule)
                logger.debug(f"Matched rule '{rule['rule_id']}' to field '{field_name}' (name only)")
            elif self._advanced_pattern_matching(field_name, field_data, field_info, rule):
                applicable_rules.append(rule)
                logger.debug(f"Matched rule '{rule['rule_id']}' to field '{field_name}' (advanced pattern)")
        
        return applicable_rules
    
    def _check_boolean_patterns(self, field_data: pl.Series) -> bool:
        """
        Check if field contains boolean-like patterns.
        
        Args:
            field_data: Field data to analyze
            
        Returns:
            True if field appears to contain boolean values
        """
        if field_data.dtype == pl.Boolean:
            return True
            
        # Check string patterns
        if field_data.dtype == pl.Utf8:
            unique_values = set(str(val).lower() for val in field_data.drop_nulls().unique().to_list())
            boolean_indicators = {
                'true', 'false', 'yes', 'no', '1', '0', 
                'y', 'n', 'on', 'off', 'enabled', 'disabled'
            }
            
            # If most unique values are boolean indicators
            overlap = len(unique_values.intersection(boolean_indicators))
            return overlap >= len(unique_values) * 0.7
        
        # Check numeric boolean (0/1)
        if field_data.dtype in [pl.Int64, pl.Int32, pl.Float64]:
            unique_values = set(field_data.drop_nulls().unique().to_list())
            return unique_values.issubset({0, 1, 0.0, 1.0})
        
        return False
    
    def _advanced_pattern_matching(self, field_name: str, field_data: pl.Series, 
                                 field_info: Dict[str, Any], rule: Dict) -> bool:
        """
        Advanced pattern matching for complex rules.
        
        Args:
            field_name: Name of the field
            field_data: Field data series
            field_info: Analysis information about the field
            rule: Rule to check for matching
            
        Returns:
            True if rule applies based on advanced patterns
        """
        rule_id = rule.get('rule_id')
        
        # Phone number detection
        if rule_id == 'normalize_phone':
            return self._detect_phone_patterns(field_data)
        
        # Email detection
        elif rule_id == 'normalize_email':
            return self._detect_email_patterns(field_data)
        
        # Currency detection
        elif rule_id == 'normalize_currency':
            return self._detect_currency_patterns(field_data, field_info)
        
        # Quantity detection
        elif rule_id == 'normalize_quantities':
            return self._detect_quantity_patterns(field_data, field_info)
        
        # ID field detection
        elif rule_id == 'validate_ids':
            return self._detect_id_patterns(field_name, field_data)
        
        return False
    
    def _detect_phone_patterns(self, field_data: pl.Series) -> bool:
        """Detect if field contains phone number patterns."""
        if field_data.dtype != pl.Utf8:
            return False
        
        sample_values = field_data.drop_nulls().head(min(50, len(field_data))).to_list()
        phone_pattern = re.compile(r'[\d\s\-\(\)\.]{10,}')
        
        phone_like_count = sum(
            1 for val in sample_values 
            if isinstance(val, str) and phone_pattern.search(val)
        )
        
        return phone_like_count >= len(sample_values) * 0.7
    
    def _detect_email_patterns(self, field_data: pl.Series) -> bool:
        """Detect if field contains email patterns."""
        if field_data.dtype != pl.Utf8:
            return False
        
        sample_values = field_data.drop_nulls().head(min(50, len(field_data))).to_list()
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        email_like_count = sum(
            1 for val in sample_values 
            if isinstance(val, str) and email_pattern.match(val)
        )
        
        return email_like_count >= len(sample_values) * 0.7
    
    def _detect_currency_patterns(self, field_data: pl.Series, field_info: Dict) -> bool:
        """Detect if field contains currency patterns."""
        # Numeric fields with decimal places often indicate currency
        if field_data.dtype in [pl.Float64, pl.Float32]:
            # Check if values commonly have 2 decimal places
            sample = field_data.drop_nulls().head(100)
            if len(sample) > 0:
                decimal_check = [
                    len(str(float(val)).split('.')[-1]) == 2 
                    for val in sample.to_list()
                    if '.' in str(float(val))
                ]
                return len(decimal_check) > 0 and sum(decimal_check) >= len(decimal_check) * 0.5
        
        # String fields with currency symbols
        elif field_data.dtype == pl.Utf8:
            sample_values = field_data.drop_nulls().head(50).to_list()
            currency_indicators = sum(
                1 for val in sample_values 
                if isinstance(val, str) and any(symbol in val for symbol in ['$', '€', '£', '¥'])
            )
            return currency_indicators >= len(sample_values) * 0.3
        
        return False
    
    def _detect_quantity_patterns(self, field_data: pl.Series, field_info: Dict) -> bool:
        """Detect if field contains quantity patterns."""
        if field_data.dtype in [pl.Int64, pl.Int32]:
            # Integer fields with positive values often indicate quantities
            sample = field_data.drop_nulls().head(100)
            if len(sample) > 0:
                positive_count = sum(1 for val in sample.to_list() if val >= 0)
                return positive_count >= len(sample) * 0.9
        
        return False
    
    def _detect_id_patterns(self, field_name: str, field_data: pl.Series) -> bool:
        """Detect if field contains ID patterns."""
        # Check for ID-like naming
        id_indicators = ['id', 'identifier', 'key', 'code', 'number']
        name_has_id = any(indicator in field_name.lower() for indicator in id_indicators)
        
        if not name_has_id:
            return False
        
        # Check for high uniqueness (typical of IDs)
        uniqueness_ratio = field_data.n_unique() / len(field_data)
        return uniqueness_ratio > 0.8
    
    def _identify_unmatched_fields(self, df: pl.DataFrame, field_analysis: Dict[str, Any]):
        """
        Identify fields that didn't match any field-specific rules.
        
        Args:
            df: DataFrame being analyzed
            field_analysis: Field analysis results
        """
        for field in df.columns:
            field_specific_rules = [
                rule for rule in self.matched_rules.get(field, [])
                if rule not in self.rules_config.get('universal_rules', [])
            ]
            
            if not field_specific_rules:
                # Field only has universal rules - flag for analysis
                field_info = field_analysis.get(field, {})
                field_data = df[field]
                
                self.unmatched_fields[field] = {
                    'data_type': str(field_data.dtype),
                    'detected_field_type': field_info.get('detected_field_type', 'unknown'),
                    'unique_count': field_data.n_unique(),
                    'null_percentage': field_info.get('null_percentage', 0),
                    'analysis': self._analyze_unmatched_field(field_data, field_info),
                    'reason': 'No field-specific rules matched'
                }
                
                logger.warning(f"Field '{field}' did not match any field-specific rules")
    
    def _analyze_unmatched_field(self, field_data: pl.Series, field_info: Dict) -> Dict[str, Any]:
        """
        Analyze unmatched field to provide insights.
        
        Args:
            field_data: Field data series
            field_info: Field analysis information
            
        Returns:
            Analysis results for unmatched field
        """
        analysis = {
            'sample_values': [],
            'value_characteristics': {},
            'recommendations': []
        }
        
        # Get sample values
        sample = field_data.drop_nulls().head(10)
        analysis['sample_values'] = sample.to_list()
        
        # Analyze value characteristics
        if field_data.dtype == pl.Utf8:
            # Text analysis
            valid_values = field_data.drop_nulls()
            if len(valid_values) > 0:
                lengths = valid_values.str.len_chars()
                analysis['value_characteristics'] = {
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'avg_length': float(lengths.mean()),
                    'contains_numbers': any(
                        any(c.isdigit() for c in str(val)) 
                        for val in sample.to_list()
                    ),
                    'contains_special_chars': any(
                        any(c in str(val) for c in '!@#$%^&*()_+-=[]{}|;:,.<>?') 
                        for val in sample.to_list()
                    ),
                    'all_uppercase': any(str(val).isupper() for val in sample.to_list()),
                    'mixed_case': any(
                        any(c.isupper() for c in str(val)) and any(c.islower() for c in str(val))
                        for val in sample.to_list()
                    )
                }
                
                # Generate recommendations
                if analysis['value_characteristics']['all_uppercase']:
                    analysis['recommendations'].append("Consider normalizing case to title case")
                if analysis['value_characteristics']['contains_special_chars']:
                    analysis['recommendations'].append("Review special characters for consistency")
                
        elif field_data.dtype in [pl.Int64, pl.Float64]:
            # Numeric analysis
            valid_values = field_data.drop_nulls()
            if len(valid_values) > 0:
                analysis['value_characteristics'] = {
                    'min_value': float(valid_values.min()),
                    'max_value': float(valid_values.max()),
                    'mean_value': float(valid_values.mean()),
                    'has_negatives': any(val < 0 for val in valid_values.to_list()),
                    'has_decimals': field_data.dtype in [pl.Float64, pl.Float32],
                    'range': float(valid_values.max() - valid_values.min())
                }
                
                # Generate recommendations
                if analysis['value_characteristics']['has_negatives']:
                    analysis['recommendations'].append("Check if negative values are valid")
                if analysis['value_characteristics']['range'] > 1000000:
                    analysis['recommendations'].append("Consider scaling for large value ranges")
        
        if not analysis['recommendations']:
            analysis['recommendations'].append("Review field purpose and consider creating custom rule")
        
        return analysis
    
    def _apply_dataset_specific_rules(self, df: pl.DataFrame, dataset_rules: List[Dict]):
        """
        Apply dataset-specific business rules.
        
        Args:
            df: DataFrame to analyze
            dataset_rules: List of dataset-specific rules
        """
        for rule in dataset_rules:
            if not rule.get('enabled', True):
                continue
                
            rule_id = rule.get('rule_id')
            
            if rule_id == 'validate_transaction_totals':
                required_fields = rule.get('required_fields', [])
                if all(field in df.columns for field in required_fields):
                    # This rule applies globally, not to specific fields
                    self.matched_rules['_dataset_rules'] = self.matched_rules.get('_dataset_rules', [])
                    self.matched_rules['_dataset_rules'].append(rule)
                    logger.info(f"Applied dataset rule: {rule_id}")
            
            elif rule_id == 'validate_discount_range':
                target_field = rule.get('field')
                if target_field and target_field in df.columns:
                    self.matched_rules[target_field] = self.matched_rules.get(target_field, [])
                    self.matched_rules[target_field].append(rule)
                    logger.info(f"Applied dataset rule: {rule_id} to field {target_field}")
    
    def get_rules_for_field(self, field_name: str) -> List[Dict]:
        """
        Get all applicable rules for a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of applicable rules sorted by priority
        """
        rules = self.matched_rules.get(field_name, [])
        # Sort by priority (lower number = higher priority)
        return sorted(rules, key=lambda x: x.get('priority', 999))
    
    def get_unmatched_fields_summary(self) -> Dict[str, Any]:
        """
        Get summary of fields that didn't match specific rules.
        
        Returns:
            Summary of unmatched fields with analysis
        """
        return {
            'count': len(self.unmatched_fields),
            'fields': self.unmatched_fields,
            'recommendations': self._generate_unmatched_recommendations()
        }
    
    def _generate_unmatched_recommendations(self) -> List[str]:
        """Generate recommendations for handling unmatched fields."""
        recommendations = []
        
        if not self.unmatched_fields:
            return ["All fields successfully matched to cleaning rules"]
        
        # Analyze patterns in unmatched fields
        text_fields = [
            field for field, info in self.unmatched_fields.items()
            if 'Utf8' in info['data_type']
        ]
        
        numeric_fields = [
            field for field, info in self.unmatched_fields.items()
            if any(num_type in info['data_type'] for num_type in ['Int', 'Float'])
        ]
        
        if text_fields:
            recommendations.append(
                f"Consider creating custom text cleaning rules for: {', '.join(text_fields[:3])}"
            )
        
        if numeric_fields:
            recommendations.append(
                f"Review numeric fields for potential scaling or validation: {', '.join(numeric_fields[:3])}"
            )
        
        # Check for high null percentage fields
        high_null_fields = [
            field for field, info in self.unmatched_fields.items()
            if info['null_percentage'] > 50
        ]
        
        if high_null_fields:
            recommendations.append(
                f"Fields with >50% missing values may need special handling: {', '.join(high_null_fields)}"
            )
        
        return recommendations
    
    def generate_rule_coverage_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report on rule coverage and matching.
        
        Returns:
            Rule coverage analysis report
        """
        # Count rule applications
        rule_usage = {}
        field_coverage = {}
        
        for field, rules in self.matched_rules.items():
            if field == '_dataset_rules':
                continue
                
            field_coverage[field] = len(rules)
            
            for rule in rules:
                rule_id = rule.get('rule_id')
                if rule_id not in rule_usage:
                    rule_usage[rule_id] = {
                        'count': 0,
                        'fields': [],
                        'rule_type': rule.get('rule_type'),
                        'confidence': rule.get('confidence', 0)
                    }
                rule_usage[rule_id]['count'] += 1
                rule_usage[rule_id]['fields'].append(field)
        
        # Calculate coverage statistics
        total_fields = len(self.matched_rules) - (1 if '_dataset_rules' in self.matched_rules else 0)
        fields_with_specific_rules = sum(
            1 for field, rules in self.matched_rules.items()
            if field != '_dataset_rules' and any(
                rule not in self.rules_config.get('universal_rules', [])
                for rule in rules
            )
        )
        
        coverage_stats = {
            'total_fields': total_fields,
            'fields_with_specific_rules': fields_with_specific_rules,
            'fields_with_only_universal_rules': total_fields - fields_with_specific_rules,
            'coverage_percentage': (fields_with_specific_rules / total_fields * 100) if total_fields > 0 else 0,
            'unmatched_fields_count': len(self.unmatched_fields)
        }
        
        return {
            'coverage_statistics': coverage_stats,
            'rule_usage': rule_usage,
            'field_coverage': field_coverage,
            'unmatched_fields': self.get_unmatched_fields_summary(),
            'recommendations': self._generate_coverage_recommendations(coverage_stats)
        }
    
    def _generate_coverage_recommendations(self, coverage_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []
        
        coverage_pct = coverage_stats.get('coverage_percentage', 0)
        
        if coverage_pct < 50:
            recommendations.append(
                "Low rule coverage detected. Consider adding more field-specific rules."
            )
        elif coverage_pct < 80:
            recommendations.append(
                "Moderate rule coverage. Review unmatched fields for potential rule additions."
            )
        else:
            recommendations.append(
                "Good rule coverage achieved. Focus on fine-tuning existing rules."
            )
        
        unmatched_count = coverage_stats.get('unmatched_fields_count', 0)
        if unmatched_count > 0:
            recommendations.append(
                f"{unmatched_count} fields lack specific rules. Review these for custom rule creation."
            )
        
        return recommendations
    
    def save_rule_analysis(self, output_path: str = "../output/rule_analysis.json") -> str:
        """
        Save complete rule analysis to JSON file.
        
        Args:
            output_path: Path to save analysis file
            
        Returns:
            Path to saved file
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            analysis = {
                'matched_rules': self.matched_rules,
                'unmatched_fields': self.unmatched_fields,
                'coverage_report': self.generate_rule_coverage_report(),
                'configuration_used': {
                    'total_universal_rules': len(self.rules_config.get('universal_rules', [])),
                    'total_field_specific_rules': len(self.rules_config.get('field_specific_rules', [])),
                    'total_dataset_specific_rules': len(self.rules_config.get('dataset_specific_rules', [])),
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Rule analysis saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving rule analysis: {e}")
            raise
    
    def get_confidence_distribution(self) -> Dict[str, Any]:
        """
        Analyze confidence distribution of matched rules.
        
        Returns:
            Confidence distribution analysis
        """
        confidences = []
        confidence_by_type = {}
        
        for field, rules in self.matched_rules.items():
            if field == '_dataset_rules':
                continue
                
            for rule in rules:
                confidence = rule.get('confidence', 1.0)
                rule_type = rule.get('rule_type', 'unknown')
                
                confidences.append(confidence)
                
                if rule_type not in confidence_by_type:
                    confidence_by_type[rule_type] = []
                confidence_by_type[rule_type].append(confidence)
        
        if not confidences:
            return {'message': 'No rules matched for confidence analysis'}
        
        # Calculate statistics
        import numpy as np
        
        confidence_stats = {
            'overall': {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            },
            'by_rule_type': {
                rule_type: {
                    'mean': float(np.mean(conf_list)),
                    'count': len(conf_list)
                }
                for rule_type, conf_list in confidence_by_type.items()
            },
            'distribution_bins': {
                'high_confidence_0.9+': sum(1 for c in confidences if c >= 0.9),
                'medium_confidence_0.7-0.9': sum(1 for c in confidences if 0.7 <= c < 0.9),
                'low_confidence_<0.7': sum(1 for c in confidences if c < 0.7)
            }
        }
        
        return confidence_stats