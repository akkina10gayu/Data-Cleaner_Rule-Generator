"""
Data Cleaner Module
Implements actual data cleaning operations based on matched rules.
"""

import polars as pl
import pandas as pd
import numpy as np
import re
import phonenumbers
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Executes data cleaning operations based on matched rules.
    Handles all cleaning transformations with comprehensive error handling.
    """
    
    def __init__(self):
        self.cleaning_log = []
        self.error_log = []
        self.transformation_stats = {}
        
    def clean_dataset(self, df: pl.DataFrame, matched_rules: Dict[str, List[Dict]], 
                     confidence_threshold: float = 0.7) -> pl.DataFrame:
        """
        Clean dataset by applying matched rules.
        
        Args:
            df: Original DataFrame
            matched_rules: Rules matched by rule engine
            confidence_threshold: Minimum confidence to apply rule
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning with confidence threshold: {confidence_threshold}")
        
        # Create working copy
        cleaned_df = df.clone()
        
        # Track initial state
        initial_stats = self._calculate_initial_stats(df)
        
        # Apply universal rules first (highest priority)
        cleaned_df = self._apply_universal_rules(cleaned_df, matched_rules, confidence_threshold)
        
        # Apply field-specific rules
        cleaned_df = self._apply_field_specific_rules(cleaned_df, matched_rules, confidence_threshold)
        
        # Apply dataset-specific rules last
        cleaned_df = self._apply_dataset_specific_rules(cleaned_df, matched_rules, confidence_threshold)
        
        # Calculate final statistics
        final_stats = self._calculate_final_stats(cleaned_df, initial_stats)
        self.transformation_stats.update(final_stats)
        
        logger.info(f"Data cleaning completed. Applied {len(self.cleaning_log)} operations.")
        
        return cleaned_df
    
    def _calculate_initial_stats(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate initial dataset statistics."""
        return {
            'initial_shape': df.shape,
            'initial_nulls': sum(df[col].null_count() for col in df.columns),
            'initial_duplicates': len(df) - len(df.unique()),
            'initial_memory_mb': df.estimated_size() / (1024 * 1024)
        }
    
    def _calculate_final_stats(self, df: pl.DataFrame, initial_stats: Dict) -> Dict[str, Any]:
        """Calculate final statistics and improvements."""
        final_nulls = sum(df[col].null_count() for col in df.columns)
        final_duplicates = len(df) - len(df.unique())
        
        return {
            'final_shape': df.shape,
            'final_nulls': final_nulls,
            'final_duplicates': final_duplicates,
            'final_memory_mb': df.estimated_size() / (1024 * 1024),
            'improvements': {
                'nulls_fixed': initial_stats['initial_nulls'] - final_nulls,
                'duplicates_removed': initial_stats['initial_duplicates'] - final_duplicates,
                'rows_removed': initial_stats['initial_shape'][0] - df.shape[0]
            }
        }
    
    def _apply_universal_rules(self, df: pl.DataFrame, matched_rules: Dict, 
                              confidence_threshold: float) -> pl.DataFrame:
        """Apply universal rules that affect the entire dataset."""
        
        # Get universal rules from any field (they're the same for all)
        universal_rules = []
        for field_rules in matched_rules.values():
            for rule in field_rules:
                if (rule.get('rule_type') in ['deduplication', 'standardization'] and 
                    rule.get('rule_id') in ['remove_exact_duplicates', 'trim_whitespace']):
                    if rule not in universal_rules:
                        universal_rules.append(rule)
        
        # Sort by priority
        universal_rules.sort(key=lambda x: x.get('priority', 999))
        
        for rule in universal_rules:
            if rule.get('confidence', 1.0) < confidence_threshold:
                continue
                
            try:
                if rule.get('rule_id') == 'remove_exact_duplicates':
                    df = self._remove_duplicates(df, rule)
                elif rule.get('rule_id') == 'trim_whitespace':
                    df = self._trim_whitespace(df, rule)
                    
            except Exception as e:
                self._log_error(rule.get('rule_id', 'unknown'), 'universal', str(e))
        
        return df
    
    def _apply_field_specific_rules(self, df: pl.DataFrame, matched_rules: Dict, 
                                   confidence_threshold: float) -> pl.DataFrame:
        """Apply field-specific cleaning rules."""
        
        for field_name, rules in matched_rules.items():
            if field_name == '_dataset_rules':
                continue
                
            # Filter field-specific rules and sort by priority
            field_rules = [
                rule for rule in rules 
                if rule.get('rule_type') not in ['deduplication'] or 
                rule.get('rule_id') not in ['remove_exact_duplicates', 'trim_whitespace']
            ]
            field_rules.sort(key=lambda x: x.get('priority', 999))
            
            for rule in field_rules:
                if rule.get('confidence', 1.0) < confidence_threshold:
                    continue
                    
                try:
                    df = self._apply_single_field_rule(df, field_name, rule)
                    
                except Exception as e:
                    self._log_error(rule.get('rule_id', 'unknown'), field_name, str(e))
        
        return df
    
    def _apply_dataset_specific_rules(self, df: pl.DataFrame, matched_rules: Dict, 
                                     confidence_threshold: float) -> pl.DataFrame:
        """Apply dataset-specific business rules."""
        
        dataset_rules = matched_rules.get('_dataset_rules', [])
        
        for rule in dataset_rules:
            if rule.get('confidence', 1.0) < confidence_threshold:
                continue
                
            try:
                if rule.get('rule_id') == 'validate_transaction_totals':
                    df = self._validate_transaction_totals(df, rule)
                    
            except Exception as e:
                self._log_error(rule.get('rule_id', 'unknown'), '_dataset', str(e))
        
        return df
    
    def _remove_duplicates(self, df: pl.DataFrame, rule: Dict) -> pl.DataFrame:
        """Remove exact duplicate rows."""
        initial_count = len(df)
        df_cleaned = df.unique()
        removed_count = initial_count - len(df_cleaned)
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field='_all',
            operation='remove_duplicates',
            result=f"Removed {removed_count} duplicate rows",
            confidence=rule.get('confidence', 1.0),
            records_affected=removed_count
        )
        
        return df_cleaned
    
    def _trim_whitespace(self, df: pl.DataFrame, rule: Dict) -> pl.DataFrame:
        """Trim whitespace from all string columns."""
        string_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]
        
        for col in string_columns:
            df = df.with_columns(pl.col(col).str.strip().alias(col))
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field='_text_fields',
            operation='trim_whitespace',
            result=f"Trimmed whitespace in {len(string_columns)} columns",
            confidence=rule.get('confidence', 1.0),
            records_affected=len(df)
        )
        
        return df
    
    def _apply_single_field_rule(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Apply a single field-specific rule."""
        rule_type = rule.get('rule_type')
        rule_id = rule.get('rule_id')
        
        if rule_type == 'missing_value':
            return self._handle_missing_values(df, field_name, rule)
        elif rule_type == 'normalization':
            return self._apply_normalization(df, field_name, rule)
        elif rule_type == 'standardization':
            return self._apply_standardization(df, field_name, rule)
        elif rule_type == 'validation':
            return self._apply_validation(df, field_name, rule)
        elif rule_type == 'outlier_detection':
            return self._detect_outliers(df, field_name, rule)
        else:
            logger.warning(f"Unknown rule type: {rule_type} for rule {rule_id}")
            return df
    
    def _handle_missing_values(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Handle missing values in a field."""
        if field_name not in df.columns:
            return df
            
        initial_nulls = df[field_name].null_count()
        if initial_nulls == 0:
            return df
        
        strategy = rule.get('strategy', 'drop')
        
        try:
            if strategy == 'median' and df[field_name].dtype in [pl.Int64, pl.Float64]:
                fill_value = df[field_name].median()
                df = df.with_columns(pl.col(field_name).fill_null(fill_value).alias(field_name))
            
            elif strategy == 'mode':
                # Get mode for categorical data
                mode_result = df[field_name].mode()
                if len(mode_result) > 0:
                    fill_value = mode_result[0]
                else:
                    fill_value = 'Unknown'
                df = df.with_columns(pl.col(field_name).fill_null(fill_value).alias(field_name))
            
            elif strategy == 'forward_fill':
                df = df.with_columns(pl.col(field_name).fill_null_forward().alias(field_name))
            
            else:  # Default to Unknown for strings, 0 for numbers
                if df[field_name].dtype == pl.Utf8:
                    fill_value = 'Unknown'
                else:
                    fill_value = 0
                df = df.with_columns(pl.col(field_name).fill_null(fill_value).alias(field_name))
            
            final_nulls = df[field_name].null_count()
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field=field_name,
                operation='fill_missing',
                result=f"Filled {initial_nulls - final_nulls} missing values using {strategy}",
                confidence=rule.get('confidence', 1.0),
                records_affected=initial_nulls - final_nulls
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), field_name, f"Missing value handling failed: {e}")
        
        return df
    
    def _apply_normalization(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Apply normalization rules."""
        if field_name not in df.columns:
            return df
            
        rule_id = rule.get('rule_id')
        
        try:
            if rule_id == 'normalize_phone':
                df = self._normalize_phone_numbers(df, field_name, rule)
            elif rule_id == 'normalize_currency':
                df = self._normalize_currency(df, field_name, rule)
            elif rule_id == 'normalize_quantities':
                df = self._normalize_quantities(df, field_name, rule)
            else:
                logger.warning(f"Unknown normalization rule: {rule_id}")
                
        except Exception as e:
            self._log_error(rule_id, field_name, f"Normalization failed: {e}")
        
        return df
    
    def _normalize_phone_numbers(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Normalize phone numbers to standard format."""
        
        def normalize_phone(phone_str):
            if phone_str is None or phone_str == '':
                return None
            
            try:
                # Remove all non-digit characters first
                digits = re.sub(r'\D', '', str(phone_str))
                
                # Handle US phone formats
                if len(digits) == 10:
                    return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
                elif len(digits) == 11 and digits[0] == '1':
                    return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
                
                # Try international parsing
                parsed = phonenumbers.parse(phone_str, "US")
                if phonenumbers.is_valid_number(parsed):
                    formatted = phonenumbers.format_number(
                        parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                    )
                    return formatted.replace(' ', '-')
                
            except Exception:
                pass
            
            return phone_str  # Return original if normalization fails
        
        # Apply normalization
        original_values = df[field_name]
        normalized_values = original_values.map_elements(normalize_phone, return_dtype=pl.Utf8)
        
        # Count changes
        changes = sum(1 for orig, norm in zip(original_values.to_list(), normalized_values.to_list()) 
                     if orig != norm and orig is not None)
        
        df = df.with_columns(normalized_values.alias(field_name))
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field=field_name,
            operation='normalize_phone',
            result=f"Normalized {changes} phone numbers",
            confidence=rule.get('confidence', 1.0),
            records_affected=changes
        )
        
        return df
    
    def _normalize_currency(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Normalize currency values."""
        decimal_places = rule.get('decimal_places', 2)
        
        try:
            # Handle string currency fields
            if df[field_name].dtype == pl.Utf8:
                # Remove currency symbols and convert to numeric
                df = df.with_columns(
                    pl.col(field_name)
                    .str.replace_all(r'[$,€£¥]', '')
                    .cast(pl.Float64, strict=False)
                    .alias(field_name)
                )
            
            # Round to specified decimal places
            df = df.with_columns(
                pl.col(field_name).round(decimal_places).alias(field_name)
            )
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field=field_name,
                operation='normalize_currency',
                result=f"Normalized currency to {decimal_places} decimal places",
                confidence=rule.get('confidence', 1.0),
                records_affected=len(df)
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), field_name, f"Currency normalization failed: {e}")
        
        return df
    
    def _normalize_quantities(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Normalize quantities to positive integers."""
        
        try:
            # Ensure numeric type
            if df[field_name].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(field_name).cast(pl.Float64, strict=False).alias(field_name)
                )
            
            # Convert to positive integers
            df = df.with_columns(
                pl.col(field_name).abs().round().cast(pl.Int64).alias(field_name)
            )
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field=field_name,
                operation='normalize_quantity',
                result="Converted to positive integers",
                confidence=rule.get('confidence', 1.0),
                records_affected=len(df)
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), field_name, f"Quantity normalization failed: {e}")
        
        return df
    
    def _apply_standardization(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Apply standardization rules."""
        if field_name not in df.columns:
            return df
            
        rule_id = rule.get('rule_id')
        
        try:
            if rule_id == 'standardize_names':
                df = self._standardize_names(df, field_name, rule)
            elif rule_id == 'standardize_addresses':
                df = self._standardize_addresses(df, field_name, rule)
            elif rule_id == 'standardize_categories':
                df = self._standardize_categories(df, field_name, rule)
            elif rule_id == 'standardize_boolean':
                df = self._standardize_boolean(df, field_name, rule)
            elif rule_id == 'parse_dates':
                df = self._parse_dates(df, field_name, rule)
            else:
                logger.warning(f"Unknown standardization rule: {rule_id}")
                
        except Exception as e:
            self._log_error(rule_id, field_name, f"Standardization failed: {e}")
        
        return df
    
    def _standardize_names(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Standardize name formatting."""
        
        # Apply title case and clean spacing
        df = df.with_columns(
            pl.col(field_name)
            .str.to_titlecase()
            .str.replace_all(r'\s+', ' ')
            .str.strip()
            .alias(field_name)
        )
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field=field_name,
            operation='standardize_names',
            result="Applied title case and cleaned spacing",
            confidence=rule.get('confidence', 1.0),
            records_affected=len(df)
        )
        
        return df
    
    def _standardize_boolean(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Standardize boolean values."""
        mappings = rule.get('mappings', {})
        
        def map_boolean(value):
            if value is None:
                return False  # Default for missing boolean values
            
            str_val = str(value).lower()
            
            # Direct mapping
            if value in mappings:
                return mappings[value]
            if str_val in {k.lower(): v for k, v in mappings.items()}:
                return next(v for k, v in mappings.items() if k.lower() == str_val)
            
            # Fallback boolean conversion
            if str_val in ['true', '1', 'yes', 'y', 'on', 'enabled']:
                return True
            elif str_val in ['false', '0', 'no', 'n', 'off', 'disabled']:
                return False
            
            return bool(value)  # Final fallback
        
        df = df.with_columns(
            pl.col(field_name).map_elements(map_boolean, return_dtype=pl.Boolean).alias(field_name)
        )
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field=field_name,
            operation='standardize_boolean',
            result="Standardized boolean values",
            confidence=rule.get('confidence', 1.0),
            records_affected=len(df)
        )
        
        return df
    
    def _standardize_addresses(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Standardize address formatting."""
        transformations = rule.get('transformations', {})
        abbreviations = transformations.get('abbreviations', {})
        
        # Apply title case
        df = df.with_columns(
            pl.col(field_name).str.to_titlecase().alias(field_name)
        )
        
        # Apply abbreviations
        for full_form, abbrev in abbreviations.items():
            df = df.with_columns(
                pl.col(field_name).str.replace_all(full_form, abbrev).alias(field_name)
            )
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field=field_name,
            operation='standardize_address',
            result="Applied address standardization and abbreviations",
            confidence=rule.get('confidence', 1.0),
            records_affected=len(df)
        )
        
        return df
    
    def _standardize_categories(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Standardize categorical values using fuzzy matching."""
        # For now, apply basic standardization (title case)
        # TODO: Implement fuzzy matching for similar categories
        # Alternative methods: Levenshtein distance, Jaccard similarity, embedding-based clustering
        
        df = df.with_columns(
            pl.col(field_name).str.to_titlecase().str.strip().alias(field_name)
        )
        
        self._log_operation(
            rule_id=rule.get('rule_id'),
            field=field_name,
            operation='standardize_categories',
            result="Applied basic category standardization",
            confidence=rule.get('confidence', 1.0),
            records_affected=len(df)
        )
        
        return df
    
    def _parse_dates(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Parse and standardize dates."""
        target_format = rule.get('target_format', '%Y-%m-%d')
        
        try:
            # Try to parse dates
            df = df.with_columns(
                pl.col(field_name).str.to_date(strict=False).alias(field_name)
            )
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field=field_name,
                operation='parse_dates',
                result=f"Parsed dates to standard format",
                confidence=rule.get('confidence', 1.0),
                records_affected=len(df)
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), field_name, f"Date parsing failed: {e}")
        
        return df
    
    def _apply_validation(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Apply validation rules and flag issues."""
        checks = rule.get('checks', [])
        issues_found = []
        
        try:
            if 'not_null' in checks:
                null_count = df[field_name].null_count()
                if null_count > 0:
                    issues_found.append(f"{null_count} null values")
            
            if 'unique' in checks:
                duplicate_count = len(df) - df[field_name].n_unique()
                if duplicate_count > 0:
                    issues_found.append(f"{duplicate_count} duplicate values")
            
            if 'format_consistency' in checks:
                # Check format consistency for string fields
                if df[field_name].dtype == pl.Utf8:
                    valid_values = df[field_name].drop_nulls()
                    if len(valid_values) > 0:
                        lengths = valid_values.str.len_chars()
                        length_variance = lengths.std()
                        mean_length = lengths.mean()
                        if length_variance > mean_length * 0.5:  # High variance indicates inconsistency
                            issues_found.append("Inconsistent format lengths detected")
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field=field_name,
                operation='validation',
                result=f"Validation issues: {'; '.join(issues_found) if issues_found else 'None'}",
                confidence=rule.get('confidence', 1.0),
                records_affected=0,
                validation_issues=issues_found
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), field_name, f"Validation failed: {e}")
        
        return df
    
    def _detect_outliers(self, df: pl.DataFrame, field_name: str, rule: Dict) -> pl.DataFrame:
        """Detect outliers using specified method."""
        if df[field_name].dtype not in [pl.Int64, pl.Float64]:
            return df
        
        method = rule.get('method', 'iqr')
        action = rule.get('action', 'flag')  # flag, remove, or cap
        
        try:
            valid_values = df[field_name].drop_nulls()
            if len(valid_values) < 4:  # Need minimum data for outlier detection
                return df
            
            outlier_indices = []
            
            if method == 'iqr':
                # IQR method - commonly used, robust to normal distribution assumption
                # Alternative methods: Z-score (assumes normal distribution), 
                # Isolation Forest (for multivariate outliers), 
                # Modified Z-score using MAD (more robust than Z-score)
                q1 = valid_values.quantile(0.25)
                q3 = valid_values.quantile(0.75)
                iqr = q3 - q1
                threshold = rule.get('threshold', 1.5)
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outlier_mask = (df[field_name] < lower_bound) | (df[field_name] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if action == 'cap' and outlier_count > 0:
                    # Cap outliers to bounds
                    df = df.with_columns(
                        pl.col(field_name).clip(lower_bound, upper_bound).alias(field_name)
                    )
                
            elif method == 'zscore':
                # Z-score method - assumes normal distribution
                # Alternative: Modified Z-score using median and MAD for non-normal data
                mean = valid_values.mean()
                std = valid_values.std()
                threshold = rule.get('threshold', 3)
                
                if std > 0:
                    z_scores = (df[field_name] - mean).abs() / std
                    outlier_mask = z_scores > threshold
                    outlier_count = outlier_mask.sum()
                    
                    if action == 'cap' and outlier_count > 0:
                        cap_lower = mean - threshold * std
                        cap_upper = mean + threshold * std
                        df = df.with_columns(
                            pl.col(field_name).clip(cap_lower, cap_upper).alias(field_name)
                        )
                else:
                    outlier_count = 0
            
            else:
                outlier_count = 0
                logger.warning(f"Unknown outlier detection method: {method}")
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field=field_name,
                operation='outlier_detection',
                result=f"Detected {outlier_count} outliers using {method} method",
                confidence=rule.get('confidence', 1.0),
                records_affected=outlier_count,
                outlier_info={
                    'method': method,
                    'count': int(outlier_count),
                    'action': action
                }
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), field_name, f"Outlier detection failed: {e}")
        
        return df
    
    def _validate_transaction_totals(self, df: pl.DataFrame, rule: Dict) -> pl.DataFrame:
        """Validate transaction totals against price * quantity."""
        required_fields = rule.get('required_fields', [])
        
        if not all(field in df.columns for field in required_fields):
            logger.warning("Required fields for transaction validation not found")
            return df
        
        try:
            total_field, price_field, qty_field = required_fields
            tolerance = rule.get('tolerance', 0.01)
            
            # Calculate expected total
            expected_total = df[price_field] * df[qty_field]
            
            # Find discrepancies
            discrepancy_mask = (df[total_field] - expected_total).abs() > tolerance
            discrepancy_count = discrepancy_mask.sum()
            
            if discrepancy_count > 0:
                # Fix discrepancies
                df = df.with_columns(
                    pl.when(discrepancy_mask)
                    .then(expected_total.round(2))
                    .otherwise(pl.col(total_field))
                    .alias(total_field)
                )
            
            self._log_operation(
                rule_id=rule.get('rule_id'),
                field='_transaction_validation',
                operation='validate_totals',
                result=f"Fixed {discrepancy_count} transaction total discrepancies",
                confidence=rule.get('confidence', 1.0),
                records_affected=discrepancy_count
            )
            
        except Exception as e:
            self._log_error(rule.get('rule_id'), '_transaction_validation', f"Transaction validation failed: {e}")
        
        return df
    
    def _log_operation(self, rule_id: str, field: str, operation: str, result: str, 
                      confidence: float, records_affected: int, **kwargs):
        """Log a cleaning operation."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'rule_id': rule_id,
            'field': field,
            'operation': operation,
            'result': result,
            'confidence': confidence,
            'records_affected': records_affected,
            'status': 'success'
        }
        log_entry.update(kwargs)
        self.cleaning_log.append(log_entry)
    
    def _log_error(self, rule_id: str, field: str, error_message: str):
        """Log a cleaning error."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'rule_id': rule_id,
            'field': field,
            'error': error_message,
            'status': 'error'
        }
        self.error_log.append(error_entry)
        logger.error(f"Cleaning error - Rule: {rule_id}, Field: {field}, Error: {error_message}")
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning summary."""
        successful_operations = [log for log in self.cleaning_log if log.get('status') == 'success']
        
        # Group operations by type
        operations_by_type = {}
        for op in successful_operations:
            op_type = op.get('operation', 'unknown')
            if op_type not in operations_by_type:
                operations_by_type[op_type] = []
            operations_by_type[op_type].append(op)
        
        # Calculate totals
        total_records_affected = sum(op.get('records_affected', 0) for op in successful_operations)
        
        summary = {
            'operations_performed': len(successful_operations),
            'errors_encountered': len(self.error_log),
            'total_records_affected': total_records_affected,
            'operations_by_type': {
                op_type: {
                    'count': len(ops),
                    'total_records_affected': sum(op.get('records_affected', 0) for op in ops)
                }
                for op_type, ops in operations_by_type.items()
            },
            'transformation_stats': self.transformation_stats,
            'error_summary': [
                {
                    'rule_id': error.get('rule_id'),
                    'field': error.get('field'),
                    'error': error.get('error')
                }
                for error in self.error_log
            ]
        }
        
        return summary
    
    def save_cleaning_report(self, output_path: str = "../output/cleaning_report.json") -> str:
        """Save detailed cleaning report."""
        try:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'summary': self.get_cleaning_summary(),
                'detailed_log': self.cleaning_log,
                'error_log': self.error_log,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Cleaning report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving cleaning report: {e}")
            raise