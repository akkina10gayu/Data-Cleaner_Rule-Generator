import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

import polars as pl
from data_cleaner import DataCleaner
from rule_engine import RuleEngine


class TestUniversalRules(unittest.TestCase):
    
    def setUp(self):
        self.cleaner = DataCleaner()
        self.rule_engine = RuleEngine()
    
    def test_remove_exact_duplicates(self):
        """Test remove_exact_duplicates rule"""
        # Create test data with duplicates
        df = pl.DataFrame({
            'id': [1, 2, 2, 3],
            'name': ['Alice', 'Bob', 'Bob', 'Charlie'],
            'amount': [100.0, 200.0, 200.0, 300.0]
        })
        
        rule = {
            'rule_id': 'remove_exact_duplicates',
            'rule_type': 'deduplication',
            'confidence': 1.0
        }
        
        result = self.cleaner._remove_duplicates(df, rule)
        
        # Should remove 1 duplicate row
        self.assertEqual(len(result), 3)
        self.assertEqual(len(df) - len(result), 1)
        
        # Verify no duplicates remain
        self.assertEqual(len(result.unique()), len(result))
    
    def test_trim_whitespace(self):
        """Test trim_whitespace rule"""
        df = pl.DataFrame({
            'name': [' Alice ', '  Bob  ', 'Charlie\t', ' \n David \r '],
            'city': ['  New York  ', 'Boston ', ' Chicago', 'Denver\n']
        })
        
        rule = {
            'rule_id': 'trim_whitespace',
            'rule_type': 'standardization',
            'confidence': 1.0
        }
        
        result = self.cleaner._trim_whitespace(df, rule)
        
        # Check that whitespace is trimmed
        expected_names = ['Alice', 'Bob', 'Charlie', 'David']
        expected_cities = ['New York', 'Boston', 'Chicago', 'Denver']
        
        self.assertEqual(result['name'].to_list(), expected_names)
        self.assertEqual(result['city'].to_list(), expected_cities)
    
    def test_handle_missing_numeric_median(self):
        """Test handle_missing_numeric rule with median strategy"""
        df = pl.DataFrame({
            'price': [10.0, None, 30.0, 40.0, None],
            'quantity': [1, 2, None, 4, 5]
        })
        
        rule = {
            'rule_id': 'handle_missing_numeric',
            'rule_type': 'missing_value',
            'strategy': 'median',
            'confidence': 0.85
        }
        
        # Test price field (median of [10.0, 30.0, 40.0] = 30.0)
        result = self.cleaner._handle_missing_values(df, 'price', rule)
        price_values = result['price'].to_list()
        
        # Should fill nulls with median (30.0)
        self.assertEqual(price_values, [10.0, 30.0, 30.0, 40.0, 30.0])
        self.assertEqual(result['price'].null_count(), 0)
    
    def test_handle_missing_categorical_mode(self):
        """Test handle_missing_categorical rule with mode strategy"""
        df = pl.DataFrame({
            'category': ['A', 'B', 'A', None, 'A', None],
            'status': ['active', None, 'active', 'inactive', None, 'active']
        })
        
        rule = {
            'rule_id': 'handle_missing_categorical',
            'rule_type': 'missing_value',
            'strategy': 'mode',
            'confidence': 0.80
        }
        
        # Test category field (mode = 'A')
        result = self.cleaner._handle_missing_values(df, 'category', rule)
        
        # Should fill nulls with mode ('A')
        expected = ['A', 'B', 'A', 'A', 'A', 'A']
        self.assertEqual(result['category'].to_list(), expected)


class TestFieldSpecificRules(unittest.TestCase):
    """Test field-specific rules based on field patterns and data types"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
        self.rule_engine = RuleEngine()
    
    def test_normalize_phone(self):
        """Test normalize_phone rule"""
        df = pl.DataFrame({
            'phone': ['1234567890', '123-456-7890', '(123) 456-7890', '123.456.7890', None]
        })
        
        rule = {
            'rule_id': 'normalize_phone',
            'rule_type': 'normalization',
            'confidence': 0.94
        }
        
        result = self.cleaner._normalize_phone_numbers(df, 'phone', rule)
        phone_values = result['phone'].to_list()
        
        # Should normalize to +1-XXX-XXX-XXXX format
        expected_format = '+1-123-456-7890'
        self.assertEqual(phone_values[0], expected_format)
        self.assertEqual(phone_values[1], expected_format)
        self.assertEqual(phone_values[2], expected_format)
        self.assertEqual(phone_values[3], expected_format)
        self.assertIsNone(phone_values[4])  # None should remain None
    
    def test_standardize_names(self):
        """Test standardize_names rule"""
        df = pl.DataFrame({
            'customer_name': ['john doe', 'JANE SMITH', 'bob   johnson', ' alice brown ']
        })
        
        rule = {
            'rule_id': 'standardize_names',
            'rule_type': 'standardization',
            'confidence': 0.92
        }
        
        result = self.cleaner._standardize_names(df, 'customer_name', rule)
        names = result['customer_name'].to_list()
        
        # Should be title case with clean spacing
        expected = ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown']
        self.assertEqual(names, expected)
    
    def test_standardize_addresses(self):
        """Test standardize_addresses rule"""
        df = pl.DataFrame({
            'address': ['123 main Street', '456 oak Avenue', '789 pine Road', '321 elm Boulevard']
        })
        
        rule = {
            'rule_id': 'standardize_addresses',
            'rule_type': 'standardization',
            'confidence': 0.88,
            'transformations': {
                'abbreviations': {
                    'Street': 'St',
                    'Avenue': 'Ave',
                    'Road': 'Rd',
                    'Boulevard': 'Blvd'
                }
            }
        }
        
        result = self.cleaner._standardize_addresses(df, 'address', rule)
        addresses = result['address'].to_list()
        
        # Should apply title case and abbreviations
        expected = ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm Blvd']
        self.assertEqual(addresses, expected)
    
    def test_parse_dates(self):
        """Test parse_dates rule"""
        df = pl.DataFrame({
            'transaction_date': ['2023-01-15', '01/15/2023', '15-Jan-2023', '2023-01-15 10:30:00']
        })
        
        rule = {
            'rule_id': 'parse_dates',
            'rule_type': 'standardization',
            'confidence': 0.95,
            'target_format': '%Y-%m-%d'
        }
        
        result = self.cleaner._parse_dates(df, 'transaction_date', rule)
        
        # Should parse all to date format
        self.assertEqual(result['transaction_date'].dtype, pl.Date)
    
    def test_normalize_currency(self):
        """Test normalize_currency rule"""
        df = pl.DataFrame({
            'price': ['$10.99', '€25.50', '£15.75', '¥1000', '$99.999'],
            'amount': [10.996, 25.503, 15.751, 99.99, 100.001]
        })
        
        rule = {
            'rule_id': 'normalize_currency',
            'rule_type': 'normalization',
            'confidence': 0.95,
            'decimal_places': 2
        }
        
        # Test string currency normalization
        result = self.cleaner._normalize_currency(df, 'price', rule)
        
        # Should remove symbols and round to 2 decimal places
        expected_prices = [10.99, 25.50, 15.75, 1000.00, 100.00]
        self.assertEqual(result['price'].to_list(), expected_prices)
        
        # Test numeric rounding
        result2 = self.cleaner._normalize_currency(df, 'amount', rule)
        expected_amounts = [11.00, 25.50, 15.75, 99.99, 100.00]
        self.assertEqual(result2['amount'].to_list(), expected_amounts)
    
    def test_standardize_categories(self):
        """Test standardize_categories rule"""
        df = pl.DataFrame({
            'category': ['electronics', 'FOOD', 'home & garden', 'books   ']
        })
        
        rule = {
            'rule_id': 'standardize_categories',
            'rule_type': 'standardization',
            'confidence': 0.85
        }
        
        result = self.cleaner._standardize_categories(df, 'category', rule)
        categories = result['category'].to_list()
        
        # Should apply title case and clean spacing
        expected = ['Electronics', 'Food', 'Home & Garden', 'Books']
        self.assertEqual(categories, expected)
    
    def test_normalize_email(self):
        """Test normalize_email rule"""
        df = pl.DataFrame({
            'customer_email': ['USER@EXAMPLE.COM', 'Test@Gmail.COM', '  admin@site.org  ', 'SUPPORT@COMPANY.NET']
        })
        
        rule = {
            'rule_id': 'normalize_email',
            'rule_type': 'normalization', 
            'confidence': 0.92
        }
        
        result = self.cleaner._normalize_email(df, 'customer_email', rule)
        emails = result['customer_email'].to_list()
        
        # Should normalize to lowercase and trim whitespace
        expected = ['user@example.com', 'test@gmail.com', 'admin@site.org', 'support@company.net']
        self.assertEqual(emails, expected)
        
        # Verify no nulls introduced
        self.assertEqual(result['customer_email'].null_count(), 0)
    
    def test_validate_ids(self):
        """Test validate_ids rule"""
        df = pl.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_001', None, 'INVALID'],
            'customer_id': ['CUST_01', 'CUST_02', 'CUST_03', 'CUST_04', 'CUST_05']
        })
        
        rule = {
            'rule_id': 'validate_ids',
            'rule_type': 'validation',
            'confidence': 0.90,
            'checks': ['not_null', 'unique', 'format_consistency']
        }
        
        result = self.cleaner._apply_validation(df, 'transaction_id', rule)
        
        # Should identify validation issues (duplicates, nulls)
        log_entry = self.cleaner.cleaning_log[-1]
        self.assertIn('duplicate', log_entry['result'].lower())
        self.assertIn('null', log_entry['result'].lower())
    
    def test_normalize_quantities(self):
        """Test normalize_quantities rule"""
        df = pl.DataFrame({
            'quantity': [-5, 10.7, 0, 15.9, -2.3]
        })
        
        rule = {
            'rule_id': 'normalize_quantities',
            'rule_type': 'normalization',
            'confidence': 0.95
        }
        
        result = self.cleaner._normalize_quantities(df, 'quantity', rule)
        quantities = result['quantity'].to_list()
        
        # Should convert to positive integers
        expected = [5, 11, 0, 16, 2]
        self.assertEqual(quantities, expected)
        self.assertEqual(result['quantity'].dtype, pl.Int64)
    
    def test_standardize_boolean(self):
        """Test standardize_boolean rule"""
        df = pl.DataFrame({
            'discount_applied': ['true', 'False', '1', '0', 'yes', 'no', None]
        })
        
        rule = {
            'rule_id': 'standardize_boolean',
            'rule_type': 'standardization',
            'confidence': 0.90,
            'mappings': {
                'true': True, 'True': True, '1': True, 'yes': True,
                'false': False, 'False': False, '0': False, 'no': False
            }
        }
        
        result = self.cleaner._standardize_boolean(df, 'discount_applied', rule)
        boolean_values = result['discount_applied'].to_list()
        
        # Should convert to boolean values, check that all non-None values are correct
        # Note: polars may keep None as None in boolean columns
        expected_non_none = [True, False, True, False, True, False]
        actual_non_none = [v for v in boolean_values if v is not None]
        self.assertEqual(actual_non_none, expected_non_none)
        self.assertEqual(result['discount_applied'].dtype, pl.Boolean)
    
    def test_detect_numeric_outliers_iqr(self):
        """Test detect_numeric_outliers rule with IQR method"""
        df = pl.DataFrame({
            'price': [10, 15, 12, 14, 11, 13, 100, 16, 12, 15]  # 100 is outlier
        })
        
        rule = {
            'rule_id': 'detect_numeric_outliers',
            'rule_type': 'outlier_detection',
            'confidence': 0.85,
            'method': 'iqr',
            'threshold': 1.5,
            'action': 'flag'
        }
        
        result = self.cleaner._detect_outliers(df, 'price', rule)
        
        # Should detect 1 outlier
        log_entry = self.cleaner.cleaning_log[-1]
        self.assertIn('1', log_entry['result'])
        self.assertIn('outlier', log_entry['result'].lower())
    
    def test_detect_numeric_outliers_cap(self):
        """Test detect_numeric_outliers rule with capping action"""
        df = pl.DataFrame({
            'amount': [10, 15, 12, 14, 11, 13, 100, 16, 12, 15]  # 100 is outlier
        })
        
        rule = {
            'rule_id': 'detect_numeric_outliers',
            'rule_type': 'outlier_detection',
            'confidence': 0.85,
            'method': 'iqr',
            'threshold': 1.5,
            'action': 'cap'
        }
        
        original_max = df['amount'].max()
        result = self.cleaner._detect_outliers(df, 'amount', rule)
        new_max = result['amount'].max()
        
        # Should cap the outlier
        self.assertLess(new_max, original_max)


class TestDatasetSpecificRules(unittest.TestCase):
    """Test dataset-specific business rules"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
    
    def test_validate_transaction_totals(self):
        """Test validate_transaction_totals rule"""
        df = pl.DataFrame({
            'Price Per Unit': [10.0, 25.0, 15.0, 30.0],
            'Quantity': [2, 3, 4, 2],
            'Total Spent': [20.0, 75.0, 60.0, 59.98]  # Last one has error > tolerance
        })
        
        rule = {
            'rule_id': 'validate_transaction_totals',
            'rule_type': 'validation',
            'confidence': 0.98,
            'tolerance': 0.01,
            'required_fields': ['Total Spent', 'Price Per Unit', 'Quantity']
        }
        
        result = self.cleaner._validate_transaction_totals(df, rule)
        
        # Should fix the incorrect total (59.98 -> 60.00)
        expected_totals = [20.0, 75.0, 60.0, 60.0]
        self.assertEqual(result['Total Spent'].to_list(), expected_totals)
        
        # Should log the correction
        log_entry = self.cleaner.cleaning_log[-1]
        self.assertIn('1', log_entry['result'])  # 1 discrepancy fixed


class TestRuleEngineMatching(unittest.TestCase):
    """Test rule matching logic in RuleEngine"""
    
    def setUp(self):
        self.rule_engine = RuleEngine()
    
    def test_boolean_pattern_detection(self):
        """Test _check_boolean_patterns method"""
        # Test boolean field
        bool_series = pl.Series([True, False, True, None])
        self.assertTrue(self.rule_engine._check_boolean_patterns(bool_series))
        
        # Test string boolean patterns
        string_bool = pl.Series(['true', 'false', 'yes', 'no'])
        self.assertTrue(self.rule_engine._check_boolean_patterns(string_bool))
        
        # Test numeric boolean (0/1)
        numeric_bool = pl.Series([0, 1, 1, 0])
        self.assertTrue(self.rule_engine._check_boolean_patterns(numeric_bool))
        
        # Test non-boolean patterns
        non_bool = pl.Series(['apple', 'banana', 'orange'])
        self.assertFalse(self.rule_engine._check_boolean_patterns(non_bool))
    
    def test_phone_pattern_detection(self):
        """Test _detect_phone_patterns method"""
        # Test phone-like patterns
        phone_series = pl.Series(['123-456-7890', '(123) 456-7890', '1234567890'])
        self.assertTrue(self.rule_engine._detect_phone_patterns(phone_series))
        
        # Test non-phone patterns
        non_phone = pl.Series(['abc', '123', 'hello world'])
        self.assertFalse(self.rule_engine._detect_phone_patterns(non_phone))
    
    def test_email_pattern_detection(self):
        """Test _detect_email_patterns method"""
        # Test email patterns
        email_series = pl.Series(['user@example.com', 'test@domain.org', 'name@site.net'])
        self.assertTrue(self.rule_engine._detect_email_patterns(email_series))
        
        # Test non-email patterns
        non_email = pl.Series(['notanemail', '@domain.com', 'user@'])
        self.assertFalse(self.rule_engine._detect_email_patterns(non_email))
    
    def test_currency_pattern_detection(self):
        """Test _detect_currency_patterns method"""
        # Test string currency patterns
        currency_string = pl.Series(['$10.99', '€25.50', '£15.75'])
        self.assertTrue(self.rule_engine._detect_currency_patterns(currency_string, {}))
        
        # Test float currency patterns (2 decimal places)
        currency_float = pl.Series([10.99, 25.50, 15.75, 99.00])
        self.assertTrue(self.rule_engine._detect_currency_patterns(currency_float, {}))
        
        # Test non-currency patterns
        non_currency = pl.Series(['apple', 'banana', 'orange'])
        self.assertFalse(self.rule_engine._detect_currency_patterns(non_currency, {}))
    
    def test_id_pattern_detection(self):
        """Test _detect_id_patterns method"""
        # Test ID-like field with high uniqueness
        id_series = pl.Series(['ID_001', 'ID_002', 'ID_003', 'ID_004', 'ID_005'])
        self.assertTrue(self.rule_engine._detect_id_patterns('customer_id', id_series))
        
        # Test non-ID field name
        self.assertFalse(self.rule_engine._detect_id_patterns('description', id_series))
        
        # Test low uniqueness
        low_unique = pl.Series(['A', 'A', 'A', 'B', 'B'])
        self.assertFalse(self.rule_engine._detect_id_patterns('transaction_id', low_unique))


class TestRuleConfiguration(unittest.TestCase):
    """Test rule configuration loading and validation"""
    
    def setUp(self):
        self.rule_engine = RuleEngine()
    
    def test_rule_config_structure(self):
        """Test that rule configuration has expected structure"""
        config = self.rule_engine.rules_config
        
        # Should have all required sections
        self.assertIn('universal_rules', config)
        self.assertIn('field_specific_rules', config)
        self.assertIn('dataset_specific_rules', config)
        
        # Universal rules should have required fields
        for rule in config['universal_rules']:
            self.assertIn('rule_id', rule)
            self.assertIn('rule_type', rule)
            self.assertIn('confidence', rule)
            self.assertIn('enabled', rule)
    
    def test_rule_priorities(self):
        """Test that rules have proper priority ordering"""
        config = self.rule_engine.rules_config
        
        # Universal rules should have ascending priorities
        universal_priorities = [rule.get('priority', 999) for rule in config['universal_rules']]
        self.assertEqual(universal_priorities, sorted(universal_priorities))
        
        # Field-specific rules should have priorities
        for rule in config['field_specific_rules']:
            self.assertIn('priority', rule)
            self.assertIsInstance(rule['priority'], int)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete rule application scenarios"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
        self.rule_engine = RuleEngine()
    
    def test_complete_cleaning_pipeline(self):
        """Test complete cleaning pipeline with multiple rules"""
        # Create realistic test data
        df = pl.DataFrame({
            'Transaction ID': ['TXN_001', 'TXN_002', 'TXN_002', 'TXN_003'],  # Has duplicate
            'Customer Name': [' john doe ', 'JANE SMITH', 'bob johnson', None],
            'Price Per Unit': [10.99, None, 25.50, 15.75],
            'Quantity': [2, 3, None, 4],
            'Total Spent': [21.98, 75.00, 51.00, 63.00],
            'Phone': ['123-456-7890', '(555) 123-4567', None, '9876543210'],
            'Discount Applied': ['true', 'False', '1', '0']
        })
        
        # Create matched rules (simplified)
        matched_rules = {
            'Transaction ID': [
                {'rule_id': 'remove_exact_duplicates', 'rule_type': 'deduplication', 'confidence': 1.0},
                {'rule_id': 'validate_ids', 'rule_type': 'validation', 'confidence': 0.9}
            ],
            'Customer Name': [
                {'rule_id': 'trim_whitespace', 'rule_type': 'standardization', 'confidence': 1.0},
                {'rule_id': 'standardize_names', 'rule_type': 'standardization', 'confidence': 0.92},
                {'rule_id': 'handle_missing_categorical', 'rule_type': 'missing_value', 'strategy': 'mode', 'confidence': 0.8}
            ],
            'Price Per Unit': [
                {'rule_id': 'handle_missing_numeric', 'rule_type': 'missing_value', 'strategy': 'median', 'confidence': 0.85},
                {'rule_id': 'normalize_currency', 'rule_type': 'normalization', 'confidence': 0.95, 'decimal_places': 2}
            ],
            'Quantity': [
                {'rule_id': 'handle_missing_numeric', 'rule_type': 'missing_value', 'strategy': 'median', 'confidence': 0.85},
                {'rule_id': 'normalize_quantities', 'rule_type': 'normalization', 'confidence': 0.95}
            ],
            'Total Spent': [
                {'rule_id': 'normalize_currency', 'rule_type': 'normalization', 'confidence': 0.95, 'decimal_places': 2}
            ],
            'Phone': [
                {'rule_id': 'normalize_phone', 'rule_type': 'normalization', 'confidence': 0.94}
            ],
            'Discount Applied': [
                {'rule_id': 'standardize_boolean', 'rule_type': 'standardization', 'confidence': 0.9,
                 'mappings': {'true': True, 'False': False, '1': True, '0': False}}
            ],
            '_dataset_rules': [
                {'rule_id': 'validate_transaction_totals', 'rule_type': 'validation', 'confidence': 0.98,
                 'tolerance': 0.01, 'required_fields': ['Total Spent', 'Price Per Unit', 'Quantity']}
            ]
        }
        
        # Apply cleaning
        result = self.cleaner.clean_dataset(df, matched_rules)
        
        # Verify results
        self.assertLessEqual(len(result), len(df))  # Should remove duplicates
        self.assertEqual(result['Phone'].null_count(), 1)  # One phone should remain null
        self.assertEqual(result['Discount Applied'].dtype, pl.Boolean)  # Should be boolean
        
        # Verify cleaning summary
        summary = self.cleaner.get_cleaning_summary()
        self.assertGreater(summary['operations_performed'], 0)
        self.assertGreaterEqual(summary['total_records_affected'], 0)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUniversalRules,
        TestFieldSpecificRules,
        TestDatasetSpecificRules,
        TestRuleEngineMatching,
        TestRuleConfiguration,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with file logging
    import io
    import sys
    from datetime import datetime
    
    # Capture test output
    test_output = io.StringIO()
    runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
    result = runner.run(suite)
    
    # Get the captured output
    test_log = test_output.getvalue()
    
    # Print summary to console
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    print(f"{'='*60}")
    
    # Save test logs to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_results_{timestamp}.log"
    
    with open(log_filename, 'w') as f:
        f.write(f"Data Cleaner Unit Test Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write("DETAILED TEST OUTPUT:\n")
        f.write(test_log)
        f.write(f"\n{'='*60}\n")
        f.write("SUMMARY:\n")
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%\n")
        f.write(f"{'='*60}\n")
        
        if result.failures:
            f.write("\nFAILURES:\n")
            for test, traceback in result.failures:
                f.write(f"\n{test}:\n{traceback}\n")
        
        if result.errors:
            f.write("\nERRORS:\n")
            for test, traceback in result.errors:
                f.write(f"\n{test}:\n{traceback}\n")
    
    print(f"Test results saved to: {log_filename}")