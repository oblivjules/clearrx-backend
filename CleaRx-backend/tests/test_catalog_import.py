import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import psycopg2

spec = importlib.util.spec_from_file_location(
    'catalog_import', Path(__file__).resolve().parents[1] / 'scripts/import_drug_catalog.py')
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class CatalogImportTests(unittest.TestCase):
    def read_rows(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'catalog.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=catalog.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return catalog.read_catalog(path)

    def test_shared_aliases_and_raw_values_survive(self):
        rows = [dict.fromkeys(catalog.FIELDS, '') for _ in range(2)]
        for i, row in enumerate(rows):
            row.update(drug_id=str(i), brand_name='A &amp; B',
                       generic_name='Ingredient', common_ocr_variants='shared')
        products, report = self.read_rows(rows)
        self.assertEqual(report['errors'], [])
        self.assertIn('shared', products[0][4])
        self.assertIn('shared', products[1][4])
        self.assertEqual(products[0][0]['brand_name'], 'A & B')
        self.assertEqual(products[0][1]['brand_name'], 'A &amp; B')

    def test_duplicate_ids_block_import_and_placeholders_are_not_aliases(self):
        row = dict.fromkeys(catalog.FIELDS, '')
        row.update(drug_id='same', brand_name='Example', generic_name='(see reverse)',
                   common_ocr_variants='(see reverse);example', ph_fda_number='VR-1')
        products, report = self.read_rows([row, row])
        self.assertTrue(report['errors'])
        self.assertNotIn('(see reverse)', products[0][4])
        self.assertTrue(products[0][3])

    def test_normalization_preserves_strength_and_combination(self):
        self.assertEqual(catalog.normalize(' A+B 500 mg/5 mL XR '), 'a+b 500 mg/5 ml xr')

    def test_malformed_columns_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'catalog.csv'
            path.write_text(','.join(catalog.FIELDS) + '\nonly-one-column\n', encoding='utf-8')
            products, report = catalog.read_catalog(path)
        self.assertEqual(products, [])
        self.assertIn('malformed column count', report['errors'][0])

    def test_unexpected_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'catalog.csv'
            path.write_text('unexpected\nvalue\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'CSV headers'):
                catalog.read_catalog(path)

    def import_input(self):
        row = dict.fromkeys(catalog.FIELDS, '')
        row.update(drug_id='one', brand_name='Example', generic_name='Ingredient',
                   common_ocr_variants='shared', route_of_administration='Oral')
        return self.read_rows([row])

    def connection(self, duplicate=False):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None if duplicate else ('import-1',)
        cursor.fetchall.return_value = [('one', 'product-1')]
        return connection, cursor

    def test_import_writes_products_and_aliases_in_one_transaction(self):
        products, report = self.import_input()
        connection, cursor = self.connection()
        with patch.dict(catalog.os.environ, DATABASE_URL='test-only'), \
                patch.object(psycopg2, 'connect', return_value=connection) as connect, \
                patch('psycopg2.extras.execute_values') as insert:
            result = catalog.apply_import(products, report)
        connect.assert_called_once_with('test-only', sslmode='require')
        self.assertIn('Imported 1 records', result)
        self.assertEqual(insert.call_count, 2)
        self.assertIs(insert.call_args_list[0].args[0], cursor)
        self.assertEqual(insert.call_args_list[0].args[2][0][0], 'import-1')
        self.assertIn(('product-1', 'shared'), insert.call_args_list[1].args[2])
        connection.__exit__.assert_called_once_with(None, None, None)

    def test_identical_source_skips_product_and_alias_writes(self):
        products, report = self.import_input()
        connection, _ = self.connection(duplicate=True)
        with patch.dict(catalog.os.environ, DATABASE_URL='test-only'), \
                patch.object(psycopg2, 'connect', return_value=connection), \
                patch('psycopg2.extras.execute_values') as insert:
            result = catalog.apply_import(products, report)
        self.assertIn('no changes made', result)
        insert.assert_not_called()

    def test_alias_failure_exits_transaction_with_error(self):
        products, report = self.import_input()
        connection, _ = self.connection()
        error = RuntimeError('alias insertion failed')
        with patch.dict(catalog.os.environ, DATABASE_URL='test-only'), \
                patch.object(psycopg2, 'connect', return_value=connection), \
                patch('psycopg2.extras.execute_values', side_effect=[None, error]):
            with self.assertRaisesRegex(RuntimeError, 'alias insertion failed'):
                catalog.apply_import(products, report)
        # psycopg2 rolls back when the connection context receives an exception.
        self.assertIs(connection.__exit__.call_args.args[0], RuntimeError)
        self.assertIs(connection.__exit__.call_args.args[1], error)

    def test_validation_errors_cannot_reach_database(self):
        products, report = self.import_input()
        report['errors'] = ['invalid source']
        with patch.object(psycopg2, 'connect') as connect:
            with self.assertRaisesRegex(ValueError, 'Validation failed'):
                catalog.apply_import(products, report)
        connect.assert_not_called()

    def test_missing_database_url_cannot_connect(self):
        products, report = self.import_input()
        with patch.dict(catalog.os.environ, {}, clear=True), \
                patch.object(psycopg2, 'connect') as connect:
            with self.assertRaisesRegex(ValueError, 'DATABASE_URL'):
                catalog.apply_import(products, report)
        connect.assert_not_called()

    def test_cli_dry_run_does_not_connect(self):
        row = dict.fromkeys(catalog.FIELDS, '')
        row.update(drug_id='one', brand_name='Example')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'catalog.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=catalog.FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with patch('sys.argv', ['import_drug_catalog.py', str(path)]), \
                    patch.object(psycopg2, 'connect') as connect, patch('builtins.print') as output:
                catalog.main()
        connect.assert_not_called()
        self.assertIn('Dry run only', output.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
