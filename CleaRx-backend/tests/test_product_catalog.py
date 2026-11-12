import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product_catalog import ProductCatalog, strength


def product(pid, amount, **kwargs):
    return dict(product_id=pid, brand_name='Example', generic_name='Ingredient',
                dosage_strength=amount, dosage_form='Tablet', routes=['Oral'],
                is_veterinary=False, aliases=['unrelated-ocr-alias'], **kwargs)


class ProductMatchingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ProductCatalog([product('a', '500mg'), product('b', '250mg')], 'version')

    def test_equivalent_strength_resolves_product(self):
        result = self.catalog.match('Example', strength_value='0.5 g', dosage_form='tab', route_value='PO')
        self.assertEqual(result['product_id'], 'a')
        self.assertEqual(result['review_reasons'], [])

    def test_name_only_is_not_a_product(self):
        result = self.catalog.match('Example')
        self.assertTrue(result['matched'])
        self.assertIsNone(result['product_id'])
        self.assertEqual(result['review_reasons'], [])
        self.assertEqual(result['formulation_status'], 'unresolved')

    def test_correct_ocr_capitalization_is_preserved(self):
        first = product('a', '500mg')
        first['generic_name'] = 'azithromycin'
        second = dict(first, product_id='b', dosage_strength='250mg')
        catalog = ProductCatalog([first, second], 'v')
        result = catalog.match('Azithromycin')
        self.assertTrue(result['matched'])
        self.assertEqual(result['corrected'], 'Azithromycin')
        self.assertEqual(result['confidence'], 1.0)
        self.assertEqual(result['review_reasons'], [])
        self.assertIsNone(result['product_id'])

    def test_strength_conflict_preserves_name_but_flags(self):
        result = self.catalog.match('Example', strength_value='750mg')
        self.assertIsNone(result['product_id'])
        self.assertIn('strength_conflict', result['review_reasons'])

    def test_n_acetylcysteine_display_preserves_ocr_spelling(self):
        entry = product('a', '600mg')
        entry['generic_name'] = 'acetylcysteine'
        result = ProductCatalog([entry], 'v').match('N-Acetylcysteine')
        self.assertTrue(result['matched'])
        self.assertEqual(result['corrected'], 'N-Acetylcysteine')
        self.assertEqual(result['canonical_name'], 'acetylcysteine')
        self.assertEqual(result['match_type'], 'catalog_fuzzy')
        self.assertLess(result['confidence'], 1)

    def test_concentrations_and_amounts_remain_distinct(self):
        self.assertEqual(strength('125mg/5ml'), strength('25 mg/ml'))
        self.assertNotEqual(strength('125mg/5ml'), strength('125mg/ml'))
        self.assertNotEqual(strength('125mg'), strength('125mg/ml'))

    def test_ignores_ocr_alias_but_matches_name_typo(self):
        self.assertFalse(self.catalog.match('unrelated-ocr-alias')['matched'])
        self.assertTrue(self.catalog.match('Exampl')['matched'])
        self.assertFalse(self.catalog.match('xyz')['matched'])

    def test_ambiguous_ingredients_are_not_corrected(self):
        other = product('c', '500mg')
        other['generic_name'] = 'Different ingredient'
        catalog = ProductCatalog([product('a', '500mg'), other], 'v')
        self.assertFalse(catalog.match('Example')['matched'])

    def test_incomplete_record_does_not_hide_known_strength_conflict(self):
        incomplete = product('missing', '')
        catalog = ProductCatalog([product('a', '500mg'), incomplete], 'v')
        result = catalog.match('Example', strength_value='750mg')
        self.assertIn('strength_conflict', result['review_reasons'])
        self.assertIsNone(result['product_id'])

    def test_cache_results_cannot_be_mutated_by_callers(self):
        first = self.catalog.match('Example')
        first['review_reasons'].append('caller changed this')
        self.assertEqual(self.catalog.match('Example')['review_reasons'], [])

    def test_veterinary_exclusion_and_route_conflict(self):
        vet = product('v', '500mg')
        vet['is_veterinary'] = True
        self.assertFalse(ProductCatalog([vet], 'v').match('Example')['matched'])
        result = self.catalog.match('Example', strength_value='500mg', route_value='IV')
        self.assertIsNone(result['product_id'])
        self.assertIn('route_of_administration_conflict', result['review_reasons'])


if __name__ == '__main__':
    unittest.main()
