import pytest

from drug_validator import DrugValidator
from parser import parse_prescription, refresh_drug_catalog
from product_catalog import ProductCatalog


@pytest.fixture
def numeric_brand_catalog():
    # Preserve runtime indexes so fixture setup cannot change other tests.
    original = {name: value for name, value in vars(DrugValidator).items()
                if name.isupper() or name == '_product_catalog'}
    product = dict(product_id='p', brand_name='Brumox-500',
                   generic_name='amoxicillin (as trihydrate)', dosage_strength='500mg',
                   dosage_form='Capsule', routes=['Oral'], is_veterinary=False)
    catalog = ProductCatalog([product], 'v')
    DrugValidator.use_product_catalog(catalog)
    refresh_drug_catalog()
    try:
        yield catalog
    finally:
        for name, value in original.items():
            setattr(DrugValidator, name, value)
        refresh_drug_catalog()


@pytest.mark.parametrize('text', ['Brumox-500', 'Brumox-500 500mg capsule'])
def test_numeric_brand_survives_full_parser(numeric_brand_catalog, text):
    medicine = parse_prescription(text)[0]
    assert medicine['brand_name'] == 'Brumox-500'
    assert medicine['drug_name'] == 'amoxicillin (as trihydrate)'
    assert numeric_brand_catalog.match(medicine['brand_name'])['matched']
