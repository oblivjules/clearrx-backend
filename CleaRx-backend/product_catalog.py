"""In-memory product matching. Unknown catalog fields never imply agreement."""
import html
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal
from copy import deepcopy
from functools import lru_cache
from rapidfuzz import fuzz, process


def key(value):
    return ' '.join(unicodedata.normalize('NFKC', html.unescape(str(value or ''))).casefold().split())


def strength(value):
    text = key(value).replace('μg', 'mcg').replace('µg', 'mcg')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(mcg|mg|g)(?:\s*(?:/|per)\s*(\d+(?:\.\d+)?)?\s*(ml))?', text)
    if not match:
        return None
    amount, unit, volume, denominator = match.groups()
    mass = Decimal(amount) * {'mcg': Decimal('.001'), 'mg': Decimal(1), 'g': Decimal(1000)}[unit]
    if denominator:
        if Decimal(volume or '1') == 0:
            return None
        return ('mg/ml', mass / Decimal(volume or '1'))
    return ('mg', mass)


def form(value):
    text = key(value)
    # Release formulations remain distinct from ordinary tablets/capsules.
    release = ' modified' if re.search(r'\b(?:xr|sr|er|cr|modified|extended|sustained|controlled|delayed)\b', text) else ''
    for pattern, label in [(r'\b(tab|tabs|tablet|tablets)\b', 'tablet'),
                           (r'\b(cap|caps|capsule|capsules)\b', 'capsule'),
                           (r'\bsusp(?:ension)?\b', 'suspension'),
                           (r'\bsyrup\b', 'syrup')]:
        if re.search(pattern, text):
            return label + release
    return text


def route(value):
    text = key(value)
    return {'po': 'oral', 'iv': 'intravenous', 'im': 'intramuscular',
            'sc': 'subcutaneous', 'sq': 'subcutaneous'}.get(text, text)


class ProductCatalog:
    def __init__(self, products, import_id):
        self.import_id = str(import_id)
        self.products = {str(p['product_id']): p for p in products if not p['is_veterinary']}
        self.name_products = defaultdict(set)
        for pid, p in self.products.items():
            for alias in [p['brand_name'], p['generic_name']]:
                if key(alias) and 'see reverse' not in key(alias):
                    self.name_products[key(alias)].add(pid)
        self.names = sorted(self.name_products)
        self._cached_match = lru_cache(maxsize=2048)(self._match)

    def match(self, name, *, brand_name=None, strength_value=None, dosage_form=None, route_value=None):
        # Callers attach metadata to results; never return mutable cache entries.
        return deepcopy(self._cached_match(name, brand_name, strength_value, dosage_form, route_value))

    def _match(self, name, brand_name, strength_value, dosage_form, route_value):
        query = key(brand_name or name)
        result = dict(original=name, corrected=name, confidence=0.0, matched=False,
                      match_type=None, product_id=None, catalog_import_id=self.import_id,
                      candidates=[], field_validation={}, review_reasons=[])
        if not query:
            return result
        hits = [(query, 100, 0)] if query in self.name_products else process.extract(
            query, self.names, scorer=fuzz.ratio, score_cutoff=90 if len(query) <= 4 else 88, limit=50)
        scores = {}
        for alias, score, _ in hits:
            for pid in self.name_products[alias]:
                scores[pid] = max(scores.get(pid, 0), score)
        ranked = []
        for pid, score in scores.items():
            p = self.products[pid]
            checks = {}
            for field, observed, expected, normalize in [
                ('strength', strength_value, p['dosage_strength'], strength),
                ('dosage_form', dosage_form, p['dosage_form'], form)]:
                a, b = normalize(observed), normalize(expected)
                checks[field] = 'unknown' if not observed or not expected or a is None or b is None else 'match' if a == b else 'conflict'
            checks['route_of_administration'] = ('unknown' if not route_value or not p['routes'] else
                'match' if route(route_value) in {route(r) for r in p['routes']} else 'conflict')
            if brand_name and name and key(name) not in {key(brand_name), key(p['generic_name'])}:
                checks['drug_name'] = 'conflict'
            rank = score + 4 * list(checks.values()).count('match') - 30 * list(checks.values()).count('conflict')
            ranked.append((rank, pid, score, checks))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        result['candidates'] = [dict(product_id=pid, name_score=score, field_validation=checks)
                                for _, pid, score, checks in ranked[:5]]
        if not ranked:
            return result
        _, pid, score, checks = ranked[0]
        p = self.products[pid]
        # Incomplete duplicates must not conceal a contradiction supported by
        # every known formulation for this same name and ingredient.
        peers = [x for x in ranked if x[2] == score and
                 key(self.products[x[1]]['generic_name']) == key(p['generic_name'])]
        for field, status in list(checks.items()):
            known = {x[3].get(field) for x in peers} - {'unknown', None}
            if status == 'unknown' and known == {'conflict'}:
                checks[field] = 'conflict'
        # Name resolution may succeed while formulation resolution remains ambiguous.
        near = [self.products[x[1]] for x in ranked if ranked[0][0] - x[0] < 5]
        resolved_names = {key(x['generic_name'] or x['brand_name']) for x in near}
        if len(resolved_names) != 1:
            result['review_reasons'] = ['ambiguous_drug_name']
            return result
        canonical_name = p['generic_name'] or p['brand_name']
        # An already correct OCR name needs no cosmetic replacement from the CSV.
        display_name = str(name).strip() if key(name) == key(canonical_name) else canonical_name
        # Preserve this explicitly supported OCR spelling; do not strip arbitrary
        # chemical prefixes or change the candidate score to an exact match.
        if key(name) == 'n-acetylcysteine' and key(canonical_name) == 'acetylcysteine':
            display_name = str(name).strip()
        result.update(corrected=display_name, canonical_name=canonical_name, matched=True,
                      confidence=score / 100, raw_score=score, match_type='catalog_exact' if score == 100 else 'catalog_fuzzy',
                      algorithm_used='exact' if score == 100 else 'ratio', field_validation=checks)
        conflicts = [f for f, status in checks.items() if status == 'conflict']
        result['review_reasons'] = [f'{f}_conflict' for f in conflicts]
        if len(near) == 1 and not conflicts and strength_value and checks['strength'] == 'match':
            result['product_id'] = pid
            result['formulation_status'] = 'resolved'
        else:
            # Many registrations can share the same correctly recognized ingredient.
            # Keep product_id unknown without treating that alone as an OCR error.
            result['formulation_status'] = 'unresolved'
        return result
