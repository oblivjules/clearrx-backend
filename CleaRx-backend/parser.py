from __future__ import annotations

import re
import logging
from rapidfuzz import fuzz, process

from drug_validator import DrugValidator as _DrugValidator

logger = logging.getLogger(__name__)

# Access lexicon data through the class — decouples parser from drug_validator internals.
_DV_DRUG_LEXICON     = _DrugValidator._DRUG_LEXICON
_DV_LEXICON_LOWER    = _DrugValidator._LEXICON_LOWER
_DV_BRAND_TO_GENERIC = _DrugValidator._BRAND_TO_GENERIC


class PrescriptionParser:

    _DRUG_CLASS_WORDS = frozenset({
        "antibiotic", "antibiotics", "antibacterial", "antiviral",
        "antifungal", "analgesic", "analgesics", "antipyretic",
        "antihistamine", "antacid", "anti-inflammatory", "painkiller",
        "pain reliever", "supplement", "supplements", "medicine",
        "medication", "drug",
    })

    @classmethod
    def refresh_drug_catalog(cls) -> None:
        """Refresh parser indexes after the validator loads a database catalog."""
        cls._DRUG_LEXICON = _DrugValidator._DRUG_LEXICON
        cls._LEXICON_LOWER = _DrugValidator._LEXICON_LOWER
        cls._BRAND_TO_GENERIC = _DrugValidator._BRAND_TO_GENERIC
        cls._LEXICON_LOWER_SET = frozenset(_DrugValidator._LEXICON_LOWER)
        cls._COMBINATION_LOOKUP = _DrugValidator._COMBINATION_LOOKUP

    # ── Non-drug token blocklist ───────────────────────────────────────────────────
    _NON_DRUG_TOKENS: frozenset[str] = frozenset({
        # Frequencies / timing abbreviations
        'once', 'twice', 'thrice', 'daily', 'weekly', 'monthly', 'nightly', 'hourly',
        'morning', 'evening', 'night', 'bedtime', 'noon', 'afternoon',
        'bid', 'tid', 'qid', 'qd', 'od', 'bd', 'tds', 'qds', 'hs', 'qhs', 'prn',
        'stat', 'mane', 'nocte', 'every', 'biweekly',
        'eod', 'tiw', 'biw', 'qwk', 'qam', 'qpm', 'qn', 'sos', 'adlib',
        # Numeric / time units
        'times', 'time', 'days', 'day', 'weeks', 'week', 'months', 'month', 'hours', 'hour',
        # Number words
        'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'half', 'one', 'two',
        # Dose forms
        'tablet', 'tablets', 'tab', 'tabs', 'capsule', 'capsules', 'cap', 'caps',
        'syrup', 'suspension', 'susp', 'injection', 'inj', 'solution',
        'cream', 'ointment', 'gel', 'patch', 'patches', 'drops', 'drop',
        'vial', 'ampoule', 'amp', 'spray', 'inhaler', 'lozenge', 'sachet',
        'liquid', 'powder', 'suppository',
        'gtt', 'gtts', 'ung', 'elix', 'sol', 'soln', 'supp', 'lot', 'oph', 'ophth',
        # Routes
        'oral', 'orally', 'topical', 'topically', 'sublingual', 'sublingually',
        'intravenous', 'intramuscular', 'subcutaneous', 'rectal', 'nasal',
        'ophthalmic', 'otic', 'inhalation', 'transdermal', 'vaginal',
        'po', 'iv', 'im', 'sc', 'sq', 'sl', 'pr', 'pv', 'ivtt', 'ngt',
        # Instruction verbs
        'take', 'apply', 'use', 'give', 'instill', 'inject', 'dissolve',
        'swallow', 'chew', 'crush', 'dilute', 'administer', 'rinse', 'gargle',
        # Meal/food context
        'meals', 'meal', 'food', 'eating', 'stomach',
        # Administrative
        'sig', 'rx', 'dispense', 'disp', 'mitte', 'mitt', 'refill', 'quantity', 'qty', 'dose',
        'directions', 'instructions',
        # Common connectors / qualifiers
        'with', 'without', 'after', 'before', 'during', 'needed', 'until',
        'finished', 'finish', 'needed', 'immediately', 'now',
    })

    _SIG_VOCAB_RE = re.compile(
        r'\b(?:tab(?:let)?s?|cap(?:sule)?s?|one|two|three|four|1|2|bid|tid|qid|'
        r'daily|once|twice|a\s+day|after|before|glass|water|with|at\s+bedtime)\b',
        re.IGNORECASE,
    )
    _CONTINUATION_START_RE = re.compile(
        r'^\s*(?:sig|sg|s|disp|qty|quantity|#|take|give|apply|use|'
        r'instill|inject|dissolve|swallow|chew|administer|rinse|gargle)\b',
        re.IGNORECASE,
    )

    # Pharmaceutical suffix patterns
    _DRUG_SUFFIX_RE = re.compile(
        r'(?:cillin|mycin|cycline|floxacin|oxacin|azole|statin|sartan|pril|'
        r'olol|dipine|tidine|prazole|mab|nib|zumab|tinib|vir|navir|'
        r'lukast|setron|triptan|oxetine|pamine|zepam|xam|zosin|gliptin|'
        r'gliflozin|glutide|tide|pamide|thiazide|lactam|penem|cycline|'
        r'icillin|axone|triaxone|ximab|zumab|fenac|profen|dronate|calciferol)$',
        re.IGNORECASE,
    )

    # ── Normalization regex ────────────────────────────────────────────────────────
    _NEWLINE_RE     = re.compile(r'[\r\n\t]+')
    _MULTI_SPACE_RE = re.compile(r' {2,}')
    _OCR_NOISE_RE   = re.compile(r'[|\\^~`_]+')
    _DIGIT_O_RE     = re.compile(r'(?<=[A-Za-z])0(?=[A-Za-z])')
    _DIGIT_I_RE     = re.compile(r'(?<=[A-Za-z])1(?=[A-Za-z])')

    # ── % vs × disambiguation ──────────────────────────────────────────────────────
    _PCT_CONC_RE = re.compile(
        r'\b(\d+(?:\.\d+)?)\s*%(?=[ \t]{0,5}'
        r'(?:solution|cream|gel|ointment|lotion|drops?|liquid|suspension|syrup|'
        r'emulsion|w/v|v/v|w/w|topical|eye|nasal|patch|spray)\b)',
        re.IGNORECASE,
    )
    _PCT_NXM_RE  = re.compile(r'(\d)\s*%\s*(\d)')
    _PCT_FREQ_RE = re.compile(
        r'(\d)\s*%(?=\s*(?:daily|day|days|week|wk|month|mo|hour|hr|'
        r'tab(?:let)?s?|cap(?:sule)?s?|times?)\b)',
        re.IGNORECASE,
    )
    _PCT_TRAIL_RE = re.compile(r'(?<=\d)[ \t]*%(?![ \t]*(?:\d|solution|cream|gel|ointment|lotion|drop|w/))')
    _PCT_LEAD_RE  = re.compile(r'(?<!\d)%[ \t]*(?=\d)')
    _PCT_ABBRV_RE = re.compile(
        r'(?<!\d)%(?=\s*(?:q\.?i\.?d\.?|t\.?i\.?d\.?|b\.?i\.?d\.?|o\.?d\.?|q\.?d\.?'
        r'|bd|tds|qds|prn|stat|mane|nocte|om)\b)',
        re.IGNORECASE,
    )

    # ── Philippine N×M dosing shorthand ───────────────────────────────────────────
    _NXM_RE   = re.compile(r'\b([1-4])\s*[xX]\s*([1-4])\b')
    _NXM_FREQ = {1: 'once daily', 2: 'twice daily', 3: 'three times daily', 4: 'four times daily'}

    # ── Meal / food relation ──────────────────────────────────────────────────────
    _MEAL_MAP: list[tuple[re.Pattern, str]] = [
        (re.compile(
            r'\b(?:with\s+(?:food|meals?)|c\.?c\.?|cum\s+cibo'
            r'|take\s+with\s+(?:food|meals?))\b', re.IGNORECASE,
        ), 'with meals'),
        (re.compile(
            r'\b(?:after\s+(?:meals?|eating|food|breakfast|lunch|dinner|supper)|p\.?c\.?|post\s+cibum)\b',
            re.IGNORECASE,
        ), 'after meals'),
        (re.compile(
            r'\b(?:before\s+(?:meals?|eating|food)|a\.?c\.?|ante\s+cibum'
            r'|on\s+(?:an\s+)?empty\s+stomach)\b', re.IGNORECASE,
        ), 'before meals'),
    ]

    # ── Dosage (strength) ─────────────────────────────────────────────────────────
    _DOSAGE_RE = re.compile(
        r'\b(\d+(?:[.,]\d+)?)\s*(mg|mcg|ug\b|g\b|ml|iu|units?|%|meq)'
        r'(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|g|ml|iu|units?|%|meq))?\b',
        re.IGNORECASE,
    )
    _UNIT_ALIAS = {'ug': 'mcg', 'meq': 'mEq', 'unit': 'unit', 'units': 'units'}

    # ── Duration ──────────────────────────────────────────────────────────────────
    _DURATION_RE = re.compile(
        r'\b(?:[x\xd7]|for)?\s*(\d+)\s*(day|d(?![\w.])|week|wk|month|mo)s?\b',
        re.IGNORECASE,
    )
    _DURATION_UNTIL_RE = re.compile(
        r'\buntil\s+(?:finish(?:ed)?|gone|complet(?:ed?)?|all\s+(?:taken|used))\b',
        re.IGNORECASE,
    )

    # ── Frequency ─────────────────────────────────────────────────────────────────
    _FREQ_MAP: list[tuple[re.Pattern, str | None]] = [
        (re.compile(r'\bQ[Il1]D\b', re.IGNORECASE), 'four times daily'),
        (re.compile(r'\bT[Il1]D\b', re.IGNORECASE), 'three times daily'),
        (re.compile(r'\bB[Il1]D\b', re.IGNORECASE), 'twice daily'),
        (re.compile(r'\b0[Dd]\b'),                   'once daily'),
        (re.compile(r'\b(four\s+times?\s+(?:a\s+)?day|q\.?i\.?d\.?|qid|qds|4x\s*daily)\b', re.IGNORECASE), 'four times daily'),
        (re.compile(r'\b4\s*[xX×%]\s*(?:(?:a\s+)?(?:day|daily))\b', re.IGNORECASE), 'four times daily'),
        (re.compile(r'\b4\s+times?\s+(?:a\s+)?(?:day|daily)\b', re.IGNORECASE), 'four times daily'),
        (re.compile(r'\b(three\s+times?\s+(?:a\s+)?day|thrice\s+(?:a\s+)?day|t\.?i\.?d\.?|tid|tds|3x\s*daily)\b', re.IGNORECASE), 'three times daily'),
        (re.compile(r'\b3\s*[xX×%]\s*(?:(?:a\s+)?(?:day|daily))\b', re.IGNORECASE), 'three times daily'),
        (re.compile(r'\b3\s+times?\s+(?:a\s+)?(?:day|daily)\b', re.IGNORECASE), 'three times daily'),
        (re.compile(r'\b(twice\s+(?:a\s+)?daily|twice\s+(?:a\s+)?day|b\.?i\.?d\.?|bid|bd|2x\s*daily)\b', re.IGNORECASE), 'twice daily'),
        (re.compile(r'\b2\s*[xX×%]\s*(?:(?:a\s+)?(?:day|daily))\b', re.IGNORECASE), 'twice daily'),
        (re.compile(r'\b2\s+times?\s+(?:a\s+)?(?:day|daily)\b', re.IGNORECASE), 'twice daily'),
        (re.compile(r'\b(once\s+(?:a\s+)?daily|once\s+(?:a\s+)?day|[o0]\.?d\.?|q\.?d\.?|qd|1x\s*daily)(?!\w)', re.IGNORECASE), 'once daily'),
        (re.compile(r'\b1\s*[xX×%]\s*(?:(?:a\s+)?(?:day|daily))\b', re.IGNORECASE), 'once daily'),
        (re.compile(r'\b1\s+times?\s+(?:a\s+)?(?:day|daily)\b', re.IGNORECASE), 'once daily'),
        (re.compile(r'\b(?:every|q\.?)\s*(\d+)\s*(?:h(?:rs?|ours?)?)?(?!\s*(?:day|week|wk|month|mo)s?\b)\b', re.IGNORECASE), None),
        (re.compile(r'\b(at\s+bedtime|h\.?s\.?|qhs|before\s+bed|at\s+night|nightly|nocte|omni\s+nocte|q\.?n\.?)(?!\w)', re.IGNORECASE), 'at bedtime'),
        (re.compile(r'\b(as\s+needed|p\.?r\.?n\.?|prn|ad\s+lib(?:itum)?)(?!\w)', re.IGNORECASE), 'as needed'),
        (re.compile(r'\b(every\s+other\s+day|q\.?[o0]\.?d\.?|q[o0]d|e\.?o\.?d\.?)(?!\w)', re.IGNORECASE), 'every other day'),
        (re.compile(r'\b(three\s+times\s+(?:a\s+)?week(?:ly)?|t\.?i\.?w\.?)(?!\w)', re.IGNORECASE), 'three times weekly'),
        (re.compile(r'\b(twice\s+(?:a\s+)?week(?:ly)?|biweekly|b\.?i\.?w\.?)(?!\w)', re.IGNORECASE), 'twice weekly'),
        (re.compile(r'\b(once\s+(?:a\s+)?week(?:ly)?|weekly|q\.?wk\.?)(?!\w)', re.IGNORECASE), 'once weekly'),
        (re.compile(r'\b(mane|omni\s+mane|om|q\.?a\.?m\.?)(?!\w)', re.IGNORECASE), 'once daily (morning)'),
        (re.compile(r'\b(omni\s+nocte|q\.?p\.?m\.?)(?!\w)', re.IGNORECASE), 'once daily (evening)'),
        (re.compile(r'\bstat\b', re.IGNORECASE), 'immediately'),
        (re.compile(r'\b4\s*[xX×%]\b'), 'four times daily'),
        (re.compile(r'\b3\s*[xX×%]\b'), 'three times daily'),
        (re.compile(r'\b2\s*[xX×%]\b'), 'twice daily'),
        (re.compile(r'\b1\s*[xX×%]\b'), 'once daily'),
        (re.compile(r'\b(daily)\b', re.IGNORECASE), 'once daily'),
    ]

    # ── Route of administration ───────────────────────────────────────────────────
    _ROUTE_MAP: list[tuple[re.Pattern, str]] = [
        (re.compile(r'\b(?:by\s+mouth|oral(?:ly)?|p\.?[o0]\.?|p[o0])(?!\w)', re.IGNORECASE),             'oral'),
        (re.compile(r'\b(?:subcut(?:aneous(?:ly)?)?|s\.?c\.?|sq|subq)(?!\w)', re.IGNORECASE),             'subcutaneous'),
        (re.compile(r'\b(?:intramuscular(?:ly)?|i\.?m\.?)(?!\w)', re.IGNORECASE),                         'intramuscular'),
        (re.compile(r'\b(?:intravenous(?:ly)?|i\.?v\.?)(?!\w)', re.IGNORECASE),                           'intravenous'),
        (re.compile(r'\b(?:sublingual(?:ly)?|under\s+(?:the\s+)?tongue|s\.?l\.?)(?!\w)', re.IGNORECASE),  'sublingual'),
        (re.compile(r'\b(?:inhal(?:e(?:d)?|ation)|via\s+inhaler|puffs?|inh)(?!\w)', re.IGNORECASE),       'inhalation'),
        (re.compile(r'\b(?:ophthalmic|eye\s+drops?|instill(?:ed)?|oculus)(?!\w)', re.IGNORECASE),         'ophthalmic'),
        (re.compile(r'\b(?:otic|ear\s+drops?)(?!\w)', re.IGNORECASE),                                     'otic'),
        (re.compile(r'\b(?:rectal(?:ly)?|per\s+rectum|p\.?r\.?)(?!\w)', re.IGNORECASE),                   'rectal'),
        (re.compile(r'\b(?:transdermal|skin\s+patch)(?!\w)', re.IGNORECASE),                              'transdermal'),
        (re.compile(r'\b(?:nasal(?:ly)?|intranasal(?:ly)?|nose\s+drops?)(?!\w)', re.IGNORECASE),          'nasal'),
        (re.compile(r'\b(?:topical(?:ly)?|apply\s+to|applied\s+to|rub\s+on)(?!\w)', re.IGNORECASE),       'topical'),
        (re.compile(r'\b(?:patch(?:es)?)(?!\w)', re.IGNORECASE),                                          'transdermal'),
        (re.compile(r'\b(?:rinse|mouth\s*(?:rinse|wash)|gargle|mouthwash|oral\s+rinse)\b', re.IGNORECASE), 'mouth rinse'),
        (re.compile(r'\b(?:vaginal(?:ly)?|intravaginal(?:ly)?|p\.?v\.?|per\s+vaginam)(?!\w)', re.IGNORECASE), 'vaginal'),
        (re.compile(r'\b(?:i\.?v\.?t\.?t\.?|intravenous(?:ly)?\s+(?:through\s+)?tubing)(?!\w)', re.IGNORECASE), 'intravenous'),
        (re.compile(r'\b(?:n\.?g\.?t\.?|nasogastric(?:\s+tube)?)(?!\w)', re.IGNORECASE),         'nasogastric'),
    ]

    # ── Drug name stripping ───────────────────────────────────────────────────────
    _DRUG_STRIP_RE = re.compile(
        r'\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|g|ml|iu|units?|%|meq)\b'
        r'|\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|susp(?:ension)?|inj(?:ection)?'
        r'|amp(?:oule)?s?|vial|drops?|gtt[s]?|patch(?:es)?|cream|ointment|ung'
        r'|gel|solution|sol(?:n)?|elix(?:ir)?|lot(?:ion)?|powder|sachet|lozenge'
        r'|spray|inhaler|liquid|supp(?:ository)?|gabigabi)\b'
        r'|\b(?:once|twice|thrice|daily|weekly|bedtime|needed|every|hours?'
        r'|bid|tid|qid|prn|qd|od|hs|qhs|qod|eod|tiw|biw|qwk|qam|qpm|qn'
        r'|po|sc|sq|im|iv|ivtt|sl|pr|pv'
        r'|oral(?:ly)?|topical(?:ly)?|sublingual(?:ly)?|intravenous(?:ly)?'
        r'|intramuscular(?:ly)?|subcutaneous(?:ly)?|transdermal|nasal|rectal'
        r'|vaginal(?:ly)?|ophthalmic|otic|inhalation|by\s+mouth|take|apply'
        r'|use|instill|puff[s]?|day[s]?|night|morning|evening|afternoon'
        r'|with\s+meals?|with\s+food|times?|tablet[s]?|capsule[s]?)\b'
        r'|\b(?:rx|sig|mitte|mitt|disp(?:ense)?|refill[s]?|quantity|qty|no\.|#\d*)\b'
        r'|\b(?:directions?|instructions?|take|give|rinse|gargle|ad\s+lib(?:itum)?)\b'
    )
    _CLEAN_PUNCT_RE = re.compile(r'[,;:()/\[\]{}.]+')

    # ── Drug lexicon (sourced from drug_validator) ────────────────────────────────
    _DRUG_LEXICON    = _DV_DRUG_LEXICON
    _LEXICON_LOWER   = _DV_LEXICON_LOWER
    _BRAND_TO_GENERIC = _DV_BRAND_TO_GENERIC
    _LEXICON_LOWER_SET: frozenset[str] = frozenset(_DV_LEXICON_LOWER)
    _COMBINATION_LOOKUP = _DrugValidator._COMBINATION_LOOKUP

    _DRUG_DETECT_THRESHOLD   = 73
    _SPELL_CORRECT_THRESHOLD = 82

    # ── Frequency → daily dose count (used for dispense estimation) ───────────────
    _FREQ_DAILY_DOSES: dict[str, int] = {
        'once daily': 1,
        'once daily (morning)': 1,
        'twice daily': 2,
        'three times daily': 3,
        'four times daily': 4,
        'at bedtime': 1,
    }

    # ── Methods ───────────────────────────────────────────────────────────────────

    def _is_non_drug_candidate(self, candidate: str) -> bool:
        tokens = candidate.lower().split()
        return bool(tokens) and all(t in self._NON_DRUG_TOKENS for t in tokens)

    def _is_drug_class(self, candidate: str | None) -> bool:
        return bool(candidate) and candidate.strip().lower() in self._DRUG_CLASS_WORDS

    def _drug_name_passes_sanity(self, name: str) -> bool:
        if not name or len(name.strip()) < 3:
            return False
        stripped = name.strip()
        if self._is_drug_class(stripped):
            return False
        if not re.search(r'[A-Za-z]', stripped):
            return False
        if self._is_non_drug_candidate(stripped):
            return False
        if self._extract_frequency(stripped) is not None:
            return False
        if self._extract_route(stripped) is not None:
            return False
        return True

    def _resolve_percent_symbol(self, text: str) -> str:
        protected: list[str] = []

        def _guard(m: re.Match) -> str:
            protected.append(m.group(0))
            return f'\x00G{len(protected) - 1}\x00'

        text = self._PCT_CONC_RE.sub(_guard, text)
        text = self._PCT_NXM_RE.sub(r'\1x\2', text)
        text = self._PCT_FREQ_RE.sub(r'\1x', text)
        text = self._PCT_TRAIL_RE.sub('x', text)
        text = self._PCT_LEAD_RE.sub('x ', text)
        text = self._PCT_ABBRV_RE.sub('x', text)

        for i, val in enumerate(protected):
            text = text.replace(f'\x00G{i}\x00', val)

        return text

    def _parse_nxm(self, text: str) -> tuple[str | None, str | None]:
        m = self._NXM_RE.search(text)
        if not m:
            return None, None
        n, freq_n = int(m.group(1)), int(m.group(2))
        qty_str  = f"{n} tablet{'s' if n > 1 else ''}"
        freq_str = self._NXM_FREQ.get(freq_n, f"{freq_n} times daily")
        return qty_str, freq_str

    def _extract_meal_relation(self, text: str) -> str | None:
        for pattern, label in self._MEAL_MAP:
            if pattern.search(text):
                return label
        return None

    def _followed_by_uncertain(
        self,
        sig_text: str,
        matched_text: str | re.Match[str] | None,
    ) -> bool:
        if isinstance(matched_text, re.Match):
            if "[UNCERTAIN]" in matched_text.group(0):
                return True
            end = matched_text.end()
            return re.match(r'\s*\[UNCERTAIN\]', sig_text[end:]) is not None
        if not matched_text:
            return False
        match = re.search(re.escape(matched_text), sig_text)
        if not match:
            return False
        return re.match(r'\s*\[UNCERTAIN\]', sig_text[match.end():]) is not None

    def _field_source_match(
        self,
        text: str,
        field_name: str,
        source_value: str | None = None,
    ) -> re.Match[str] | None:
        patterns = {
            "strength": self._DOSAGE_RE,
            "dosage_form": re.compile(
                r'\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|susp(?:ension)?|'
                r'solution|soln|suppository|supp|injection|inj|patch(?:es)?|'
                r'spray|inhaler|lozenge|sachet|powder|lotion|ointment|cream|'
                r'gel|drops?)\b', re.IGNORECASE,
            ),
            "dose_amount": re.compile(
                r'\b[1-9]\d*(?:/\d+)?\s*(?:tab(?:let)?s?|cap(?:sule)?s?|'
                r'drop[s]?|puff[s]?)\b', re.IGNORECASE,
            ),
            "frequency": re.compile(
                r'\b(?:once\s+(?:a\s+)?(?:day|daily)|twice\s+(?:a\s+)?'
                r'(?:day|daily)|three\s+(?:times?\s+)?(?:a\s+)?(?:day|daily)|'
                r'four\s+times?\s+(?:a\s+)?(?:day|daily)|daily|bid|tid|qid|'
                r'qd|od|bd|3x|2x|1x|every\s+\d+\s*(?:h|hr|hour|hours)?)\b',
                re.IGNORECASE,
            ),
            "timing": re.compile(
                r'\b(?:before|after|with)\s+[A-Za-z]+', re.IGNORECASE,
            ),
            "administration_instructions": re.compile(
                r'\b(?:dissolve|rinse|gargle|apply|instill|swallow|chew|'
                r'drink|take)\b[^,;.]*(?:glass\s+of\s+water)?',
                re.IGNORECASE,
            ),
            "route_of_administration": None,
            "quantity": re.compile(
                r'(?:#|qty\.?|quantity\s*:?)\s*\d+\b', re.IGNORECASE,
            ),
            "duration": re.compile(
                r'\b(?:until\s+(?:finish(?:ed)?|gone|complet(?:ed?)?|'
                r'all\s+(?:taken|used))|\d+\s*(?:day|d|week|wk|month|mo)s?)\b',
                re.IGNORECASE,
            ),
            "brand_name": re.compile(re.escape(source_value), re.IGNORECASE)
            if source_value else None,
        }
        pattern = patterns.get(field_name)
        if field_name == "route_of_administration":
            for route_pattern, _ in self._ROUTE_MAP:
                match = route_pattern.search(text)
                if match:
                    return match
            return None
        if pattern is None:
            return None
        return pattern.search(text)

    def _normalize(self, text: str) -> str:
        text = self._NEWLINE_RE.sub(' ', text)
        text = self._OCR_NOISE_RE.sub(' ', text)
        text = re.sub(r'(?<=\s)/(?=\s)', ' ', text)
        text = self._resolve_percent_symbol(text)
        text = self._DIGIT_O_RE.sub('o', text)
        text = self._DIGIT_I_RE.sub('i', text)

        # Unit suffix fixes
        text = re.sub(r'(?<=[0-9])\s*rng\b',  'mg',  text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[0-9])\s*rnl\b',  'ml',  text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[0-9])\s*rncg\b', 'mcg', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[0-9])\s+m\s*g\b',  'mg',  text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[0-9])\s+m\s*l\b',  'ml',  text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[0-9])\s+mc\s*g\b', 'mcg', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[0-9])\s+m\s*cg\b', 'mcg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d) ?M\b(?![a-zA-Z])', r'\1mg', text)
        text = re.sub(r'\b(mcg|mg|ml)\s*(?=[A-Za-z])', r'\1 ', text, flags=re.IGNORECASE)

        # Letter-as-digit fixes
        text = re.sub(r'\bl(\s*(?:tab|cap|tsp|tbsp|ml)\b)', r'1\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=[A-Za-z])vv(?=[A-Za-z])', 'w', text, flags=re.IGNORECASE)
        text = re.sub(r'\bIII\s+(tab|cap|tablet|capsule)\b', r'3 \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bII\s+(tab|cap|tablet|capsule)\b',  r'2 \1', text, flags=re.IGNORECASE)

        # Word-form quantities → numerals
        text = re.sub(r'\bhalf\s+(tab|cap|tablet|capsule|tsp|tbsp)\b', r'1/2 \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:one|a)\s+(tab|cap|tablet|capsule|tsp|tbsp|drop|puff|spray|ml)\b', r'1 \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\btwo\s+(tab|cap|tablet|capsule|tsp|tbsp|drop|puff|spray|ml)\b',       r'2 \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bthree\s+(tab|cap|tablet|capsule|tsp|tbsp|drop|puff|spray|ml)\b',     r'3 \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfour\s+(tab|cap|tablet|capsule|tsp|tbsp|drop|puff|spray|ml)\b',      r'4 \1', text, flags=re.IGNORECASE)

        # Dose form abbreviation normalisation
        text = re.sub(r'\bgtts?\b', 'drops',   text, flags=re.IGNORECASE)
        text = re.sub(r'\bung\b',   'ointment', text, flags=re.IGNORECASE)
        text = re.sub(r'\belix\b',  'elixir',   text, flags=re.IGNORECASE)

        # Route abbreviation normalisation
        text = re.sub(r'\bI\.?V\.?T\.?T\.?\b', 'IV', text, flags=re.IGNORECASE)

        # British / Philippine signa abbreviation normalisation
        text = re.sub(r'\bS\.?O\.?S\.?\b', 'PRN', text, flags=re.IGNORECASE)
        text = re.sub(r'\bE\.?O\.?D\.?\b', 'QOD', text, flags=re.IGNORECASE)
        text = re.sub(r'\bT\.?I\.?W\.?\b', 'three times weekly', text, flags=re.IGNORECASE)
        text = re.sub(r'\bB\.?I\.?W\.?\b', 'twice weekly',       text, flags=re.IGNORECASE)
        text = re.sub(r'\bq\.?wk\.?\b', 'once weekly', text, flags=re.IGNORECASE)
        text = re.sub(r'\bq\.?a\.?m\.?\b', 'mane', text, flags=re.IGNORECASE)
        text = re.sub(r'\bq\.?n\.?\b', 'hs', text, flags=re.IGNORECASE)
        text = re.sub(r'\bad\s+lib(?:itum)?\b', 'PRN', text, flags=re.IGNORECASE)
        text = re.sub(r'\bss\b(?=\s*(?:tab|cap|tablet|capsule|tsp|tbsp)\b)', '1/2', text, flags=re.IGNORECASE)
        text = re.sub(r'(?i)^(?:sig\.?|s)\s*[:.]\s*', '', text)
        text = re.sub(r'\b8[Dd]\b(?!\d)', 'BD', text)
        text = re.sub(r'\b80\b(?=\s*(?:daily|tab|cap))', 'BD', text, flags=re.IGNORECASE)
        text = re.sub(r'\bT[O0][Ss]\b', 'TDS', text, flags=re.IGNORECASE)
        text = re.sub(r'\bQ[O0][Ss]\b', 'QDS', text, flags=re.IGNORECASE)
        text = re.sub(r'\bB\.D\.\B', 'BD',  text, flags=re.IGNORECASE)
        text = re.sub(r'\bT\.D\.S\.\B', 'TDS', text, flags=re.IGNORECASE)
        text = re.sub(r'\bQ\.D\.S\.\B', 'QDS', text, flags=re.IGNORECASE)
        text = re.sub(r'\b0[Dd]\b(?!\d)', 'OD', text)
        text = re.sub(r'\bq\.(\d+)\.h\.?(?=\s|[,;)]|\Z)', r'q\1h', text, flags=re.IGNORECASE)
        text = re.sub(r'\blx([1-4])\b', r'1x\1', text)
        text = re.sub(r'\b([1-4])xl\b', r'\1x1', text)

        # Philippine N-N-N dosing shorthand
        text = re.sub(r'\b1[ -]1[ -]1\b', '1 tab three times daily', text)
        text = re.sub(r'\b1[ -]0[ -]1\b', '1 tab twice daily',        text)
        text = re.sub(r'\b1[ -]1[ -]0\b', '1 tab twice daily',        text)
        text = re.sub(r'\b0[ -]1[ -]1\b', '1 tab twice daily',        text)
        text = re.sub(r'\b1[ -]0[ -]0\b', '1 tab once daily',         text)
        text = re.sub(r'\b0[ -]0[ -]1\b', '1 tab at bedtime',         text)

        text = re.sub(r'(tab(?:s|let(?:s)?)?)([Oo][Dd])\b', r'\1 \2', text, flags=re.IGNORECASE)

        text = self._MULTI_SPACE_RE.sub(' ', text)
        return text.strip()

    def _extract_dosage(self, text: str) -> str | None:
        m = self._DOSAGE_RE.search(text)
        if not m:
            return None
        amount = m.group(1).replace(',', '.')
        unit   = m.group(2).lower().rstrip('s') if m.group(2).lower() not in ('units',) else m.group(2).lower()
        unit   = self._UNIT_ALIAS.get(unit, unit)
        base   = f"{amount}{unit}"
        full   = m.group(0)
        if '/' in full:
            return re.sub(r'\s*/\s*', '/', full.strip())
        return base

    def _extract_duration(self, text: str) -> str | None:
        if self._DURATION_UNTIL_RE.search(text):
            return 'until finished'
        m = self._DURATION_RE.search(text)
        if m:
            num  = m.group(1)
            unit = m.group(2).lower()
            if unit in ('d', 'day'):
                unit = 'day'
            elif unit in ('wk', 'week'):
                unit = 'week'
            elif unit in ('mo', 'month'):
                unit = 'month'
            if int(num) == 1:
                return f"{num} {unit}"
            else:
                return f"{num} {unit}s"
        return None

    def _extract_frequency(self, text: str) -> str | None:
        explicit = (
            (r'\bonce\s+(?:a\s+)?(?:day|daily)\b', 'once daily'),
            (r'\btwice\s+(?:a\s+)?(?:day|daily)\b', 'twice daily'),
            (r'\bthree\s+(?:times?\s+)?(?:a\s+)?(?:day|daily)\b', 'three times daily'),
            (r'\bfour\s+times?\s+(?:a\s+)?(?:day|daily)\b', 'four times daily'),
            (r'\bq\s*([46812])\s*h\b', None),
        )
        for pattern, label in explicit:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if label:
                    return label
                return f"every {match.group(1)} hours"
        for pattern, label in self._FREQ_MAP:
            m = pattern.search(text)
            if m:
                if label is None:
                    hours = m.group(1)
                    return f"every {hours} hours"
                return label
        return None

    def _extract_quantity(self, text: str) -> str | None:
        match = re.search(r'(?:#|qty\.?|quantity\s*:?)\s*(\d+)\b', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_strength(self, text: str) -> str | None:
        dosage = self._extract_dosage(text)
        if not dosage:
            return None
        return re.sub(r'(\d)([A-Za-z])', r'\1 \2', dosage)

    def _extract_dosage_form(self, text: str) -> str | None:
        forms = (
            (r'\b(?:tab|tabs|tablet|tablets)\b', 'tablet'),
            (r'\b(?:cap|caps|capsule|capsules)\b', 'capsule'),
            (r'\bsyrup\b', 'syrup'),
            (r'\bsusp(?:ension)?\b', 'suspension'),
            (r'\b(?:solution|soln)\b', 'solution'),
            (r'\b(?:suppository|supp)\b', 'suppository'),
            (r'\b(?:injection|inj)\b', 'injection'),
            (r'\bpatch(?:es)?\b', 'patch'),
            (r'\bspray\b', 'spray'),
            (r'\binhaler\b', 'inhaler'),
            (r'\blozenge\b', 'lozenge'),
            (r'\bsachet\b', 'sachet'),
            (r'\bpowder\b', 'powder'),
            (r'\blotion\b', 'lotion'),
            (r'\bointment\b', 'ointment'),
            (r'\bcream\b', 'cream'),
            (r'\bgel\b', 'gel'),
            (r'\bdrops?\b', 'drops'),
        )
        for pattern, label in forms:
            if re.search(pattern, text, re.IGNORECASE):
                return label
        return None

    def _extract_dose_amount(self, text: str) -> str | None:
        match = re.search(
            r'\b([1-9]\d*(?:/\d+)?)\s*(tab(?:let)?s?|cap(?:sule)?s?|drop[s]?|puff[s]?)\b',
            text, re.IGNORECASE,
        )
        if not match:
            return None
        unit = match.group(2).lower()
        return f"{match.group(1)} {unit.rstrip('s')}"

    def _extract_timing(self, text: str) -> str | None:
        match = re.search(
            r'\b((?:before|after|with)\s+(?:breakfast|lunch|dinner|supper|meals?|food))\b',
            text, re.IGNORECASE,
        )
        if match:
            return match.group(1)
        uncertain_match = re.search(
            r'\b(before|after)\s+([A-Za-z]{3,})\s+\[UNCERTAIN\]',
            text,
        )
        if uncertain_match:
            return f"{uncertain_match.group(1).lower()} {uncertain_match.group(2)}"
        return None

    def _timing_is_uncertain(self, text: str) -> bool:
        return re.search(
            r'\b(?:before|after)\s+[A-Za-z]{3,}\s+\[UNCERTAIN\]',
            text,
        ) is not None

    def _extract_administration_instructions(self, text: str) -> str | None:
        match = re.search(
            r'\b((?:dissolve|rinse|gargle|apply|instill|swallow|chew|drink)\b[^,;.]*)',
            text, re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        match = re.search(r'\b(take\b[^,;.]*)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(
            r'\b(in\s+(?:full|half)\s+(?:a\s+)?glass\s+of\s+water)\b',
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _default_route(self, route: str | None, dosage_form: str | None) -> str | None:
        """Use oral only for oral dose forms when no route is explicitly stated."""
        if route:
            return route
        if dosage_form in {"tablet", "capsule", "syrup"}:
            return "oral"
        return None

    def _extract_route(self, text: str) -> str | None:
        for pattern, label in self._ROUTE_MAP:
            if pattern.search(text):
                return label
        return None

    def _adaptive_drug_threshold(self, candidate: str) -> int:
        n = len(candidate.replace(' ', ''))
        if n <= 4:  return 90
        if n <= 5:  return 87
        if n <= 6:  return 83
        if n <= 8:  return 78
        return self._DRUG_DETECT_THRESHOLD

    def _spell_correct_drug(self, candidate: str) -> str:
        if _DrugValidator._product_catalog is not None:
            result = _DrugValidator._product_catalog.match(candidate)
            return result['corrected'] if result['matched'] else candidate
        if len(candidate) < 4:
            return candidate
        lower = candidate.lower().strip()
        if lower in self._BRAND_TO_GENERIC:
            return self._BRAND_TO_GENERIC[lower]
        result = process.extractOne(
            lower, self._LEXICON_LOWER,
            scorer=fuzz.WRatio,
            score_cutoff=self._SPELL_CORRECT_THRESHOLD,
        )
        if result:
            _, _, idx = result
            return self._DRUG_LEXICON[idx]
        return candidate

    def _ocr_correct_token(self, tok: str) -> str:
        tok = re.sub(r'(?<=[a-z])0(?=[a-z])', 'o', tok)
        tok = re.sub(r'(?<=[a-z])1(?=[a-z])', 'i', tok)
        tok = re.sub(r'(?<=[a-z])5(?=[a-z])', 's', tok)
        tok = re.sub(r'rn(?=[aeiou])', 'm', tok)
        tok = re.sub(r'vv', 'w', tok)
        return tok

    def _extract_drug_name(self, text: str) -> str | None:
        stripped = self._DRUG_STRIP_RE.sub(' ', text)
        stripped = self._CLEAN_PUNCT_RE.sub(' ', stripped)
        stripped = self._MULTI_SPACE_RE.sub(' ', stripped).strip()
        stripped = re.sub(r'^\d+\s*', '', stripped)
        tokens = stripped.split()
        parts: list[str] = []
        for tok in tokens:
            if re.match(r'^[A-Za-z][A-Za-z\-]*$', tok):
                if tok.lower() not in self._NON_DRUG_TOKENS:
                    parts.append(tok)
            elif parts:
                break
        if not parts:
            for cap_m in re.finditer(r'\b([A-Z][a-z]{3,})\b', text):
                candidate = cap_m.group(1)
                if (
                    candidate.lower() not in self._NON_DRUG_TOKENS
                    and not self._is_drug_class(candidate)
                ):
                    corrected = self._spell_correct_drug(candidate)
                    if (corrected.lower() in self._LEXICON_LOWER_SET
                            and self._drug_name_passes_sanity(corrected)):
                        return corrected
            return None
        candidate = ' '.join(parts)
        if self._is_drug_class(candidate):
            return None
        corrected = self._spell_correct_drug(candidate)
        if corrected.lower() not in self._LEXICON_LOWER_SET or not self._drug_name_passes_sanity(corrected):
            return None
        return corrected

    def _detect_drug_in_line(self, line: str) -> tuple[str, str, bool] | None:
        catalog = _DrugValidator._product_catalog
        if catalog is not None:
            result = catalog.match(line)
            if result['matched']:
                is_brand = line.lower().strip() in self._BRAND_TO_GENERIC
                return (line, result['corrected'], is_brand)
            # Preserve unmatched OCR for review rather than applying a second,
            # differently scored correction before the catalog sees it.
            return None
        lower = line.lower()
        ocr_corrected_lower = ' '.join(self._ocr_correct_token(w) for w in lower.split())
        for brand, generic in self._BRAND_TO_GENERIC.items():
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, lower) or re.search(pattern, ocr_corrected_lower):
                occ_match = re.search(pattern, line, re.IGNORECASE)
                matched_substring = occ_match.group(0) if occ_match else brand
                return (matched_substring, generic, True)

        stripped_lower = self._DRUG_STRIP_RE.sub(' ', lower)
        stripped_lower = re.sub(r'\d+', ' ', stripped_lower)
        stripped_lower = re.sub(r'\s{2,}', ' ', stripped_lower).strip()

        if re.search(r'[+/&]|\bplus\b', stripped_lower):
            canonical = _DrugValidator._canonicalize_combination(stripped_lower)
            combination_match = self._COMBINATION_LOOKUP.get(canonical)
            if combination_match is not None:
                matched_substring = re.search(
                    r'[^\s]+\s*(?:[+/&]|plus)\s*[^\s]+', line, re.IGNORECASE
                )
                return (
                    matched_substring.group(0) if matched_substring else canonical,
                    combination_match,
                    False,
                )

        raw_tokens = stripped_lower.split()
        tokens = [self._ocr_correct_token(t) for t in raw_tokens]
        tokens = [t for t in tokens if len(t) >= 2 and re.search(r'[A-Za-z]', t)]
        n = len(tokens)
        candidates: list[str] = (
            tokens
            + [f"{tokens[i]} {tokens[i+1]}"             for i in range(n - 1)]
            + [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(n - 2)]
        )
        for candidate in candidates:
            if len(candidate) < 4:
                continue
            if self._is_non_drug_candidate(candidate) or self._is_drug_class(candidate):
                continue
            result = process.extractOne(
                candidate, self._LEXICON_LOWER,
                scorer=fuzz.WRatio,
                score_cutoff=self._adaptive_drug_threshold(candidate),
            )
            if result:
                _, score, idx = result
                generic = self._DRUG_LEXICON[idx]
                cand_len = len(candidate.replace(' ', ''))
                if cand_len <= 5 and score < 90:
                    if not self._DRUG_SUFFIX_RE.search(generic):
                        continue
                occ_match = re.search(re.escape(candidate), line, re.IGNORECASE)
                matched_substring = occ_match.group(0) if occ_match else candidate
                return (matched_substring, generic, False)

        return None

    def _is_continuation_line(self, line: str) -> bool:
        if self._CONTINUATION_START_RE.search(line):
            return True
        return (
            self._SIG_VOCAB_RE.search(line) is not None
            and self._DOSAGE_RE.search(line) is None
            and self._extract_quantity(line) is None
        )

    def _leading_catalog_brand(self, line: str) -> str | None:
        # Preserve strength-like digits that belong to a registered brand name.
        words = line.split()
        for count in range(min(len(words), 12), 0, -1):
            candidate = ' '.join(words[:count]).strip(' ,:;')
            if candidate.lower() in self._BRAND_TO_GENERIC:
                return candidate
        return None

    def _drug_name_part(self, line: str) -> str:
        """Return only the name-bearing portion before strength/form/quantity."""
        line = re.sub(r'^\s*\d+\s*[.)]\s*', '', line)
        line = re.sub(r'\[(?:UNCERTAIN|CUTOFF)\]', ' ', line)
        line = re.sub(r'\(\s*\)', ' ', line)
        catalog_brand = self._leading_catalog_brand(line)
        if catalog_brand:
            return catalog_brand
        name_part = re.split(r'\b(?:sig|sg|disp)\b|#\s*\d+|'
                             r'\b(?:qty|quantity)\.?\s*:?\s*\d+', line,
                             maxsplit=1, flags=re.IGNORECASE)[0]
        raw_name = name_part.strip()
        class_qualifier = re.match(
            r'^([A-Za-z][A-Za-z-]*)\s+'
            r'(cream|ointment|gel|lotion|drops?|patch|spray|powder|sachet)\b',
            raw_name,
            re.IGNORECASE,
        )
        if class_qualifier and self._is_drug_class(class_qualifier.group(1)):
            return class_qualifier.group(0).strip()
        name_part = self._DOSAGE_RE.sub(' ', name_part)
        name_part = re.sub(
            r'\b\d+(?:/\d+)?\s*(?:tab(?:let)?s?|cap(?:sule)?s?|drop[s]?|'
            r'puff[s]?|ml)\b',
            ' ',
            name_part,
            flags=re.IGNORECASE,
        )
        name_part = re.sub(
            r'\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|susp(?:ension)?|'
            r'solution|soln|cream|ointment|gel|drops?|lotion|injection|inj|'
            r'patch(?:es)?|spray|inhaler|lozenge|sachet|powder|suppository|supp)\b',
            ' ', name_part, flags=re.IGNORECASE,
        )
        return self._MULTI_SPACE_RE.sub(' ', name_part).strip(' /,-')

    def _line_has_medicine_evidence(self, line: str) -> bool:
        return bool(
            self._leading_catalog_brand(line)
            or self._DOSAGE_RE.search(line)
            or re.search(
                r'\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|susp(?:ension)?|'
                r'solution|soln|cream|ointment|gel|drops?)\b',
                line, re.IGNORECASE,
            )
            or self._extract_quantity(line)
        )

    def _parse_block(
        self,
        drug_name: str,
        brand_substring: str | None,
        is_brand: bool,
        lines: list[str],
        ocr_confidence: float = 1.0,
        drug_match_score: float = 1.0,
        drug_name_raw: str | None = None,
    ) -> dict[str, str | None]:
        dosage    = None
        frequency = None
        route     = None
        quantity  = None
        duration  = None
        dispense  = None
        meal_rel  = None
        combined_text = " ".join(lines)

        for line in lines:
            if dosage    is None: dosage    = self._extract_dosage(line)
            if frequency is None: frequency = self._extract_frequency(line)
            if route     is None: route     = self._extract_route(line)
            if duration  is None: duration  = self._extract_duration(line)
            if meal_rel  is None: meal_rel  = self._extract_meal_relation(line)
            if quantity is None: quantity = self._extract_quantity(line)

        sig_text = " ".join(lines)
        if frequency is None:
            frequency = self._extract_frequency(sig_text)
        if route is None:
            route = self._extract_route(sig_text)
        if duration is None:
            duration = self._extract_duration(sig_text)
        if meal_rel is None:
            meal_rel = self._extract_meal_relation(sig_text)
        if quantity is None:
            quantity = self._extract_quantity(sig_text)

        if frequency is None:
            for line in lines:
                _, nxm_freq = self._parse_nxm(line)
                if nxm_freq:
                    frequency = nxm_freq
                    break

        strength = self._extract_strength(combined_text)
        dosage_form = self._extract_dosage_form(combined_text)
        dose_amount = self._extract_dose_amount(combined_text)
        timing = self._extract_timing(combined_text)
        administration_instructions = self._extract_administration_instructions(combined_text)
        uncertain_fields = ["timing"] if timing and self._timing_is_uncertain(combined_text) else []
        route_source = "stated" if route else None
        route = self._default_route(route, dosage_form)
        if route and route_source is None:
            route_source = "inferred"
        duration_source = "stated" if duration else None
        if duration and quantity and re.search(r'#\s*' + re.escape(quantity), combined_text):
            duration = None
            duration_source = None
        if duration is None and quantity and dose_amount and frequency:
            daily_doses = self._FREQ_DAILY_DOSES.get(frequency)
            dose_match = re.match(r'(\d+)', dose_amount)
            if daily_doses and dose_match:
                days_value = int(quantity) / (int(dose_match.group(1)) * daily_doses)
                if days_value.is_integer() and days_value >= 1:
                    duration = f"{int(days_value)} days"
                    duration_source = "derived"

        if drug_name and not self._drug_name_passes_sanity(drug_name):
            drug_name = ""

        uncertain = (
            ocr_confidence < 0.45
            or drug_match_score < 0.60
            or not drug_name
        )
        field_values = {
            "drug_name": drug_name or None,
            "strength": strength,
            "dosage_form": dosage_form,
            "dose_amount": dose_amount,
            "frequency": frequency,
            "timing": timing,
            "administration_instructions": administration_instructions,
            "route_of_administration": route,
            "quantity": quantity,
            "duration": duration,
            "brand_name": brand_substring if is_brand else None,
        }
        marker_fields = {
            name for name, value in field_values.items()
            if value and name != "drug_name"
            and self._followed_by_uncertain(
                combined_text,
                self._field_source_match(combined_text, name, str(value)),
            )
        }
        field_flags = {
            name: bool(value) and (
                ocr_confidence < 0.45
                or name == "drug_name" and uncertain
                or name in marker_fields
            )
            for name, value in field_values.items()
        }
        if duration and duration_source == "derived":
            field_flags["duration"] = any(
                field_flags.get(name, False)
                for name in ("quantity", "dose_amount", "frequency")
            )
            marker_fields.discard("duration")
            if any(
                name in marker_fields
                for name in ("quantity", "dose_amount", "frequency")
            ):
                marker_fields.add("duration")
        uncertain_fields = [
            name for name, value in field_values.items()
            if value and field_flags.get(name, False)
        ]

        return {
            "brand_name":              brand_substring if is_brand else None,
            "drug_name":               drug_name or None,
            "dosage":                  dosage,
            "frequency":               frequency,
            "route_of_administration": route,
            "quantity":                quantity,
            "duration":                duration,
            "dispense_quantity":       dispense,
            "meal_relation":           meal_rel,
            "uncertain":               uncertain,
            "strength":                strength,
            "dosage_form":             dosage_form,
            "dose_amount":             dose_amount,
            "timing":                  timing,
            "administration_instructions": administration_instructions,
            "duration_source":         duration_source,
            "route_source":            route_source,
            "uncertain_fields":         uncertain_fields,
            "drug_name_raw":            drug_name_raw or brand_substring or drug_name or None,
            "_drug_name_raw":           drug_name_raw or brand_substring or drug_name or None,
            "_marker_fields":           list(marker_fields),
            "_drug_name_is_class":      self._is_drug_class(drug_name_raw or brand_substring or drug_name),
        }

    def parse_prescription(
        self,
        raw_text: str,
        ocr_confidence: float = 1.0,
    ) -> list[dict[str, str | None]]:
        lines = [self._normalize(line) for line in raw_text.splitlines()]
        lines = [l for l in lines if l]

        _empty = [{"brand_name": None, "drug_name": None, "dosage": None,
                   "frequency": None, "route_of_administration": None,
                   "quantity": None, "duration": None, "dispense_quantity": None,
                   "meal_relation": None, "uncertain": True,
                   "strength": None, "dosage_form": None, "dose_amount": None,
                   "timing": None, "administration_instructions": None,
                   "duration_source": None, "uncertain_fields": [],
                   "drug_name_raw": None}]

        if not lines:
            return _empty

        blocks: list[tuple[str, str | None, bool, float, list[str]]] = []
        current_generic:   str | None = None
        current_brand_sub: str | None = None
        current_is_brand:  bool = False
        current_score:     float = 1.0
        current_lines:     list[str] = []

        for line in lines:
            is_continuation = self._is_continuation_line(line)
            detected = None
            if not is_continuation and self._line_has_medicine_evidence(line):
                raw_name = self._drug_name_part(line)
                detected = self._detect_drug_in_line(raw_name)
                # Keep a separate unmatched medicine rather than joining its sig
                # to the preceding catalog match.
                if not detected and re.search(r'[A-Za-z]{3,}', raw_name) and not self._is_non_drug_candidate(raw_name):
                    detected = (raw_name, "", False)
            if detected:
                if current_generic is not None:
                    blocks.append((current_generic, current_brand_sub,
                                   current_is_brand, current_score, current_lines))
                matched_sub, generic, is_brand = detected
                if not generic:
                    score = 0.0
                elif is_brand:
                    score = 1.0
                else:
                    r = process.extractOne(
                        matched_sub.lower(), self._LEXICON_LOWER, scorer=fuzz.WRatio
                    )
                    score = (r[1] / 100.0) if r else 0.70
                current_generic   = generic
                current_brand_sub = matched_sub
                current_is_brand  = is_brand
                current_score     = score
                current_lines     = [line]
            else:
                current_lines.append(line)

        if current_generic is not None:
            blocks.append((current_generic, current_brand_sub,
                           current_is_brand, current_score, current_lines))

        if not blocks:
            if not any(self._line_has_medicine_evidence(line) for line in lines):
                logger.info("Dropping medicine candidate with no evidence: %s", raw_text.strip())
                return []
            raw_name = self._drug_name_part(lines[0])
            if self._is_drug_class(raw_name):
                return [self._parse_block(
                    "", None, False, lines,
                    ocr_confidence=ocr_confidence,
                    drug_match_score=0.0,
                    drug_name_raw=raw_name,
                )]
            drug_name = self._extract_drug_name(raw_name)
            if drug_name is None:
                if raw_name:
                    return [self._parse_block(
                        "", None, False, lines,
                        ocr_confidence=ocr_confidence,
                        drug_match_score=0.0,
                        drug_name_raw=raw_name,
                    )]
                logger.info("Dropping medicine candidate with no evidence: %s", raw_text.strip())
                return []
            return [self._parse_block(
                drug_name or "", drug_name, False, lines,
                ocr_confidence=ocr_confidence,
                drug_match_score=(
                    1.0 if drug_name.lower() in self._LEXICON_LOWER_SET else 0.5
                ),
                drug_name_raw=raw_name,
            )]

        return [
            self._parse_block(
                generic, brand_sub, is_brand, blk_lines,
                ocr_confidence=ocr_confidence,
                drug_match_score=match_score,
                drug_name_raw=self._drug_name_part(blk_lines[0]) if blk_lines else None,
            )
            for generic, brand_sub, is_brand, match_score, blk_lines in blocks
        ]


# Module-level singleton — preserves existing imports in main.py:
#   from parser import parse_prescription, _drug_name_passes_sanity
_parser = PrescriptionParser()
parse_prescription    = _parser.parse_prescription
_drug_name_passes_sanity = _parser._drug_name_passes_sanity
refresh_drug_catalog = _parser.refresh_drug_catalog
