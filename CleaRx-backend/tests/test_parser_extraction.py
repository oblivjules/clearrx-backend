import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from parser import parse_prescription


def test_extracts_sig_fields_from_raw_ocr_text():
    medicine = parse_prescription(
        "N-Acetylcysteine\n600mg Tab\n#7\nSg one TAB in half\n"
        "glass of water once\na day after sugar [UNCERTAIN]"
    )[0]

    assert medicine["frequency"] == "once daily"
    assert medicine["quantity"] == "7"
    assert medicine["duration"] == "7 days"
    assert medicine["duration_source"] == "derived"
    assert medicine["strength"] == "600 mg"
    assert medicine["dosage_form"] == "tablet"
    assert medicine["dose_amount"] == "1 tab"
    assert medicine["administration_instructions"] == "in half glass of water"
    assert medicine["route_of_administration"] == "oral"
    assert medicine["timing"] == "after sugar"
    assert medicine["meal_relation"] is None
    assert medicine["uncertain_fields"] == ["timing"]


def test_preserves_quantity_and_derives_duration_for_once_daily_dose():
    medicine = parse_prescription(
        "Acetylcysteine 600mg Tab #7 take 1 tab once daily after supper "
        "dissolve in half a glass of water"
    )[0]

    assert medicine["quantity"] == "7"
    assert medicine["frequency"] == "once daily"
    assert medicine["timing"] == "after supper"
    assert medicine["administration_instructions"] == "dissolve in half a glass of water"
    assert medicine["duration"] == "7 days"
    assert medicine["duration_source"] == "derived"


def test_full_glass_capture_is_one_medicine_with_derived_duration():
    medicines = parse_prescription(
        "N-Acetylcysteine / 600mg Tab / #7 / Sig: one TAB in full / "
        "glass of water once / a day after [UNCERTAIN]"
    )

    assert len(medicines) == 1
    medicine = medicines[0]
    assert medicine["strength"] == "600 mg"
    assert medicine["dosage_form"] == "tablet"
    assert medicine["dose_amount"] == "1 tab"
    assert medicine["frequency"] == "once daily"
    assert medicine["administration_instructions"] == "in full glass of water"
    assert medicine["route_of_administration"] == "oral"
    assert medicine["quantity"] == "7"
    assert medicine["duration"] == "7 days"
    assert medicine["duration_source"] == "derived"
    assert medicine["timing"] is None
    assert medicine["uncertain_fields"] == []


def test_unknown_timing_is_captured_only_with_uncertain_marker():
    uncertain = parse_prescription(
        "N-Acetylcysteine 600mg Tab #7 Sig one tab after Sugar [UNCERTAIN]"
    )[0]
    without_marker = parse_prescription(
        "N-Acetylcysteine 600mg Tab #7 Sig one tab after surgery"
    )[0]

    assert uncertain["timing"] == "after Sugar"
    assert uncertain["meal_relation"] is None
    assert uncertain["uncertain_fields"] == ["timing"]
    assert without_marker["timing"] is None
    assert without_marker["uncertain_fields"] == []


def test_split_route_and_meal_timing_are_joined():
    medicine = parse_prescription(
        "N-Acetylcysteine 600mg Tab #7\nSig one tab by\nmouth after\nmeals"
    )[0]

    assert medicine["route_of_administration"] == "oral"
    assert medicine["timing"] == "after meals"
    assert medicine["meal_relation"] == "after meals"


def test_three_times_daily_does_not_derive_non_whole_duration():
    medicine = parse_prescription(
        "N-Acetylcysteine\n600mg Tab\n#7\nSig one tab in half\n"
        "glass of water three\na day [UNCERTAIN]"
    )[0]

    assert medicine["frequency"] == "three times daily"
    assert medicine["quantity"] == "7"
    assert medicine["duration"] is None
    assert medicine["duration_source"] is None


def test_single_line_frequency_forms_are_unchanged():
    for sig, expected in (
        ("once a day", "once daily"),
        ("bid", "twice daily"),
        ("3x/day", "three times daily"),
    ):
        medicine = parse_prescription(
            f"N-Acetylcysteine 600mg Tab #7 Sig one tab {sig}"
        )[0]
        assert medicine["frequency"] == expected


def test_two_drugs_keep_each_sig_with_its_drug():
    medicines = parse_prescription(
        "Acetylcysteine 600mg Tab #7\n"
        "Sig one tab once a day\n"
        "Amoxicillin 500mg Cap #10\n"
        "Sig two cap bid"
    )

    assert len(medicines) == 2
    assert medicines[0]["drug_name"] == "Acetylcysteine"
    assert medicines[0]["frequency"] == "once daily"
    assert medicines[1]["drug_name"] == "Amoxicillin"
    assert medicines[1]["frequency"] == "twice daily"


def test_sig_only_text_never_becomes_a_medicine():
    assert parse_prescription("Sig one tab twice daily glass of water") == []


def test_per_field_uncertainty_flags_do_not_copy_line_flag():
    text = (
        "N-Acetylcysteine / 600mg Tab / #7 / Sg one TAB in half / "
        "glass of water once / a day after sugar [UNCERTAIN]"
    )
    medicine = parse_prescription(text, ocr_confidence=0.9)[0]

    assert medicine["uncertain"] is False
    assert medicine["uncertain_fields"] == ["timing"]


def test_low_ocr_flags_every_populated_field():
    medicine = parse_prescription(
        "N-Acetylcysteine 600mg Tab #7 Sig one tab once a day",
        ocr_confidence=0.4,
    )[0]

    assert medicine["uncertain"] is True
    assert set(medicine["uncertain_fields"]) == {
        "drug_name", "strength", "dosage_form", "dose_amount",
        "frequency", "route_of_administration", "quantity", "duration",
    }


def test_unmatched_drug_flags_only_drug_name():
    medicine = parse_prescription("Mysterydrug 500mg tab")[0]

    assert medicine["drug_name"] is None
    assert medicine["drug_name_raw"] == "Mysterydrug"
    assert medicine["uncertain_fields"] == []


def test_drug_class_is_not_fuzzy_matched():
    medicine = parse_prescription("Antibiotic 1 g, 3 drops")[0]

    assert medicine["drug_name_raw"] == "Antibiotic"
    assert medicine["drug_name"] is None
    assert medicine["dosage_form"] == "drops"
    assert medicine["_drug_name_is_class"] is True
    assert medicine["uncertain"] is True


def test_dosage_forms_are_specific():
    for form in (
        "cream", "ointment", "gel", "lotion", "drops", "tablet", "capsule",
        "syrup", "suspension", "solution", "injection", "patch", "spray",
        "inhaler", "lozenge", "sachet", "powder", "suppository",
    ):
        medicine = parse_prescription(f"N-Acetylcysteine 1 g {form}")[0]
        assert medicine["dosage_form"] == form
