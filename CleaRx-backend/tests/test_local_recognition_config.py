import pytest

from recognition_config import local_model_configuration
from trocr_service import resolve_model


def test_public_primary_resolves_to_local_engine():
    assert resolve_model("clearx-trocr") == ("local-trocr", "CleaRx TrOCR")


def test_missing_explicit_checkpoint_fails_instead_of_silently_substituting(monkeypatch, tmp_path):
    monkeypatch.setenv("TROCR_CHECKPOINT_DIR", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="does not exist"):
        local_model_configuration()


def test_configured_checkpoint_records_location_without_inventing_training_data(monkeypatch, tmp_path):
    monkeypatch.setenv("TROCR_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("TROCR_BASE_MODEL", "microsoft/trocr-base-handwritten")
    config = local_model_configuration()
    assert config["weights_path"] == str(tmp_path.resolve())
    assert config["fine_tuned_on"] == "unverified local checkpoint"
    assert config["base_model"] == "microsoft/trocr-base-handwritten"


def test_registry_uses_local_configuration_instead_of_cloud_weights(monkeypatch):
    from unittest.mock import Mock
    import main
    import recognition_config
    monkeypatch.setattr(main, "_DB_AVAILABLE", True)
    config = dict(base_model="microsoft/trocr-base-handwritten", weights_path="n/a", fine_tuned_on="n/a")
    monkeypatch.setattr(recognition_config, "local_model_configuration", lambda: config)
    registry = Mock(return_value="local-model")
    monkeypatch.setattr(main, "_db_get_or_create_trocr_model", registry)
    monkeypatch.setattr(main, "_db_create_inference_job", Mock(return_value="job"))
    monkeypatch.setattr(main, "_db_finish_inference_job", Mock())
    monkeypatch.setattr(main, "_db_create_transcription_output_with_fields", Mock(return_value=("output", {})))
    main._persist_module_2(image_id="image", preprocessing_id="pre", provider="trocr-local",
        model="local-trocr", raw_text="test", confidence=.8, image_shape=(8, 8, 3), medicines=[],
        inference_time_ms=1., failure_reason=None)
    assert registry.call_args.kwargs["fine_tuned_on"] == "n/a"
    assert registry.call_args.kwargs["weights_path"] == "n/a"
