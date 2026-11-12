from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import trocr_service as service


@pytest.mark.parametrize('option,engine,label', [
    ('clearx-trocr', 'claude-sonnet-4-6', 'CleaRx TrOCR'),
    ('claude-sonnet-4-6', 'claude-sonnet-4-6', 'CleaRx TrOCR'),
    ('claude-sonnet-5-5', 'claude-sonnet-5-5', 'Claude Sonnet 5.5'),
])
def test_options_route_to_configured_engine(monkeypatch, option, engine, label):
    create = Mock(return_value=SimpleNamespace(content=[
        SimpleNamespace(type='thinking'), SimpleNamespace(type='text', text='Amoxicillin 500mg'),
    ]))
    monkeypatch.setattr(service, '_client', SimpleNamespace(messages=SimpleNamespace(create=create)))
    model, public_name = service.resolve_model(option)
    assert public_name == label
    assert service.run_trocr_ocr(np.zeros((8, 8, 3), dtype=np.uint8), model) == ('Amoxicillin 500mg', None)
    assert create.call_args.kwargs['model'] == engine
    image = create.call_args.kwargs['messages'][0]['content'][0]
    assert image['source']['media_type'] == 'image/jpeg'


def test_unknown_model_rejected_before_provider(monkeypatch):
    create = Mock()
    monkeypatch.setattr(service, '_client', SimpleNamespace(messages=SimpleNamespace(create=create)))
    with pytest.raises(ValueError, match='Unsupported recognition option'):
        service.run_trocr_ocr(np.zeros((8, 8, 3), dtype=np.uint8), 'invalid')
    create.assert_not_called()


def test_default_model_and_missing_provider_confidence(monkeypatch):
    create = Mock(return_value=SimpleNamespace(content=[SimpleNamespace(type='text', text='Amoxicillin')]))
    monkeypatch.setattr(service, '_client', SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert service.run_trocr_ocr(np.zeros((8, 8, 3), dtype=np.uint8)) == ('Amoxicillin', None)
    assert create.call_args.kwargs['model'] == 'claude-sonnet-4-6'


def test_retry_backoff_is_bounded(monkeypatch):
    error = RuntimeError('rate limited')
    error.status_code = 429
    create = Mock(side_effect=[error, error, SimpleNamespace(content=[SimpleNamespace(type='text', text='ok')])])
    sleep = Mock()
    monkeypatch.setattr(service, '_client', SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(service.time, 'sleep', sleep)
    assert service._message_with_retry('system', []) == 'ok'
    assert create.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [2., 4.]


def test_non_retryable_error_returns_unavailable(monkeypatch):
    create = Mock(side_effect=RuntimeError('unavailable'))
    monkeypatch.setattr(service, '_client', SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert service._message_with_retry('system', []) is None
    create.assert_called_once()


def test_sdk_automatic_retries_are_disabled(monkeypatch):
    import anthropic
    constructor = Mock()
    monkeypatch.setattr(service, '_client', None)
    monkeypatch.setattr(service, '_ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setattr(anthropic, 'Anthropic', constructor)
    assert service._init()
    assert constructor.call_args.kwargs['max_retries'] == 0
    assert constructor.call_args.kwargs['timeout'] == 45.


def test_missing_credentials_does_not_construct_client(monkeypatch):
    import anthropic
    constructor = Mock()
    monkeypatch.setattr(anthropic, 'Anthropic', constructor)
    monkeypatch.setattr(service, '_client', None)
    monkeypatch.setattr(service, '_ANTHROPIC_API_KEY', '')
    assert not service._init()
    constructor.assert_not_called()


def test_parse_entrypoint_remains_deterministic(monkeypatch):
    import parser
    parse = Mock(return_value=[{'drug_name': 'Example'}])
    monkeypatch.setattr(parser, 'parse_prescription', parse)
    monkeypatch.setattr(service, '_message_with_retry', Mock(side_effect=AssertionError('no remote parse')))
    assert service.run_trocr_parse('raw prescription') == [{'drug_name': 'Example'}]
    parse.assert_called_once_with('raw prescription')
