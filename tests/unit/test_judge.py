import json
import pytest
from trade_agent.evaluation.judge import OptionalJudge


def test_missing_judge_key_is_not_zero_score():
    outcome = OptionalJudge().evaluate({'claims': []})
    assert outcome.status == 'judge_not_run'
    assert outcome.faithfulness is None
    assert outcome.scores is None
    assert outcome.coverage == 0.0
    assert outcome.errors


def test_failure_preserves_error_without_mutating_answer():
    def client(**kwargs):
        raise RuntimeError('provider timeout')
    answer = {'faithfulness': 1.0, 'claims': []}
    outcome = OptionalJudge(api_key='test', provider='injected', client=client, model='test-model').evaluate(answer)
    assert outcome.status == 'judge_failed'
    assert outcome.scores is None
    assert 'provider timeout' in outcome.errors[0]
    assert answer == {'faithfulness': 1.0, 'claims': []}


def test_success_strict_json_temperature_hashes_and_separate_scores():
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        return '{"faithfulness":0.75,"relevance":1.0}'
    judge = OptionalJudge(api_key='test', provider='injected', client=client, model='test-model')
    answer = {'faithfulness': 0.25}
    outcome = judge.evaluate(answer, evidence=[{'text': 'fact'}])
    assert outcome.status == 'judge_completed'
    assert outcome.faithfulness == 0.75
    assert outcome.coverage == 1.0
    assert outcome.temperature == 0
    assert outcome.to_dict()['temperature'] == 0
    assert len(outcome.prompt_hash) == len(outcome.model_hash) == 64
    assert calls[0]['temperature'] == 0
    assert calls[0]['response_format']['json_schema']['strict'] is True
    assert answer['faithfulness'] == 0.25
    json.dumps(outcome.to_dict(), allow_nan=False)


@pytest.mark.parametrize('response', ['not json', '{}', '{"faithfulness":true,"relevance":1}',
    '{"faithfulness":1.1,"relevance":1}', '{"faithfulness":NaN,"relevance":1}',
    '{"faithfulness":1,"relevance":1,"extra":2}', '{"faithfulness":0,"faithfulness":1,"relevance":1}'])
def test_malformed_provider_scores_are_failures(response):
    outcome = OptionalJudge(api_key='test', provider='injected', model='test', client=lambda **_: response).evaluate({})
    assert outcome.status == 'judge_failed'
    assert outcome.faithfulness is None
    assert outcome.scores is None
