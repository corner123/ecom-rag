"""Optional schema-bound DeepSeek generation without credential disclosure."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from urllib.request import Request, urlopen

from pydantic import ValidationError

from trade_agent.agents.intent import QueryIntent
from trade_agent.evidence.models import Claim, Evidence
from trade_agent.evidence.validator import ValidationOutcome
from trade_agent.generation.base import AnswerGenerator, DraftAnswer, generation_evidence


Transport = Callable[[Mapping[str, object]], Mapping[str, object]]


class DeepSeekAnswerGenerator(AnswerGenerator):
    """A deliberately optional provider that accepts only a strict JSON response."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        timeout_seconds: float = 20.0,
        max_output_tokens: int = 1_024,
        max_response_bytes: int = 1_048_576,
        transport: Transport | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise ValueError("DeepSeek generation requires DEEPSEEK_API_KEY")
        if not isinstance(model, str) or not model.strip() or type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("DeepSeek configuration is invalid")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 8_192:
            raise ValueError("max_output_tokens must be an integer from 1 through 8192")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 4_194_304:
            raise ValueError("max_response_bytes must be an integer from 1 through 4194304")
        self._api_key = resolved_key
        self._base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._max_response_bytes = max_response_bytes
        self._transport = transport

    def generate(
        self,
        intent: QueryIntent,
        evidence: Sequence[Evidence],
        validation: ValidationOutcome,
    ) -> DraftAnswer:
        retained = generation_evidence(intent, evidence, validation)
        if not validation.can_answer:
            return DraftAnswer(answer=None, claims=(), refusal_reason=validation.error_code or "evidence_insufficient")
        payload = self._request_payload(intent, retained)
        response = self._transport(payload) if self._transport is not None else self._request(payload)
        draft = _parse_response(response)
        allowed_ids = {item.evidence_id for item in retained}
        if any(not set(claim.evidence_ids).issubset(allowed_ids) for claim in draft.claims):
            raise ValueError("DeepSeek response cited Evidence outside validated generation context")
        return draft

    def _request_payload(self, intent: QueryIntent, evidence: Sequence[Evidence]) -> dict[str, object]:
        context = [_public_evidence(item) for item in evidence]
        schema = DraftAnswer.model_json_schema()
        return {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "draft_answer", "strict": True, "schema": schema}},
            "messages": (
                {"role": "system", "content": "Return only JSON conforming to the supplied schema. Cite only supplied evidence IDs."},
                {"role": "user", "content": json.dumps({"question": intent.question, "evidence": context}, ensure_ascii=False, sort_keys=True)},
            ),
        }

    def _request(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        request = Request(
            self._base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - endpoint is explicitly configured
            raw_body = response.read(self._max_response_bytes + 1)
        if len(raw_body) > self._max_response_bytes:
            raise ValueError("DeepSeek response exceeds the maximum response size")
        body = json.loads(raw_body.decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("DeepSeek response was not an object")
        return body


def _public_evidence(evidence: Evidence) -> dict[str, object]:
    """Only reviewed, public semantic fields can cross the provider boundary."""
    return {
        "evidence_id": evidence.evidence_id,
        "entity_id": evidence.entity_id,
        "company_name": evidence.company_name,
        "country_code": evidence.country_code,
        "hs_code": evidence.hs_code,
        "fact_type": evidence.fact_type,
        "content": evidence.content,
        "valid_from": evidence.valid_from.isoformat() if evidence.valid_from else None,
        "valid_to": evidence.valid_to.isoformat() if evidence.valid_to else None,
        "time_grain": evidence.time_grain,
        "currencies": evidence.currencies,
        "units": evidence.units,
        "aggregation_grain": evidence.aggregation_grain,
    }


def _parse_response(response: Mapping[str, object]) -> DraftAnswer:
    content: object = response
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
    try:
        if isinstance(content, str):
            return DraftAnswer.model_validate_json(content)
        if not isinstance(content, dict):
            raise ValueError("DeepSeek response does not contain a JSON draft")
        return DraftAnswer.model_validate(content)
    except (ValidationError, ValueError) as exc:
        raise ValueError("DeepSeek response violates DraftAnswer schema") from exc
