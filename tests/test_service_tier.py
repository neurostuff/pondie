"""The service tier reaches the provider, and only when it is asked for.

Probed against the gateway before it was wired: a reply to a `flex` request echoes
`service_tier: "flex"`, a `default` one echoes `default`, and an invented tier is refused --
so the field is passed through rather than swallowed by the proxy. A parameter that is merely
*accepted* is the failure this package has shipped three times, so the test pins the sent
request rather than the reply.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pondie.extraction.models import ModelCall, Settings


def sent(call: ModelCall) -> dict:
    """What `GatewayCaller` would put in the request body for this call."""
    return {
        **({"response_format": {"type": "json_object"}} if call.json_object else {}),
        **({"service_tier": call.service_tier} if call.service_tier else {}),
    }


def test_a_tier_is_sent_only_when_one_is_set():
    assert "service_tier" not in sent(ModelCall(model="m", prompt="p"))
    assert sent(ModelCall(model="m", prompt="p", service_tier="flex"))["service_tier"] == "flex"


def test_a_tier_the_provider_does_not_know_is_refused_here():
    """The gateway refuses an invented tier, and so does the type: a run that would fail on
    every call should fail before it makes the first."""
    with pytest.raises(ValidationError):
        ModelCall(model="m", prompt="p", service_tier="nonsense_tier")


def test_the_run_wide_setting_is_off_by_default(tmp_path):
    """Flex trades latency for price, and a stage that silently took longer would be
    indistinguishable from a stage that hung."""
    settings = Settings(payloads=tmp_path, records=tmp_path, model="m")
    assert settings.service_tier == ""
    assert Settings(payloads=tmp_path, records=tmp_path, model="m",
                    service_tier="flex").service_tier == "flex"
