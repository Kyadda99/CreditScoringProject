"""What the integration suite does when no stack is listening."""

from __future__ import annotations

import httpx
import pytest

from integration.test_stack import stack_required, unreachable_stack

ERROR = httpx.ConnectError("refused")


def test_skips_when_the_stack_is_optional() -> None:
    with pytest.raises(pytest.skip.Exception):
        unreachable_stack("http://localhost:8080", ERROR, required=False)


def test_fails_when_the_stack_is_required() -> None:
    with pytest.raises(pytest.fail.Exception, match="localhost:8080"):
        unreachable_stack("http://localhost:8080", ERROR, required=True)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " on "])
def test_truthy_values_require_the_stack(value: str) -> None:
    assert stack_required(value) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_falsy_values_leave_the_stack_optional(value: str | None) -> None:
    assert stack_required(value) is False


def test_an_unrecognised_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="CS_INTEGRATION_REQUIRE_STACK"):
        stack_required("sure")
