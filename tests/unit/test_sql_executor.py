from datetime import date
from decimal import Decimal

from trade_agent.db.sql_executor import ReadOnlySqlExecutor


def test_parameter_digest_is_order_independent_and_type_aware() -> None:
    values = {
        "amount": Decimal("1.00"),
        "active": True,
        "day": date(2025, 3, 2),
        "count": 1,
        "country": "US",
    }
    reversed_values = dict(reversed(tuple(values.items())))

    assert ReadOnlySqlExecutor._parameter_digest(values) == ReadOnlySqlExecutor._parameter_digest(
        reversed_values
    )
    assert ReadOnlySqlExecutor._parameter_digest({"value": 1}) != ReadOnlySqlExecutor._parameter_digest(
        {"value": "1"}
    )
    assert ReadOnlySqlExecutor._parameter_digest(
        {"value": date(2025, 3, 2)}
    ) != ReadOnlySqlExecutor._parameter_digest({"value": "2025-03-02"})
    assert ReadOnlySqlExecutor._parameter_digest(
        {"value": Decimal("1.00")}
    ) == ReadOnlySqlExecutor._parameter_digest({"value": Decimal("1")})


def test_parameter_digest_has_unambiguous_names_and_does_not_disclose_values() -> None:
    first = ReadOnlySqlExecutor._parameter_digest({"a": "bc"})
    second = ReadOnlySqlExecutor._parameter_digest({"ab": "c"})
    private = ReadOnlySqlExecutor._parameter_digest({"country": "secret-value"})

    assert first != second
    assert "secret-value" not in private
