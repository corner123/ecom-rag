from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from sqlalchemy import Connection
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from trade_agent.db.sql_executor import (
    ReadOnlySqlExecutor,
    SqlExecutionError,
    SqlExecutionResult,
    SqlExecutionTimeout,
    SqlTransportError,
)
from trade_agent.db.sql_validator import ValidatedSql


@pytest.fixture
def identity_hmac_key() -> bytes:
    return bytes(range(32))


def _validated() -> ValidatedSql:
    return ValidatedSql(
        sql="SELECT SUM(tr.trade_amount) AS trade_amount FROM trade_records AS tr LIMIT 1",
        params={},
        schema_fingerprint="fingerprint",
        dataset_id="dataset",
        is_synthetic=True,
        effective_start_date=date(2025, 3, 1),
        effective_end_date=date(2025, 3, 2),
        aggregation_grain=(),
        time_grain="total",
        metric_names=("trade_amount",),
        projection_aliases=("trade_amount",),
        bound_filter_names=(),
        limit=1,
    )


class _Socket:
    def __init__(
        self, timeout: float | None, events: list[str], *, fail_restore: bool = False
    ) -> None:
        self.timeout = timeout
        self.events = events
        self.fail_restore = fail_restore

    def gettimeout(self) -> float | None:
        return self.timeout

    def settimeout(self, timeout: float | None) -> None:
        if self.fail_restore and timeout == 7.0:
            raise OSError("socket already closed")
        self.timeout = timeout
        self.events.append(f"socket:{timeout}")


def _connection_that_fails_initial_sql(
    mysql_code: int, message: str
) -> tuple[Connection, SimpleNamespace, _Socket, list[str]]:
    connection = create_autospec(Connection, instance=True)
    connection.__dict__["dialect"] = SimpleNamespace(name="mysql", driver="pymysql")
    events: list[str] = []
    socket = _Socket(7.0, events)
    driver = SimpleNamespace(_read_timeout=None, _sock=socket)
    connection.connection.driver_connection = driver
    connection.in_transaction.return_value = False

    def fail_scalar(*_args: object, **_kwargs: object) -> object:
        events.append("initial_sql")
        assert driver._read_timeout == 3.0
        assert socket.timeout == 3.0
        raise DBAPIError("state query", {}, Exception(mysql_code, message), False)

    connection.scalar.side_effect = fail_scalar
    return connection, driver, socket, events


def test_parameter_digest_is_order_independent_and_type_aware(identity_hmac_key: bytes) -> None:
    values = {
        "amount": Decimal("1.00"),
        "active": True,
        "day": date(2025, 3, 2),
        "count": 1,
        "country": "US",
    }
    reversed_values = dict(reversed(tuple(values.items())))

    assert ReadOnlySqlExecutor._parameter_digest(
        values, identity_hmac_key
    ) == ReadOnlySqlExecutor._parameter_digest(
        reversed_values, identity_hmac_key
    )
    assert ReadOnlySqlExecutor._parameter_digest(
        {"value": 1}, identity_hmac_key
    ) != ReadOnlySqlExecutor._parameter_digest(
        {"value": "1"}, identity_hmac_key
    )
    assert ReadOnlySqlExecutor._parameter_digest(
        {"value": date(2025, 3, 2)}, identity_hmac_key
    ) != ReadOnlySqlExecutor._parameter_digest({"value": "2025-03-02"}, identity_hmac_key)
    assert ReadOnlySqlExecutor._parameter_digest(
        {"value": Decimal("1.00")}, identity_hmac_key
    ) == ReadOnlySqlExecutor._parameter_digest({"value": Decimal("1")}, identity_hmac_key)


def test_parameter_digest_is_keyed_and_not_the_enumerable_public_sha256(
    identity_hmac_key: bytes,
) -> None:
    first = ReadOnlySqlExecutor._parameter_digest({"a": "bc"}, identity_hmac_key)
    second = ReadOnlySqlExecutor._parameter_digest({"ab": "c"}, identity_hmac_key)
    private = ReadOnlySqlExecutor._parameter_digest({"country": "secret-value"}, identity_hmac_key)
    other_key = ReadOnlySqlExecutor._parameter_digest({"country": "secret-value"}, bytes(range(1, 33)))
    public_payload = json.dumps(
        [{"name": "country", "value": {"type": "str", "value": "secret-value"}}],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert first != second
    assert private != other_key
    assert private != sha256(public_payload).hexdigest()
    assert "secret-value" not in private


def test_final_query_identity_has_an_independent_hmac_domain(
    identity_hmac_key: bytes,
) -> None:
    validated = _validated()
    parameter_digest = ReadOnlySqlExecutor._parameter_digest(
        validated.params, identity_hmac_key
    )
    public_identity_payload = json.dumps(
        {
            "sql": validated.sql,
            "filters": validated.bound_filter_names,
            "parameter_digest": parameter_digest,
            "schema": validated.schema_fingerprint,
            "dataset": validated.dataset_id,
            "is_synthetic": validated.is_synthetic,
            "effective_start_date": validated.effective_start_date.isoformat(),
            "effective_end_date": validated.effective_end_date.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    query_id = ReadOnlySqlExecutor._query_identity(
        validated, parameter_digest, identity_hmac_key
    )
    other_key_query_id = ReadOnlySqlExecutor._query_identity(
        validated, parameter_digest, bytes(range(1, 33))
    )

    assert query_id != sha256(public_identity_payload).hexdigest()
    assert query_id != other_key_query_id


def test_public_execution_result_does_not_expose_internal_parameter_digest() -> None:
    assert "parameter_digest" not in SqlExecutionResult.model_fields


def test_executor_requires_a_nonempty_runtime_identity_hmac_key() -> None:
    connection = create_autospec(Connection, instance=True)

    for key in (None, b""):
        with pytest.raises(SqlExecutionError, match="identity HMAC key"):
            ReadOnlySqlExecutor(connection, identity_hmac_key=key)


@pytest.mark.parametrize(
    ("mysql_code", "message", "expected"),
    [
        (2013, "operation timed out", SqlExecutionTimeout),
        (2006, "server has gone away", SqlTransportError),
    ],
)
def test_initial_session_state_sql_is_bounded_classified_and_restores_driver_timeout(
    identity_hmac_key: bytes,
    mysql_code: int,
    message: str,
    expected: type[SqlExecutionError],
) -> None:
    connection, driver, socket, events = _connection_that_fails_initial_sql(mysql_code, message)
    executor = ReadOnlySqlExecutor(connection, identity_hmac_key=identity_hmac_key)

    with pytest.raises(expected):
        executor.execute(_validated())

    assert events == ["socket:3.0", "initial_sql", "socket:7.0"]
    assert driver._read_timeout is None
    assert socket.timeout == 7.0


def test_executor_rejects_non_pymysql_adapter_before_session_sql(identity_hmac_key: bytes) -> None:
    connection = create_autospec(Connection, instance=True)
    connection.__dict__["dialect"] = SimpleNamespace(name="mysql", driver="other")

    with pytest.raises(SqlExecutionError, match="PyMySQL"):
        ReadOnlySqlExecutor(connection, identity_hmac_key=identity_hmac_key).execute(_validated())

    connection.scalar.assert_not_called()


def test_initial_rollback_is_bounded_and_uses_timeout_taxonomy(
    identity_hmac_key: bytes,
) -> None:
    connection, driver, socket, events = _connection_that_fails_initial_sql(
        2013, "operation timed out"
    )
    connection.in_transaction.return_value = True

    def fail_rollback() -> None:
        events.append("initial_rollback")
        assert driver._read_timeout == 3.0
        assert socket.timeout == 3.0
        raise DBAPIError("rollback", {}, Exception(2013, "operation timed out"), False)

    connection.rollback.side_effect = fail_rollback

    with pytest.raises(SqlExecutionTimeout):
        ReadOnlySqlExecutor(
            connection, identity_hmac_key=identity_hmac_key
        ).execute(_validated())

    assert events == ["socket:3.0", "initial_rollback", "socket:7.0"]
    connection.scalar.assert_not_called()


def test_closed_socket_during_restore_does_not_mask_typed_transport_error(
    identity_hmac_key: bytes,
) -> None:
    connection, driver, socket, _events = _connection_that_fails_initial_sql(
        2006, "server has gone away"
    )
    socket.fail_restore = True
    connection.invalidate.side_effect = SQLAlchemyError("invalidate also failed")
    connection.close.side_effect = SQLAlchemyError("close also failed")

    with pytest.raises(SqlTransportError):
        ReadOnlySqlExecutor(
            connection, identity_hmac_key=identity_hmac_key
        ).execute(_validated())

    assert driver._read_timeout is None
    connection.invalidate.assert_called_once()
    connection.close.assert_called_once()
