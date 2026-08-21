from dataclasses import dataclass, field
from enum import StrEnum


class OrderStatus(StrEnum):
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    FULFILLING = "fulfilling"
    CANCELLED = "cancelled"
    REFUND_PENDING = "refund_pending"
    REFUNDED = "refunded"


@dataclass
class Order:
    order_id: str
    amount_cents: int
    status: OrderStatus = OrderStatus.PENDING_PAYMENT
    payment_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PaymentEvent:
    event_id: str
    payment_id: str
    order_id: str
    amount_cents: int
    event_type: str
