import hashlib
import hmac

from .models import Order, PaymentEvent
from .order_service import OrderService


def verify_signature(payload: bytes, signature: str, secret: bytes) -> bool:
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class PaymentWebhookHandler:
    def __init__(self, order_service: OrderService):
        self.order_service = order_service
        self.processed_event_ids: set[str] = set()

    def handle(self, event: PaymentEvent, order: Order) -> str:
        if event.event_id in self.processed_event_ids:
            return "duplicate"
        if event.event_type != "payment.succeeded":
            return "ignored"
        self.order_service.confirm_payment(
            order,
            payment_id=event.payment_id,
            amount_cents=event.amount_cents,
        )
        self.processed_event_ids.add(event.event_id)
        return "processed"
