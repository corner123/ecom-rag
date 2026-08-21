from .models import Order, OrderStatus


ALLOWED_TRANSITIONS = {
    OrderStatus.PENDING_PAYMENT: {OrderStatus.PAID, OrderStatus.CANCELLED},
    OrderStatus.PAID: {OrderStatus.FULFILLING, OrderStatus.REFUND_PENDING},
    OrderStatus.REFUND_PENDING: {OrderStatus.REFUNDED},
}


class InvalidOrderTransition(ValueError):
    pass


class OrderService:
    def transition(self, order: Order, target: OrderStatus) -> Order:
        if target not in ALLOWED_TRANSITIONS.get(order.status, set()):
            raise InvalidOrderTransition(f"{order.status} -> {target} is not allowed")
        order.status = target
        return order

    def confirm_payment(
        self, order: Order, *, payment_id: str, amount_cents: int
    ) -> bool:
        if payment_id in order.payment_ids:
            return False
        if amount_cents != order.amount_cents:
            raise ValueError("payment amount does not match order amount")
        self.transition(order, OrderStatus.PAID)
        order.payment_ids.add(payment_id)
        return True
