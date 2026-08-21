from commerce_demo.inventory import InventoryLedger
from commerce_demo.models import Order, PaymentEvent
from commerce_demo.order_service import OrderService
from commerce_demo.payment import PaymentWebhookHandler


def test_inventory_reservation_is_idempotent():
    ledger = InventoryLedger({"sku-1": 3})
    first = ledger.reserve_stock(reservation_id="r-1", sku="sku-1", quantity=2)
    second = ledger.reserve_stock(reservation_id="r-1", sku="sku-1", quantity=2)
    assert first is second
    assert ledger.available["sku-1"] == 1


def test_payment_event_is_processed_once():
    order = Order("o-1", 1000)
    handler = PaymentWebhookHandler(OrderService())
    event = PaymentEvent("evt-1", "pay-1", "o-1", 1000, "payment.succeeded")
    assert handler.handle(event, order) == "processed"
    assert handler.handle(event, order) == "duplicate"
