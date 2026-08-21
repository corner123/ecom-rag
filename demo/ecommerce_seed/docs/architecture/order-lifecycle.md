# Order lifecycle

Orders start in `pending_payment`. A successful payment moves an order to
`paid`; fulfilment then moves it to `fulfilling`. Cancellation is allowed only
before payment. Refunds use `refund_pending` before `refunded` so asynchronous
provider callbacks cannot skip the reviewable intermediate state.

The design document explains intent, but `ALLOWED_TRANSITIONS` in
`commerce_demo/order_service.py` is authoritative for the current state graph.
Every payment confirmation carries a `payment_id`; a repeated identifier must
not apply the transition twice.
