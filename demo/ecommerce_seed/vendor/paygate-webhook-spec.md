# PayGate webhook specification (synthetic vendor contract)

This fictional specification exists only for the ecom-rag demo.

Each callback contains `event_id`, `event_type`, `payment_id`, `order_id`, and
`amount_cents`. Consumers must deduplicate callbacks by `event_id`. The
`payment.succeeded` event confirms a payment only when its amount matches the
order amount. The `X-PayGate-Signature` header is the lowercase hexadecimal
HMAC-SHA256 digest of the raw request body using the shared webhook secret.

Consumers should acknowledge duplicate events without applying business state
again. This document does not claim compatibility with any real provider.
