# Inventory reservation decision

Checkout uses reservation rather than immediate final deduction. The ledger
reduces available stock when `reserve_stock` succeeds, then either confirms the
deduction after payment or restores availability through
`release_reservation`. `reservation_id` is the idempotency key.

This design avoids overselling while allowing payment timeouts to release
stock. Only a `reserved` record may be confirmed or released. Confirmed stock
is not restored by the release operation.
