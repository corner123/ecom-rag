# Promotion rules

Coupons are evaluated against the pre-discount subtotal. Non-stackable coupons
compete by discount amount, while eligible stackable coupons are summed. The
engine chooses the larger of those two results and caps the final discount at
the subtotal, so an order total cannot become negative.

The demo deliberately omits user segmentation, campaign budgets, distributed
locks, and production pricing experiments.
