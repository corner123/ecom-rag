from dataclasses import dataclass


@dataclass(frozen=True)
class Coupon:
    code: str
    threshold_cents: int
    discount_cents: int
    stackable: bool = False


def calculate_discount(subtotal_cents: int, coupons: list[Coupon]) -> int:
    eligible = [item for item in coupons if subtotal_cents >= item.threshold_cents]
    if not eligible:
        return 0
    stackable = [item for item in eligible if item.stackable]
    fixed = max((item.discount_cents for item in eligible if not item.stackable), default=0)
    discount = max(fixed, sum(item.discount_cents for item in stackable))
    return min(discount, subtotal_cents)
