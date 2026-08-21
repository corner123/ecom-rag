from dataclasses import dataclass


@dataclass
class Reservation:
    reservation_id: str
    sku: str
    quantity: int
    status: str = "reserved"


class InsufficientStock(ValueError):
    pass


class InventoryLedger:
    def __init__(self, available: dict[str, int]):
        self.available = dict(available)
        self.reservations: dict[str, Reservation] = {}

    def reserve_stock(
        self, *, reservation_id: str, sku: str, quantity: int
    ) -> Reservation:
        existing = self.reservations.get(reservation_id)
        if existing is not None:
            return existing
        if quantity <= 0 or self.available.get(sku, 0) < quantity:
            raise InsufficientStock(sku)
        self.available[sku] -= quantity
        reservation = Reservation(reservation_id, sku, quantity)
        self.reservations[reservation_id] = reservation
        return reservation

    def confirm_deduction(self, reservation_id: str) -> bool:
        reservation = self.reservations[reservation_id]
        if reservation.status == "confirmed":
            return False
        if reservation.status != "reserved":
            raise ValueError("only reserved stock can be confirmed")
        reservation.status = "confirmed"
        return True

    def release_reservation(self, reservation_id: str) -> bool:
        reservation = self.reservations[reservation_id]
        if reservation.status == "released":
            return False
        if reservation.status != "reserved":
            raise ValueError("only reserved stock can be released")
        self.available[reservation.sku] += reservation.quantity
        reservation.status = "released"
        return True
