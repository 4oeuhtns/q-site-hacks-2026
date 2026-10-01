"""Who sits where, kept in both directions and updated on every SWAP."""


class Layout:
    def __init__(self, placement: dict[int, int]):
        self.where = dict(placement)  # logical -> physical
        self.occupant = {physical: logical for logical, physical in placement.items()}  # physical -> logical

    def copy(self) -> "Layout":
        return Layout(self.where)

    def swap(self, p: int, q: int) -> None:
        """Exchange whatever sits on physical qubits p and q (either may be empty)."""
        a = self.occupant.pop(p, None)
        b = self.occupant.pop(q, None)
        if a is not None:
            self.occupant[q] = a
            self.where[a] = q
        if b is not None:
            self.occupant[p] = b
            self.where[b] = p

    def emit(self, op: tuple) -> tuple:
        """Rewrite a program op onto physical qubits, keeping the operand order."""
        if op[0] == "2Q":
            return ("2Q", self.where[op[1]], self.where[op[2]])
        if op[0] == "1Q":
            return ("1Q", self.where[op[1]])
        raise ValueError(f"unknown op kind: {op[0]}")
