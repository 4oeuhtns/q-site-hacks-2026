import pytest

from period_router.layout import Layout


def test_swap_exchanges_two_qubits():
    layout = Layout({0: 5, 1: 6})
    layout.swap(5, 6)
    assert layout.where == {0: 6, 1: 5}
    assert layout.occupant == {6: 0, 5: 1}


def test_swap_into_an_empty_chair():
    layout = Layout({0: 5})
    layout.swap(5, 9)
    assert layout.where == {0: 9}
    assert layout.occupant == {9: 0}


def test_copy_is_independent():
    layout = Layout({0: 5, 1: 6})
    copy = layout.copy()
    copy.swap(5, 6)
    assert layout.where == {0: 5, 1: 6}


def test_emit_keeps_operand_order():
    layout = Layout({0: 5, 1: 6})
    assert layout.emit(("2Q", 1, 0)) == ("2Q", 6, 5)
    assert layout.emit(("1Q", 1)) == ("1Q", 6)
    with pytest.raises(ValueError):
        layout.emit(("3Q", 0, 1, 2))
