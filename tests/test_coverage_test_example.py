def test_add():
    from app.utils.coverage_test_example import add
    assert add(2, 3) == 5
    assert add(-1, 1) == 0 