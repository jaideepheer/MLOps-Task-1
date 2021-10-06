
class TestMath:
    import math
    def test_sqrt(self):
        assert self.math.sqrt(4) == 2
    def test_ceil(self):
        assert self.math.ceil(1.000001) == 2
    def test_floor(self):
        assert self.math.floor(1.99999) == 1
    def test_factorial(self):
        assert self.math.factorial(5) == 120

def test_equals():
    assert 1 == 1