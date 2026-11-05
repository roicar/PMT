'''
    Utility for integer arithmatics\\
    Coded and annoated by Yuxuan SONG
    Last modified time: 28 Jan 2026
'''
from typing import Final, Union, List
from numpy import iinfo, int32
from functools import total_ordering, reduce

INT32_MAX: Final = iinfo(int32).max
INT32_MIN: Final = iinfo(int32).min
INT32_MAX_: Final = str(INT32_MAX)
INT32_MIN_: Final = str(-INT32_MIN)


@total_ordering
class Int32X:
    r'''Class for storage and computation of 32-bit integers with overflows handled.'''
    __slots__ = ('value', 'pos_inf', 'neg_inf')

    def __init__(self, number: str) -> None:
        self.pos_inf = False
        self.neg_inf = False
        sign = -1 if number.startswith('-') else 1
        unsigned = number.lstrip('+-')
        limit = INT32_MAX_ if sign == 1 else INT32_MIN_
        # Overflow detection
        if len(unsigned) > len(limit) or len(unsigned) == len(limit) and unsigned > limit:
            self.value = None
            self.pos_inf = (sign == 1)
            self.neg_inf = (sign == -1)
            return
        self.value = sign * int(unsigned)

    def is_pos_inf(self) -> bool: return self.pos_inf
    def is_neg_inf(self) -> bool: return self.neg_inf
    def numerable(self) -> bool: return self.value is not None

    def __eq__(self, other: Union[int, 'Int32X']) -> bool:
        if isinstance(other, List):
            return False
        if isinstance(other, int):
            other = Int32X(str(other))
        if self.pos_inf or self.neg_inf or other.pos_inf or other.neg_inf:
            return self.pos_inf == other.pos_inf and self.neg_inf == other.neg_inf
        return self.value == other.value

    def __lt__(self, other: Union[int, 'Int32X']) -> bool:
        if isinstance(other, int):
            other = Int32X(str(other))
        if self.neg_inf:
            return not other.neg_inf
        if other.pos_inf:
            return not self.pos_inf
        if self.numerable() and other.numerable():
            return self.value < other.value
        if self.pos_inf:
            return False
        if other.neg_inf:
            return False

    def __add__(self, other: Union[int, 'Int32X']) -> 'Int32X':
        if isinstance(other, int):
            other = Int32X(str(other))
        if self.pos_inf and other.pos_inf:
            return Int32X.from_pos_inf()
        if self.neg_inf and other.neg_inf:
            return Int32X.from_neg_inf()
        if self.pos_inf and other.neg_inf:
            return Int32X('0')
        if self.neg_inf and other.pos_inf:
            return Int32X('0')
        if self.pos_inf:
            return Int32X.from_pos_inf()
        if self.neg_inf:
            return Int32X.from_neg_inf()
        if other.pos_inf:
            return Int32X.from_pos_inf()
        if other.neg_inf:
            return Int32X.from_neg_inf()
        # Positive overflow
        if self.value > 0 and other.value > 0 and self.value > INT32_MAX - other.value:
            return Int32X.from_pos_inf()
        # Negative overflow
        if self.value < 0 and other.value < 0 and self.value < INT32_MIN - other.value:
            return Int32X.from_neg_inf()
        return Int32X(str(self.value + other.value))

    def __neg__(self) -> 'Int32X':
        if self.pos_inf:
            return Int32X.from_neg_inf()
        if self.neg_inf:
            return Int32X.from_pos_inf()
        return Int32X(str(-self.value))

    def __sub__(self, other: Union[int, 'Int32X']) -> 'Int32X':
        if isinstance(other, int):
            other = Int32X(str(other))
        return self + (-other)

    def __mul__(self, other: 'Int32X') -> 'Int32X':
        if self.pos_inf and other.pos_inf:
            return Int32X.from_pos_inf()
        if self.neg_inf and other.neg_inf:
            return Int32X.from_pos_inf()
        if self.pos_inf and other.neg_inf:
            return Int32X.from_neg_inf()
        if self.neg_inf and other.pos_inf:
            return Int32X.from_neg_inf()
        if self.pos_inf:
            return Int32X.from_pos_inf()
        if self.neg_inf:
            return Int32X.from_neg_inf()
        if other.pos_inf:
            return Int32X.from_pos_inf()
        if other.neg_inf:
            return Int32X.from_neg_inf()
        if self.value == 0 or other.value == 0:
            return Int32X('0')
        if abs(self.value) > INT32_MAX // abs(other.value):
            # Sign determines which overflow
            if (self.value > 0 and other.value > 0) or (other.value < 0 and other.value < 0):
                return Int32X.from_pos_inf()
            else:
                return Int32X.from_neg_inf()
        return Int32X(str(self.value * other.value))

    def __radd__(self, other: 'Int32X') -> 'Int32X':
        if isinstance(other, int):
            return Int32X(str(other)) + self
        if isinstance(other, Int32X):
            return other + self

    def __rsub__(self, other: Union[int, 'Int32X']) -> 'Int32X':
        if isinstance(other, int):
            return Int32X(str(other)) - self
        if isinstance(other, Int32X):
            return other - self
    
    def __rmul__(self, other: Union[int, 'Int32X']) -> 'Int32X':
        if isinstance(other, int):
            return Int32X(str(other)) * self
        if isinstance(other, Int32X):
            return other * self

    def sign(self) -> int:
        if self.pos_inf:
            return +1
        if self.neg_inf:
            return -1
        if self.value > 0:
            return +1
        if self.value < 0:
            return -1
        return 0

    def sigfig(self, n: int) -> 'Int32X':
        r'''Fetch the first n significant figures.'''
        if self.pos_inf:
            return Int32X.from_pos_inf()
        if self.neg_inf:
            return Int32X.from_neg_inf()
        unsigned = str(abs(self.value))
        if len(unsigned) <= n:
            return Int32X(str(self.value))
        return Int32X(str(self.value)[:n + 1] if self.value < 0 else str(self.value)[:n])

    @classmethod
    def from_pos_inf(cls):
        obj = cls.__new__(cls)
        obj.value = None
        obj.pos_inf = True
        obj.neg_inf = False
        return obj

    @classmethod
    def from_neg_inf(cls):
        obj = cls.__new__(cls)
        obj.value = None
        obj.pos_inf = False
        obj.neg_inf = True
        return obj


def Int32X_accum(int32x_list: List) -> Int32X:
    r'''Accumulation on a list of Int32X objects.'''
    return reduce(lambda a, b: a * b, int32x_list, Int32X('1'))
