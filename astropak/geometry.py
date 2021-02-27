__author__ = "Eric Dose, Albuquerque"

from collections import namedtuple
import numbers
from math import sqrt, atan2, pi


class XY(namedtuple('XY', ['x', 'y'])):
    """ Holds one Cartesian point (x,y) in a plane.
        Operators behave as you would expect, especially with vector objects dxy:
            xy + dxy -> new xy, displaced
            xy1 - xy2 -> dxy, the displacement from xy2 to xy1
            xy - dxy -> new xy, displaced
            other math operators are not implemented and will raise exception
            other operators are largely as for namedtuple, but probably rarely used.
    """
    __slots__ = ()

    def __add__(self, other):
        if isinstance(other, DXY):
            return XY(self.x + other.dx, self.y + other.dy)
        raise TypeError('XY.__add__() requires type DXY as operand.')

    def __and__(self, other):
        raise NotImplementedError

    def __bool__(self, other):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, XY):
            return DXY(self.x - other.x, self.y - other.y)
        elif isinstance(other, DXY):
            return XY(self.x - other.dx, self.y - other.dy)
        raise TypeError('XY.__sub__() requires type XY or DXY as operand.')

    def __truediv__(self, other):
        raise NotImplementedError

    def vector_to(self, other):
        """ Returns DXY vector to other XY point. """
        if isinstance(other, XY):
            return other - self
        raise TypeError('XY.vector_to() requires type XY as operand.')


class DXY(namedtuple('DXY', ['dx', 'dy'])):
    """ Holds one vector (dx, dy), that is, a displacement from one point to another in a plane.
            Operators behave as you would expect, especially with point objects xy:
            dxy1 + dxy1 -> new dxy, vector sum
            dxy + dy -> new xy, new XY point, displaced by DXY vector
            bool(dxy) -> True iff vector of non-zero length
            dxy * a, a * dxy -> vector with length multiplied by a, same direction
            dxy - dxy -> new dxy, vector sum
            dxy / a -> vector with length divided by a, same direction
            .angle_with(dxy2) -> angle with other DXY vector, in radians, within range 0 to pi
            .dot(dxy2) -> dot product with other DXY vector
            .length2 -> vector's squared length
            .length -> vector's length
            .direction -> vector's direction relative to +x axis, within range 0 to pi, in radians
            other math operators are not implemented and will raise exception
            other operators are largely as for namedtuple, but probably rarely used.
    """
    __slots__ = ()

    def __add__(self, other):
        if isinstance(other, DXY):
            return DXY(self.dx + other.dx, self.dy + other.dy)
        elif isinstance(other, XY):
            return XY(other.x + self.dx, other.y + self.dy)
        raise TypeError('DXY.__add__() requires type DXY or XY as operand.')

    def __bool__(self):
        return (self.length2 > 0)

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return XY(other * self.dx, other * self.dy)
        raise TypeError('DXY.__mul__() requires float scalar as operand.')

    def __rmul__(self, other):
        if isinstance(other, numbers.Real):
            return XY(other * self.dx, other * self.dy)
        raise TypeError('DXY.__rmul__() requires float scalar as operand.')

    def __sub__(self, other):
        if isinstance(other, DXY):
            return DXY(self.dx - other.dx, self.dy - other.dy)
        raise TypeError('DXY.__sub__() requires type DXY as operand.')

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            if other != 0:
                return XY(self.dx / other, self.dy / other)
            raise ZeroDivisionError
        raise TypeError('DXY.__div__() requires float scalar as operand.')

    def angle_with(self, other):
        """ Returns angle with other DXY vector, in radians, within range 0 to pi. """
        if isinstance(other, DXY):
            angle = other.direction - self.direction
            if angle < 0:
                angle += 2 * pi
            return angle
        raise TypeError('DXY.angle_with() requires type DXY as operand.')

    def dot(self, other):
        """ Returns dot product with other DXY vector. """
        if isinstance(other, DXY):
            return self.dx * other.dx + self.dy * other.dy
        raise TypeError('DXY.dot () requires type DXY as operand.')

    @property
    def length2(self):
        """ Return square of vector length. """
        return self.dx ** 2 + self.dy ** 2

    @property
    def length(self):
        """ Return vector length. """
        return sqrt(self.length2)

    @property
    def direction(self):
        """ Return angle of vector's angle relative to positive x axis
            (e.g., +y yields pi/2), in radians. Returns zero if vector is zero-length.
        """
        if self.length2 > 0.0:
            return atan2(self.dy, self.dx)
        return 0.0





