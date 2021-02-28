__author__ = "Eric Dose, Albuquerque"

from collections import namedtuple
import numbers
from math import sqrt, atan2, pi

import numpy as np


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


class Rectangle_in_2D:
    """ Contains one rectangle in a 2-D Cartesian plane. """
    def __init__(self, xy_a, xy_b, xy_c):
        """
        :param xy_a: a vertex of rectangle. [XY object]
        :param xy_b: a vertex of rectangle, adjacent to xy_a. [XY object]
        :param xy_c: a vertex of rectangle, adjacent to xy_b and opposite xy_a. [XY object]
        """
        self.a, self.b, self.c = (xy_a, xy_b, xy_c)
        self.ab, self.bc = (self.b - self.a, self.c - self.b)
        if abs(self.ab.dot(self.bc)) > 1e-10 * min(self.ab.length, self.bc.length):
            raise ValueError('Rectangle_in_2D edges are not perpendicular.')

    @property
    def area(self):
        """ Return area of rectangle. """
        return self.ab.length * self.bc.length

    def contains_point(self, xy, include_edges=True):
        """ Returns True iff this rectangle contains point xy, else return False.
        :param xy: point to test. [XY object]
        :param include_edges: True iff a point on rectangle edge is to count as contained. [boolean]
        :return: True iff this rectangle contains point xy, else return False. [boolean]
        """
        dot_ab = self.ab.length2  # reflexive dot product.
        dot_bc = self.bc.length2  # "
        dot_ab_pt = self.ab.dot(xy - self.a)
        dot_bc_pt = self.bc.dot(xy - self.b)
        if include_edges:
            return (0 <= dot_ab_pt <= dot_ab) and (0 <= dot_bc_pt <= dot_bc)
        return (0 < dot_ab_pt < dot_ab) and (0 < dot_bc_pt < dot_bc)

    def contains_points(self, xy_array, include_edges=True):
        """ Returns True for each corresponding point in xy_array iff this rectangle contains that point,
            else return False. General case.
        :param xy_array: points to test, in arbitrary position and ordering. [list or tuple of XY objects]
        :param include_edges: True iff a point on rectangle edge is to count as contained. [boolean]
        :return: List of booleans corresponding to xy_array, each element True iff this rectangle contains
                 that point, else False. [list of boolean, same length as xy_array]
        """
        dot_ab = self.ab.length2  # reflexive dot product, compute only once for array.
        dot_bc = self.bc.length2  # "
        dot_ab_array = [self.ab.dot(pt - self.a) for pt in xy_array]
        dot_bc_array = [self.bc.dot(pt - self.b) for pt in xy_array]
        if include_edges:
            return[(0 <= dot_a_pt <= dot_ab) and (0 <= dot_b_pt <= dot_bc)
                   for (dot_a_pt, dot_b_pt) in zip(dot_ab_array, dot_bc_array)]
        return[(0 < dot_a_pt < dot_ab) and (0 < dot_b_pt < dot_bc)
               for (dot_a_pt, dot_b_pt) in zip(dot_ab_array, dot_bc_array)]

    def contains_points_unitgrid(self, x_min, x_max, y_min, y_max, include_edges=True):
        """ Returns True for each point in a unit grid iff this rectangle contains that point,
            else return False. Numpy-optimized case.
            Unit grid x: x_min, x_min+1, x_min+2, ... , x_max. Same for y. All 4 values must be integers.
        :param x_min: [integer]
        :param x_max: must >= x_min. [integer]
        :param y_min: [integer]
        :param y_max: must >= y_min. [integer]
        :param include_edges: True iff a point on rectangle edge is to count as contained. [boolean]
        :return: array, True iff each corresponding point is contained by rectangle.
                 [numpy array of booleans]
        """
        if any(not isinstance(val, int) for val in (x_min, x_max, y_min, y_max)):
            raise TypeError('All 4 grid edges must be integers but are not.')
        if x_max < x_min or y_max < y_min:
            raise ValueError('Grid point specifications must be in ascending order.')
        dot_ab = self.ab.length2  # reflexive dot product, precompute once for array.
        dot_bc = self.bc.length2  # "
        x, y = np.meshgrid(range(x_min, x_max + 1), range(y_min, y_max + 1))
        dot_ab_xy = self.ab.dx * (x - self.a.x) + self.ab.dy * (y - self.a.y)  # grid of a dot products.
        dot_bc_xy = self.bc.dx * (x - self.b.x) + self.bc.dy * (y - self.b.y)  # grid of b "
        if include_edges:
            return (0 <= dot_ab_xy) & (dot_ab_xy <= dot_ab) & (0 <= dot_bc_xy) & (dot_bc_xy <= dot_bc)
        return (0 < dot_ab_xy) & (dot_ab_xy < dot_ab) & (0 < dot_bc_xy) & (dot_bc_xy < dot_bc)


class Circle_in_2D:
    """ Contains one circle in a 2-D Cartesian plane. """
    def __init__(self, xy_origin, radius):
        """
        :param xy_origin: x,y origin of circle. [XY object]
        :param radius: [float]
        """
        self.origin = xy_origin
        self.radius = radius
        if not isinstance(self.origin, XY):
            raise TypeError('Circle origin must be a XY object')
        self.x, self.y = (self.origin.x, self.origin.y)

    @property
    def area(self):
        """ Return area of this circle. """
        return pi * (self.radius ** 2)

    def contains_point(self, xy, include_edges=True):
        """ Returns True iff this circle contains point xy, else return False.
        :param xy: point to test. [XY object]
        :param include_edges: True iff a point on circle edge is to count as contained. [boolean]
        :return: True iff this circle contains point xy, else return False. [boolean]
        """
        distance2 = (xy - self.origin).length2
        if include_edges:
            return distance2 <= self.radius ** 2
        return distance2 < self.radius ** 2

    def contains_points(self, xy_array, include_edges=True):
        """ Returns True for each corresponding point in xy_array iff this circle contains that point,
            else return False. General case.
        :param xy_array: points to test, in arbitrary position and ordering. [list or tuple of XY objects]
        :param include_edges: True iff a point on circle edge is to count as contained. [boolean]
        :return: List of booleans corresponding to xy_array, each element True iff this circle contains
        that point, else False. [list of booleans, same length as xy_array]
        """
        distances2 = [(xy - self.origin).length2 for xy in xy_array]
        if include_edges:
            return [distance2 <= self.radius ** 2 for distance2 in distances2]
        return [distance2 < self.radius ** 2 for distance2 in distances2]

    def contains_points_unitgrid(self, x_min, x_max, y_min, y_max, include_edges=True):
        """ Returns True for each point in a unit grid iff this circle contains that point,
            else return False. Numpy-optimized case.
            Unit grid x: x_min, x_min+1, x_min+2, ... , x_max. Same for y. All 4 values must be integers.
        :param x_min: [integer]
        :param x_max: must >= x_min. [integer]
        :param y_min: [integer]
        :param y_max: must >= y_min. [integer]
        :param include_edges: True iff a point on circle edge is to count as contained. [boolean]
        :return: array, True iff each corresponding point is contained by circle. [numpy array of booleans]
        """
        if any(not isinstance(val, int) for val in (x_min, x_max, y_min, y_max)):
            raise TypeError('All 4 grid edges must be integers but are not.')
        if x_max < x_min or y_max < y_min:
            raise ValueError('Grid point specifications must be in ascending order.')
        dx2_values = [(x - self.origin.x) ** 2 for x in range(x_min, x_max + 1)]
        dy2_values = [(y - self.origin.y) ** 2 for y in range(y_min, y_max + 1)]
        dx2, dy2 = np.meshgrid(dx2_values, dy2_values)
        if include_edges:
            return (dx2 + dy2) <= self.radius ** 2
        return (dx2 + dy2) < self.radius ** 2


