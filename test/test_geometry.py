__author__ = "Eric Dose, Albuquerque"

from math import pi, sqrt

import pytest

import astropak.geometry as geom


@pytest.fixture(scope='function')
def setup_xy():
    xy1a = geom.XY(4, 5)
    xy1b = geom.XY(4, 5)
    xy2 = geom.XY(7, 9)
    dxy = geom.DXY(3, 4)
    return xy1a, xy1b, xy2, dxy


class Test_ClassXY:
    def test_xy_constructor(self, setup_xy):
        xy1a, xy1b, xy2, dxy = setup_xy
        assert (xy1a.x, xy1a.y) == (4, 5)
        assert xy1a == xy1b
        assert xy1a is not xy1b

    def test_xy_add(self, setup_xy):
        xy1a, xy1b, xy2, dxy = setup_xy
        assert xy1a + dxy == xy2
        with pytest.raises(TypeError):
            _ = xy1a + xy2
            _ = xy1a + (5, 6)
            _ = xy1a + 5

    def test_xy_subtract(self, setup_xy):
        xy1a, xy1b, xy2, dxy = setup_xy
        assert xy2 - xy1a == dxy
        assert xy2 - dxy == xy1a
        with pytest.raises(TypeError):
            _ = xy2 - (5, 6)
            _ = xy2 - 5

    def test_vector_to(self, setup_xy):
        xy1a, xy1b, xy2, dxy = setup_xy
        assert xy1a.vector_to(xy2) == dxy
        assert xy1a.vector_to(xy1b) == geom.DXY(0, 0)
        with pytest.raises(TypeError):
            _ = xy1a.vector_to(dxy)
            _ = xy1a.vector_to((5, 6))

    def test_xy_operators_not_implemented(self, setup_xy):
        xy1a, xy1b, xy2, dxy = setup_xy
        with pytest.raises(NotImplementedError):
            _ = xy1a.__and__(xy2)
            _ = xy1a is True
            _ = xy1a.__or__(xy2)
            _ = xy1a / xy2
            _ = xy1a / dxy


@pytest.fixture(scope='function')
def setup_dxy():
    xy1 = geom.XY(4, 5)
    xy2 = geom.XY(7, 9)
    dxy1 = geom.DXY(3, 4)
    dxy2a = geom.DXY(55, 66)
    dxy2b = geom.DXY(55, 66)
    return xy1, xy2, dxy1, dxy2a, dxy2b


class Test_ClassDXY:
    def test_dxy_constructor(self, setup_dxy):
        xy1, xy2, dxy1, dxy2a, dxy2b = setup_dxy
        assert (dxy1.dx, dxy1.dy) == (3, 4)
        assert (dxy2a.dx, dxy2b.dy) == (55, 66)
        assert dxy2a == dxy2b
        assert dxy2a is not dxy2b

    def test_dxy_add(self, setup_dxy):
        xy1, xy2, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy1 + dxy2a == geom.DXY(58, 70)
        assert dxy1 + xy1 == xy2
        with pytest.raises(TypeError):
            _ = dxy2a + (5, 6)
            _ = dxy2a + 5

    def test_dxy_subtract(self, setup_dxy):
        xy1, xy2, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy2a - dxy1 == geom.DXY(52, 62)
        with pytest.raises(TypeError):
            _ = dxy1 - xy1
            _ = dxy2a - (5, 6)
            _ = dxy2a - 5

    def test_dxy_multiply_fns(self, setup_dxy):
        xy1, xy2, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy1 * 3.5 == 3.5 * dxy1 == geom.DXY(10.5, 14)
        with pytest.raises(TypeError):
            _ = dxy1 * xy1
            _ = dxy1 * (5, 6)

    def test_dxy_division(self, setup_dxy):
        xy1, xy2, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy1 / 2 == geom.DXY(1.5, 2)
        with pytest.raises(ZeroDivisionError):
            _ = dxy1 / 0.0
        with pytest.raises(TypeError):
            _ = dxy1 / dxy2a
            _ = dxy1 / xy1
            _ = dxy1 / (5, 6)

    def test_dxy_angle_with(self, setup_dxy):
        xy1, _, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy1.angle_with(dxy1) == 0.0
        assert geom.DXY(5, 2).angle_with(geom.DXY(-2, 5)) == pytest.approx(pi / 2, abs=0.000001)
        assert geom.DXY(5, 2).angle_with(geom.DXY(2, -5)) == pytest.approx((3 / 2) * pi, abs=0.000001)
        with pytest.raises(TypeError):
            _ = dxy1.angle_with((5, 5))
            _ = dxy1.angle_with(xy1)
            _ = dxy1.angle_with(5)

    def test_dxy_dot(self, setup_dxy):
        xy1, xy2, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy1.dot(dxy2a) == 429
        assert dxy1.dot(dxy1) == 25
        with pytest.raises(TypeError):
            _ = dxy1.dot((5, 5))
            _ = dxy1.dot(xy1)
            _ = dxy1.dot(5)

    def test_dxy_properties(self, setup_dxy):
        _, _, dxy1, dxy2a, dxy2b = setup_dxy
        assert dxy1.length2 == 25
        assert dxy2a.length2 == 7381
        assert dxy1.length == 5
        assert dxy2a.length == pytest.approx(sqrt(7381), abs=0.000001)
        assert geom.DXY(5, 0).direction == 0
        assert geom.DXY(0, 5).direction == pytest.approx(pi / 2, abs=0.000001)
        assert geom.DXY(5, 5).direction == pytest.approx(pi / 4, abs=0.000001)

    def test_dxy_bool(self):
        assert not geom.DXY(0, 0)
        assert geom.DXY(0, 5)
        assert geom.DXY(5, 0)
        assert geom.DXY(-2, 5)


@pytest.fixture(scope='function')
def setup_rectangle_in_2d():
    # A(5,0) B(0,2) C(1,5) and D(6,3)
    rect_pt_a = geom.XY(5, 0)
    rect_pt_b = geom.XY(0, 2)
    rect_pt_c = geom.XY(1, 4.5)
    return geom.Rectangle_in_2D(rect_pt_a, rect_pt_b, rect_pt_c)


class Test_Rectangle_in_2D:
    def test_constructor(self, setup_rectangle_in_2d):
        rect = setup_rectangle_in_2d
        assert (rect.a, rect.b, rect.c) == (geom.XY(5, 0), geom.XY(0, 2), geom.XY(1, 4.5))
        assert rect.ab == geom.DXY(-5, 2)
        assert rect.bc == geom.DXY(1, 2.5)
        bad_pt_c = geom.XY(1, 4.49)
        with pytest.raises(ValueError):
            _ = geom.Rectangle_in_2D(rect.a, rect.b, bad_pt_c)

    def test_properties(self, setup_rectangle_in_2d):
        rect = setup_rectangle_in_2d
        assert rect.area == pytest.approx(sqrt(29) * sqrt(10), abs=0.000001)

    def test_contains_point(self, setup_rectangle_in_2d):
        rect = setup_rectangle_in_2d
        assert rect.contains_point(geom.XY(5.49, 1.25)) is True
        assert rect.contains_point(geom.XY(5.51, 1.25)) is False
        assert rect.contains_point(geom.XY(3.5, 3.49)) is True
        assert rect.contains_point(geom.XY(3.5, 3.51)) is False
        assert rect.contains_point(geom.XY(1, 4.49)) is True
        assert rect.contains_point(geom.XY(1, 4.51)) is False
        assert rect.contains_point(geom.XY(3, 2.25)) is True
        assert rect.contains_point(geom.XY(100, 100)) is False
        pts_on_edge = [rect.a, rect.b, rect.c,
                       geom.XY(2.5, 1),
                       geom.XY(0.5, 3.25),
                       geom.XY(3.5, 3.5),
                       geom.XY(5.5, 1.25)]
        assert all([rect.contains_point(pt, include_edges=True) for pt in pts_on_edge])
        assert not any([rect.contains_point(pt, include_edges=False) for pt in pts_on_edge])

    def test_contains_points(self, setup_rectangle_in_2d):
        rect = setup_rectangle_in_2d
        xy_array = (geom.XY(5.49, 1.25),
                    geom.XY(5.51, 1.25),
                    geom.XY(3.5, 3.49),
                    geom.XY(3.5, 3.51),
                    geom.XY(1, 4.49),
                    geom.XY(1, 4.51),
                    geom.XY(3, 2.25),
                    geom.XY(100, 100))
        contains = rect.contains_points(xy_array)
        assert contains == 4 * [True, False]

    def test_contains_points_unitgrid(self, setup_rectangle_in_2d):
        rect = setup_rectangle_in_2d
        pass






