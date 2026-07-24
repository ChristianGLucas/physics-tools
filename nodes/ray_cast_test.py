import math

from gen.messages_pb2 import Body, Ray, RayCastInput, Scene, Shape
from nodes.ray_cast import ray_cast
from nodes._test_helpers import _TestContext


def _ray_sphere_oracle(origin, direction, center, radius):
    """From-scratch ray/sphere intersection (the standard quadratic form),
    independent of pybullet. Returns (hit, t) where `direction` is used
    as-is (not normalized) so a `t` in [0, 1] means "within the from->to
    segment" — matching this package's `fraction` semantics directly.
    Returns the *nearest* (smallest non-negative t) root.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    cx, cy, cz = center
    ocx, ocy, ocz = ox - cx, oy - cy, oz - cz

    a = dx * dx + dy * dy + dz * dz
    b = 2 * (ocx * dx + ocy * dy + ocz * dz)
    c = (ocx * ocx + ocy * ocy + ocz * ocz) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False, None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    roots = sorted(r for r in (t1, t2) if r >= 0)
    if not roots:
        return False, None
    return True, roots[0]


def test_ray_hits_sphere_matches_ray_sphere_intersection_oracle():
    origin = (0.0, 0.0, 5.0)
    to = (0.0, 0.0, -5.0)
    direction = tuple(t - o for t, o in zip(to, origin))
    center = (0.0, 0.0, 0.0)
    radius = 1.0

    hit, t_expected = _ray_sphere_oracle(origin, direction, center, radius)
    assert hit and t_expected is not None

    scene = Scene(bodies=[Body(id="ball", shape=Shape(type="sphere", radius=radius), mass=0.0)])
    result = ray_cast(
        _TestContext(),
        RayCastInput(
            scene=scene,
            rays=[Ray(id="r1", from_point={"x": origin[0], "y": origin[1], "z": origin[2]},
                      to_point={"x": to[0], "y": to[1], "z": to[2]})],
        ),
    )
    assert result.error.code == ""
    assert len(result.hits) == 1
    h = result.hits[0]
    assert h.hit is True
    assert h.body_id == "ball"
    assert abs(h.fraction - t_expected) < 1e-6
    expected_point = tuple(o + t_expected * d for o, d in zip(origin, direction))
    assert abs(h.point.x - expected_point[0]) < 1e-6
    assert abs(h.point.y - expected_point[1]) < 1e-6
    assert abs(h.point.z - expected_point[2]) < 1e-6
    # The hit is the top of a sphere centered at the origin: normal points +Z.
    assert abs(h.normal.z - 1.0) < 1e-6


def test_ray_missing_every_body_reports_no_hit():
    scene = Scene(bodies=[Body(id="ball", shape=Shape(type="sphere", radius=1.0), mass=0.0,
                                position={"x": 10, "y": 10, "z": 10})])
    result = ray_cast(
        _TestContext(),
        RayCastInput(
            scene=scene,
            rays=[Ray(id="r1", from_point={"x": 0, "y": 0, "z": 5}, to_point={"x": 0, "y": 0, "z": -5})],
        ),
    )
    assert result.error.code == ""
    assert len(result.hits) == 1
    assert result.hits[0].hit is False
    assert result.hits[0].body_id == ""


def test_multiple_rays_are_matched_back_by_id():
    scene = Scene(bodies=[Body(id="ball", shape=Shape(type="sphere", radius=1.0), mass=0.0)])
    result = ray_cast(
        _TestContext(),
        RayCastInput(
            scene=scene,
            rays=[
                Ray(id="hits", from_point={"x": 0, "y": 0, "z": 5}, to_point={"x": 0, "y": 0, "z": -5}),
                Ray(id="misses", from_point={"x": 20, "y": 0, "z": 5}, to_point={"x": 20, "y": 0, "z": -5}),
            ],
        ),
    )
    by_id = {h.ray_id: h for h in result.hits}
    assert by_id["hits"].hit is True
    assert by_id["misses"].hit is False


def test_invalid_shape_returns_structured_error_not_crash():
    scene = Scene(bodies=[Body(id="bad", shape=Shape(type="not-a-shape"), mass=0.0)])
    result = ray_cast(
        _TestContext(),
        RayCastInput(scene=scene, rays=[Ray(id="r1", to_point={"x": 0, "y": 0, "z": -1})]),
    )
    assert result.error.code == "INVALID_SHAPE"
