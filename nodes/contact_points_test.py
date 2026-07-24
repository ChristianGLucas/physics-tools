import math

from gen.messages_pb2 import Body, ContactPointsInput, Scene, Shape
from nodes.contact_points import contact_points
from nodes._test_helpers import _TestContext


def test_overlapping_spheres_penetration_matches_distance_formula():
    """Independent oracle: for two spheres, penetration depth is the
    closed-form `center_distance - r1 - r2` (negative when overlapping).
    Computed from scratch here, not by calling into pybullet.
    """
    c1 = (0.0, 0.0, 0.0)
    c2 = (1.5, 0.0, 0.0)
    r1 = r2 = 1.0
    center_distance = math.dist(c1, c2)
    expected_penetration = center_distance - r1 - r2  # -0.5

    scene = Scene(
        bodies=[
            Body(id="a", shape=Shape(type="sphere", radius=r1), mass=0.0,
                 position={"x": c1[0], "y": c1[1], "z": c1[2]}),
            Body(id="b", shape=Shape(type="sphere", radius=r2), mass=0.0,
                 position={"x": c2[0], "y": c2[1], "z": c2[2]}),
        ]
    )
    result = contact_points(_TestContext(), ContactPointsInput(scene=scene))
    assert result.error.code == ""
    assert len(result.contacts) == 1
    c = result.contacts[0]
    assert {c.body_a_id, c.body_b_id} == {"a", "b"}
    assert abs(c.distance - expected_penetration) < 1e-6
    # Normal should point along the line joining the two centers (+/- x).
    assert abs(abs(c.normal_on_b.x) - 1.0) < 1e-6


def test_static_static_pair_still_reports_a_contact():
    """Regression for a real pybullet gotcha found while building this
    package: Bullet's own getContactPoints()/performCollisionDetection()
    pipeline silently skips a pair where BOTH bodies are static (mass 0)
    — a broadphase optimization that assumes static pairs never need a
    dynamics response. This node must not inherit that blind spot: two
    overlapping mass-0 bodies (e.g. two walls) must still be reported.
    """
    scene = Scene(
        bodies=[
            Body(id="wall_a", shape=Shape(type="box", half_extents={"x": 1, "y": 1, "z": 1}), mass=0.0,
                 position={"x": 0, "y": 0, "z": 0}),
            Body(id="wall_b", shape=Shape(type="box", half_extents={"x": 1, "y": 1, "z": 1}), mass=0.0,
                 position={"x": 1.5, "y": 0, "z": 0}),
        ]
    )
    result = contact_points(_TestContext(), ContactPointsInput(scene=scene))
    assert result.error.code == ""
    assert len(result.contacts) == 1


def test_separated_bodies_report_no_contact():
    scene = Scene(
        bodies=[
            Body(id="a", shape=Shape(type="sphere", radius=1.0), mass=0.0, position={"x": 0, "y": 0, "z": 0}),
            Body(id="b", shape=Shape(type="sphere", radius=1.0), mass=0.0, position={"x": 5, "y": 0, "z": 0}),
        ]
    )
    result = contact_points(_TestContext(), ContactPointsInput(scene=scene))
    assert result.error.code == ""
    assert len(result.contacts) == 0


def test_invalid_shape_returns_structured_error_not_crash():
    scene = Scene(bodies=[Body(id="bad", shape=Shape(type="triangle"), mass=0.0)])
    result = contact_points(_TestContext(), ContactPointsInput(scene=scene))
    assert result.error.code == "INVALID_SHAPE"
