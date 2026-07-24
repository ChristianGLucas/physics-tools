from gen.messages_pb2 import Body, ComputeAABBInput, Scene, Shape
from nodes.compute_aabb import compute_aabb
from nodes._test_helpers import _TestContext


def test_sphere_aabb_matches_center_plus_minus_radius_formula():
    """Independent oracle: a sphere's AABB is exactly
    (center - radius, center + radius) on every axis — a closed-form
    result, not dependent on pybullet's internals."""
    center = (2.0, -3.0, 5.0)
    radius = 1.5
    scene = Scene(bodies=[Body(id="ball", shape=Shape(type="sphere", radius=radius), mass=0.0,
                                position={"x": center[0], "y": center[1], "z": center[2]})])
    result = compute_aabb(_TestContext(), ComputeAABBInput(scene=scene))
    assert result.error.code == ""
    assert len(result.boxes) == 1
    box = result.boxes[0]
    assert abs(box.min.x - (center[0] - radius)) < 1e-9
    assert abs(box.min.y - (center[1] - radius)) < 1e-9
    assert abs(box.min.z - (center[2] - radius)) < 1e-9
    assert abs(box.max.x - (center[0] + radius)) < 1e-9
    assert abs(box.max.y - (center[1] + radius)) < 1e-9
    assert abs(box.max.z - (center[2] + radius)) < 1e-9


def test_box_aabb_matches_center_plus_minus_half_extents_formula():
    center = (5.0, 5.0, 5.0)
    half_extents = (1.0, 2.0, 3.0)
    scene = Scene(bodies=[Body(
        id="crate",
        shape=Shape(type="box", half_extents={"x": half_extents[0], "y": half_extents[1], "z": half_extents[2]}),
        mass=0.0,
        position={"x": center[0], "y": center[1], "z": center[2]},
    )])
    result = compute_aabb(_TestContext(), ComputeAABBInput(scene=scene))
    assert result.error.code == ""
    box = result.boxes[0]
    assert (box.min.x, box.min.y, box.min.z) == (4.0, 3.0, 2.0)
    assert (box.max.x, box.max.y, box.max.z) == (6.0, 7.0, 8.0)


def test_filters_to_requested_body_ids_only():
    scene = Scene(bodies=[
        Body(id="a", shape=Shape(type="sphere", radius=1.0), mass=0.0),
        Body(id="b", shape=Shape(type="sphere", radius=1.0), mass=0.0, position={"x": 10, "y": 0, "z": 0}),
    ])
    result = compute_aabb(_TestContext(), ComputeAABBInput(scene=scene, body_ids=["b"]))
    assert result.error.code == ""
    assert len(result.boxes) == 1
    assert result.boxes[0].body_id == "b"


def test_unknown_body_id_returns_structured_error_not_crash():
    scene = Scene(bodies=[Body(id="a", shape=Shape(type="sphere", radius=1.0), mass=0.0)])
    result = compute_aabb(_TestContext(), ComputeAABBInput(scene=scene, body_ids=["does-not-exist"]))
    assert result.error.code == "NOT_FOUND"
