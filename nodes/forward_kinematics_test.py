import math

from gen.messages_pb2 import Chain, ForwardKinematicsInput, Link, Shape
from nodes.forward_kinematics import forward_kinematics
from nodes._test_helpers import _TestContext

_TINY = Shape(type="sphere", radius=0.01)


def _planar_arm(l1, l2):
    """A textbook 2-SEGMENT planar arm, expressed as 3 links (this
    package's Chain models a JOINT's own position as fixed relative to
    its parent — see messages.proto's Link comment — so a segment of
    length L that should rotate WITH a joint has to be the OFFSET of the
    NEXT link, not that joint's own position):
      - "shoulder": revolute about +Z, at zero offset from the (fixed)
        base — this is joint 0.
      - "elbow": revolute about +Z, offset L1 along the shoulder's local
        +X — its own position is the tip of segment 1, and it is joint 1.
      - "tip": a "fixed" joint offset L2 along the elbow's local +X — the
        end effector, rigidly extending segment 2 with no extra DOF.
    Feeding joint angles (t0, t1, 0.0) through this chain reproduces the
    standard 2-link arm FK formula exactly (verified in the test below).
    """
    return Chain(
        links=[
            Link(id="shoulder", shape=_TINY, mass=1.0, parent_index=-1, joint_type="revolute",
                 joint_axis={"x": 0, "y": 0, "z": 1}),
            Link(id="elbow", shape=_TINY, mass=1.0, parent_index=0, joint_type="revolute",
                 joint_axis={"x": 0, "y": 0, "z": 1}, position={"x": l1, "y": 0, "z": 0}),
            Link(id="tip", shape=_TINY, mass=1.0, parent_index=1, joint_type="fixed",
                 position={"x": l2, "y": 0, "z": 0}),
        ]
    )


def test_two_link_planar_arm_matches_textbook_fk_formula():
    """Independent oracle: the standard 2-link planar-arm forward-
    kinematics formula, derived from scratch (not by calling into
    pybullet):
        x = L1*cos(t0) + L2*cos(t0+t1)
        y = L1*sin(t0) + L2*sin(t0+t1)
    """
    l1, l2 = 1.2, 0.8
    t0, t1 = 0.3, 0.5

    chain = _planar_arm(l1, l2)
    result = forward_kinematics(
        _TestContext(), ForwardKinematicsInput(chain=chain, joint_positions=[t0, t1, 0.0])
    )
    assert result.error.code == ""
    assert len(result.links) == 3

    expected_elbow = (l1 * math.cos(t0), l1 * math.sin(t0), 0.0)
    expected_tip = (
        l1 * math.cos(t0) + l2 * math.cos(t0 + t1),
        l1 * math.sin(t0) + l2 * math.sin(t0 + t1),
        0.0,
    )
    by_id = {l.link_id: l for l in result.links}
    assert abs(by_id["shoulder"].position.x - 0.0) < 1e-9
    assert abs(by_id["elbow"].position.x - expected_elbow[0]) < 1e-9
    assert abs(by_id["elbow"].position.y - expected_elbow[1]) < 1e-9
    assert abs(by_id["tip"].position.x - expected_tip[0]) < 1e-9
    assert abs(by_id["tip"].position.y - expected_tip[1]) < 1e-9


def test_zero_joint_angles_place_links_along_local_x_axis():
    l1, l2 = 1.0, 1.0
    chain = _planar_arm(l1, l2)
    result = forward_kinematics(
        _TestContext(), ForwardKinematicsInput(chain=chain, joint_positions=[0.0, 0.0, 0.0])
    )
    assert result.error.code == ""
    by_id = {l.link_id: l for l in result.links}
    assert abs(by_id["elbow"].position.x - 1.0) < 1e-9
    assert abs(by_id["tip"].position.x - 2.0) < 1e-9


def test_wrong_length_joint_positions_returns_structured_error_not_crash():
    chain = _planar_arm(1.0, 1.0)
    result = forward_kinematics(
        _TestContext(), ForwardKinematicsInput(chain=chain, joint_positions=[0.0])
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_bad_parent_index_returns_structured_error_not_crash():
    chain = Chain(links=[
        Link(id="orphan", shape=_TINY, mass=1.0, parent_index=5, joint_type="revolute",
             joint_axis={"x": 0, "y": 0, "z": 1}),
    ])
    result = forward_kinematics(
        _TestContext(), ForwardKinematicsInput(chain=chain, joint_positions=[0.0])
    )
    assert result.error.code == "INVALID_ARGUMENT"
