import math

from gen.messages_pb2 import Chain, InverseKinematicsInput, Link, Shape
from nodes.inverse_kinematics import inverse_kinematics
from nodes._test_helpers import _TestContext

_TINY = Shape(type="sphere", radius=0.01)


def _planar_arm(l1, l2):
    """Same 3-link textbook-2-segment-arm construction as
    forward_kinematics_test.py — see that file's _planar_arm docstring
    for why a genuine 2-segment arm needs 3 links in this package's Chain
    model (a joint's own position is fixed relative to its parent; the
    segment that rotates WITH a joint has to be the next link's offset)."""
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


def _textbook_fk(l1, l2, t0, t1):
    """Same from-scratch 2-link planar FK formula used in
    forward_kinematics_test.py — the independent oracle for this test."""
    return (
        l1 * math.cos(t0) + l2 * math.cos(t0 + t1),
        l1 * math.sin(t0) + l2 * math.sin(t0 + t1),
        0.0,
    )


def test_ik_solution_reaches_an_exactly_reachable_target():
    """Pick a target that is EXACTLY reachable (it's the textbook-FK image
    of a known, hand-picked joint configuration), then check the solver
    actually gets there and that the returned joint angles are internally
    consistent with the from-scratch FK formula."""
    l1, l2 = 1.0, 1.0
    true_t0, true_t1 = 0.4, 0.6
    target = _textbook_fk(l1, l2, true_t0, true_t1)

    chain = _planar_arm(l1, l2)
    result = inverse_kinematics(
        _TestContext(),
        InverseKinematicsInput(
            chain=chain,
            end_effector_link_index=2,  # "tip"
            target_position={"x": target[0], "y": target[1], "z": target[2]},
        ),
    )
    assert result.error.code == ""
    assert len(result.joint_positions) == 3

    # The solver reached the (exactly reachable) target closely. pybullet's
    # default iteration count/residual threshold for calculateInverseKinematics
    # converges to within a few cm, not machine precision — 0.05 gives
    # headroom above the ~0.014 residual observed for this fixture while
    # still catching a solver that's not converging at all (e.g. off by
    # a meter, or stuck at the initial all-zero pose).
    assert abs(result.achieved_position.x - target[0]) < 0.05
    assert abs(result.achieved_position.y - target[1]) < 0.05

    # Independent cross-check: feeding the RETURNED shoulder/elbow angles
    # through the from-scratch textbook FK formula must land at the
    # reported achieved_position — validates the node's FK bookkeeping
    # independently of whatever pybullet did internally to solve the IK.
    recomputed = _textbook_fk(l1, l2, result.joint_positions[0], result.joint_positions[1])
    assert abs(recomputed[0] - result.achieved_position.x) < 1e-6
    assert abs(recomputed[1] - result.achieved_position.y) < 1e-6
    # "tip" is a fixed joint: always reports 0.
    assert result.joint_positions[2] == 0.0


def test_out_of_range_end_effector_index_returns_structured_error_not_crash():
    chain = _planar_arm(1.0, 1.0)
    result = inverse_kinematics(
        _TestContext(),
        InverseKinematicsInput(chain=chain, end_effector_link_index=99, target_position={"x": 1, "y": 0, "z": 0}),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_invalid_joint_type_returns_structured_error_not_crash():
    chain = Chain(links=[
        Link(id="bad", shape=_TINY, mass=1.0, parent_index=-1, joint_type="hinge",
             joint_axis={"x": 0, "y": 0, "z": 1}),
    ])
    result = inverse_kinematics(
        _TestContext(),
        InverseKinematicsInput(chain=chain, end_effector_link_index=0, target_position={"x": 1, "y": 0, "z": 0}),
    )
    assert result.error.code == "INVALID_JOINT"
