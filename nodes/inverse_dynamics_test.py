from gen.messages_pb2 import Chain, InverseDynamicsInput, Link, Shape
from nodes.inverse_dynamics import inverse_dynamics
from nodes._test_helpers import _TestContext

G = 9.8


def _pendulum(l, m):
    """A single revolute joint at the pivot (negligible mass, so it
    contributes ~0 torque about its own axis — see messages.proto's Link
    comment) plus a rigidly-attached ("fixed" joint) child link at offset
    `l` that carries the real mass `m`. This is the standard way to model
    a point mass at a distance from a joint given that a link's own mass
    sits at its own joint frame."""
    return Chain(links=[
        Link(id="joint", shape=Shape(type="sphere", radius=1e-6), mass=1e-9,
             parent_index=-1, joint_type="revolute", joint_axis={"x": 0, "y": 1, "z": 0}),
        Link(id="bob", shape=Shape(type="sphere", radius=0.05), mass=m,
             parent_index=0, joint_type="fixed", position={"x": l, "y": 0, "z": 0}),
    ])


def test_horizontal_pendulum_matches_gravity_torque_formula():
    """Independent oracle: for a point mass m at distance l from a
    revolute joint with axis (0,1,0), position r=(l,0,0), under gravity
    F=(0,0,-m*g), the torque gravity itself exerts about the joint is
    r x F = (0, m*g*l, 0) (right-hand rule, computed from scratch here).
    To hold the arm static (zero requested acceleration), Newton's second
    law for rotation (tau_applied + tau_gravity = I*alpha = 0) requires
    tau_applied = -m*g*l about that axis — the exact signed value this
    test checks against InverseDynamics's output.
    """
    l, m = 1.0, 2.0
    chain = _pendulum(l, m)

    result = inverse_dynamics(
        _TestContext(),
        InverseDynamicsInput(
            chain=chain,
            joint_positions=[0.0, 0.0],   # horizontal: arm along local +X
            joint_velocities=[0.0, 0.0],
            joint_accelerations=[0.0, 0.0],  # hold static
            gravity={"x": 0.0, "y": 0.0, "z": -G},
        ),
    )
    assert result.error.code == ""
    assert len(result.joint_torques) == 2
    expected_torque = -m * G * l
    assert abs(result.joint_torques[0] - expected_torque) < 1e-3
    # The "bob" link's own joint is fixed — always reports 0 torque.
    assert result.joint_torques[1] == 0.0


def test_vertical_pendulum_needs_no_gravity_compensating_torque():
    """At 90 degrees the bob hangs (or points) directly along the gravity
    axis, so its moment arm relative to the joint's rotation is zero and
    no torque is needed to hold it static — an independent closed-form
    check (r parallel to F => r x F = 0), not tied to pybullet internals.
    """
    l, m = 1.0, 2.0
    chain = _pendulum(l, m)
    result = inverse_dynamics(
        _TestContext(),
        InverseDynamicsInput(
            chain=chain,
            joint_positions=[1.5707963267948966, 0.0],  # pi/2: arm along local +Z (aligned with gravity)
            joint_velocities=[0.0, 0.0],
            joint_accelerations=[0.0, 0.0],
            gravity={"x": 0.0, "y": 0.0, "z": -G},
        ),
    )
    assert result.error.code == ""
    assert abs(result.joint_torques[0]) < 1e-3


def test_unset_gravity_means_explicit_zero_gravity_not_earth_default():
    """Locks in a deliberate design choice: InverseDynamicsInput.gravity
    has NO implicit Earth-gravity default (unlike some physics APIs) —
    an unset (all-zero) gravity field means a genuine zero-gravity chain,
    exactly like Scene.gravity. So the same horizontal pendulum that
    needs -m*g*l of torque under Earth gravity needs ~0 torque here.
    """
    l, m = 1.0, 2.0
    chain = _pendulum(l, m)
    result = inverse_dynamics(
        _TestContext(),
        InverseDynamicsInput(
            chain=chain,
            joint_positions=[0.0, 0.0],
            joint_velocities=[0.0, 0.0],
            joint_accelerations=[0.0, 0.0],
            # gravity intentionally omitted
        ),
    )
    assert result.error.code == ""
    assert abs(result.joint_torques[0]) < 1e-6


def test_wrong_length_arrays_return_structured_error_not_crash():
    chain = _pendulum(1.0, 1.0)
    result = inverse_dynamics(
        _TestContext(),
        InverseDynamicsInput(chain=chain, joint_positions=[0.0], joint_velocities=[0.0, 0.0],
                              joint_accelerations=[0.0, 0.0]),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_negative_mass_returns_structured_error_not_crash():
    tiny = Shape(type="sphere", radius=0.05)
    chain = Chain(links=[
        Link(id="bad", shape=tiny, mass=-1.0, parent_index=-1, joint_type="revolute",
             joint_axis={"x": 0, "y": 1, "z": 0}),
    ])
    result = inverse_dynamics(
        _TestContext(),
        InverseDynamicsInput(chain=chain, joint_positions=[0.0], joint_velocities=[0.0],
                              joint_accelerations=[0.0]),
    )
    assert result.error.code == "INVALID_ARGUMENT"
