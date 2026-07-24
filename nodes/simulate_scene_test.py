from gen.messages_pb2 import Body, Scene, SimulateSceneInput, Shape
from nodes.simulate_scene import simulate_scene
from nodes._test_helpers import _TestContext


def _sphere(id_, radius, mass, position, velocity=(0.0, 0.0, 0.0), restitution=0.0):
    return Body(
        id=id_,
        shape=Shape(type="sphere", radius=radius),
        mass=mass,
        position={"x": position[0], "y": position[1], "z": position[2]},
        linear_velocity={"x": velocity[0], "y": velocity[1], "z": velocity[2]},
        restitution=restitution,
    )


def test_free_fall_matches_symplectic_euler_oracle():
    """Independent oracle: a body in free fall (no ground, no collision, no
    drag) under gravity integrates with pybullet's semi-implicit
    ("symplectic") Euler scheme. That scheme has an EXACT closed-form
    recurrence, derived here from scratch (not by calling into pybullet or
    this package's own code):
        v_n = v0 + n * a * dt
        x_n = x0 + n * v0 * dt + a * dt^2 * n*(n+1) / 2
    This is a stronger check than comparing to the *continuous* free-fall
    formula (y = y0 - 1/2 g t^2), which would only match to O(dt) — the
    discrete recurrence matches to floating-point precision.
    """
    g = 9.8
    y0 = 100.0
    dt = 1.0 / 240.0
    n = 240  # 1 second of sim time

    scene = Scene(
        bodies=[_sphere("ball", radius=0.5, mass=1.0, position=(0, 0, y0))],
        gravity={"x": 0.0, "y": 0.0, "z": -g},
        timestep=dt,
        steps=n,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == ""
    assert len(result.final_bodies) == 1
    final = result.final_bodies[0]

    expected_vz = 0.0 + n * (-g) * dt
    expected_z = y0 + 0.0 + (-g) * dt * dt * (n * (n + 1) / 2)

    assert abs(final.position.z - expected_z) < 1e-9, (final.position.z, expected_z)
    assert abs(final.linear_velocity.z - expected_vz) < 1e-9
    # Sanity cross-check against the *continuous* closed form: the
    # discretization error at this dt should be small (a few cm on a 100 m
    # fall), not large — catches a wrong integrator/timestep wiring.
    continuous_z = y0 - 0.5 * g * (n * dt) ** 2
    assert abs(final.position.z - continuous_z) < 0.05


def test_free_fall_trajectory_is_recorded_when_requested():
    dt = 1.0 / 240.0
    scene = Scene(
        bodies=[_sphere("ball", radius=0.5, mass=1.0, position=(0, 0, 10))],
        gravity={"x": 0.0, "y": 0.0, "z": -9.8},
        timestep=dt,
        steps=5,
        record_trajectory=True,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == ""
    assert len(result.trajectory) == 5
    assert result.trajectory[0].step_index == 1
    assert abs(result.trajectory[0].time - dt) < 1e-12
    assert result.trajectory[-1].step_index == 5
    # Trajectory's last entry must match final_bodies exactly.
    assert result.trajectory[-1].bodies[0].position.z == result.final_bodies[0].position.z


def test_elastic_collision_conserves_momentum():
    """Independent oracle: momentum conservation (Newton's third law — the
    contact impulse two colliding bodies exert on each other is equal and
    opposite) holds for an isolated two-body system regardless of the
    details of the collision response. Unequal masses/velocities are used
    so the invariant is a non-trivial, non-zero value (m1*v1 + m2*v2 = 3.0
    both before and after), not a symmetric 0 == 0 that a broken
    implementation could satisfy by accident.
    """
    m1, m2 = 1.0, 2.0
    v1x, v2x = 3.0, 0.0
    initial_momentum = m1 * v1x + m2 * v2x

    scene = Scene(
        bodies=[
            _sphere("a", radius=0.4, mass=m1, position=(-3, 0, 0), velocity=(v1x, 0, 0), restitution=1.0),
            _sphere("b", radius=0.4, mass=m2, position=(3, 0, 0), velocity=(v2x, 0, 0), restitution=1.0),
        ],
        gravity={"x": 0.0, "y": 0.0, "z": 0.0},
        timestep=1.0 / 240.0,
        steps=500,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == ""
    by_id = {b.id: b for b in result.final_bodies}
    final_momentum = m1 * by_id["a"].linear_velocity.x + m2 * by_id["b"].linear_velocity.x
    assert abs(final_momentum - initial_momentum) < 0.02, (final_momentum, initial_momentum)
    # And they must actually have collided (a started left of b and ended
    # up slower/reversed, b picked up speed) — guards against a no-op scene.
    assert by_id["a"].linear_velocity.x < v1x
    assert by_id["b"].linear_velocity.x > v2x


def test_deterministic_repeat_invocation():
    scene = Scene(
        bodies=[
            _sphere("ball", radius=0.5, mass=1.0, position=(0, 0, 5)),
            Body(id="ground", shape=Shape(type="plane", plane_normal={"x": 0, "y": 0, "z": 1}), mass=0.0),
        ],
        gravity={"x": 0.0, "y": 0.0, "z": -9.8},
        timestep=1.0 / 240.0,
        steps=100,
    )
    r1 = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    r2 = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert r1.error.code == "" and r2.error.code == ""
    assert r1.SerializeToString() == r2.SerializeToString()


def test_invalid_shape_type_returns_structured_error_not_crash():
    scene = Scene(
        bodies=[Body(id="bad", shape=Shape(type="donut"), mass=1.0)],
        gravity={"x": 0, "y": 0, "z": -9.8},
        timestep=1.0 / 240.0,
        steps=1,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == "INVALID_SHAPE"
    assert result.error.message


def test_negative_mass_returns_structured_error_not_crash():
    scene = Scene(
        bodies=[_sphere("bad", radius=0.5, mass=-1.0, position=(0, 0, 1))],
        gravity={"x": 0, "y": 0, "z": -9.8},
        timestep=1.0 / 240.0,
        steps=1,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == "INVALID_ARGUMENT"


def test_non_finite_position_returns_structured_error_not_crash():
    scene = Scene(
        bodies=[_sphere("bad", radius=0.5, mass=1.0, position=(float("inf"), 0, 1))],
        gravity={"x": 0, "y": 0, "z": -9.8},
        timestep=1.0 / 240.0,
        steps=1,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == "NON_FINITE_VALUE"


def test_duplicate_body_id_returns_structured_error_not_crash():
    scene = Scene(
        bodies=[
            _sphere("dup", radius=0.5, mass=1.0, position=(0, 0, 1)),
            _sphere("dup", radius=0.5, mass=1.0, position=(1, 0, 1)),
        ],
        gravity={"x": 0, "y": 0, "z": -9.8},
        timestep=1.0 / 240.0,
        steps=1,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == "DUPLICATE_ID"


def test_zero_timestep_returns_structured_error_not_crash():
    scene = Scene(
        bodies=[_sphere("a", radius=0.5, mass=1.0, position=(0, 0, 1))],
        gravity={"x": 0, "y": 0, "z": -9.8},
        timestep=0.0,
        steps=1,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == "INVALID_ARGUMENT"


def test_zero_steps_returns_initial_state_unchanged():
    scene = Scene(
        bodies=[_sphere("a", radius=0.5, mass=1.0, position=(1, 2, 3))],
        gravity={"x": 0, "y": 0, "z": -9.8},
        timestep=1.0 / 240.0,
        steps=0,
    )
    result = simulate_scene(_TestContext(), SimulateSceneInput(scene=scene))
    assert result.error.code == ""
    final = result.final_bodies[0]
    assert (final.position.x, final.position.y, final.position.z) == (1, 2, 3)
