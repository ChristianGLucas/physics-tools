"""Shared pybullet plumbing for christiangeorgelucas/physics-tools.

Every node follows the same pure-function pattern:
  1. connect(pybullet.DIRECT)   — headless, no GUI, no rendering
  2. validate the input, build the scene/chain from it
  3. run (step / query)
  4. read out results
  5. disconnect()

A fresh client per call (never reused across invocations) plus a fixed
timestep, a fixed solver-iteration count, and zero damping on every body
is what makes each call deterministic: the same input always produces
byte-identical output, whether re-run in the same process or a fresh one.

Every pybullet call below is passed its `physicsClientId` explicitly. That
is not cosmetic: Axiom's Python runtime can reuse a warm worker process
across many invocations, and pybullet's "no physicsClientId given" default
resolves to a fixed client slot (0), not "whichever client this call just
connected" — so two invocations that overlap (or simply run back-to-back
in the same process without a fully torn-down previous client) could
silently read/write each other's simulation state if any call omitted the
id. Threading the id through every call is what keeps each invocation
correctly isolated regardless of process reuse.

Nothing here loads a URDF, a mesh file, or a URL — every shape is built
from inline vertices/dimensions the caller supplied directly in the
request message. That is a deliberate capability-safety choice: a caller
can never make a node dereference a path or fetch a remote resource.

No size/step-count/DoS caps live here. That is the platform's job, not
this node's — see the package README and the seeding skill this package
was built under.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import pybullet as p

from gen.messages_pb2 import Error

_JOINT_TYPES = {
    "revolute": p.JOINT_REVOLUTE,
    "prismatic": p.JOINT_PRISMATIC,
    "fixed": p.JOINT_FIXED,
}

_SHAPE_TYPES = {"box", "sphere", "capsule", "cylinder", "plane", "mesh"}

DEFAULT_SOLVER_ITERATIONS = 50


def err(code: str, message: str) -> Error:
    return Error(code=code, message=message)


class PhysicsInputError(Exception):
    """Raised by every validation/build helper below to short-circuit a
    node with a structured Error. Every node's top-level try/except
    catches this (and, as a fallback, any other exception) and returns it
    as the output message's `error` field rather than letting it
    propagate as a raw crash/traceback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.error = err(code, message)


def _finite(*values):
    for v in values:
        if not math.isfinite(v):
            raise PhysicsInputError("NON_FINITE_VALUE", f"expected a finite number, got {v!r}")


def to_vec3(msg, name):
    _finite(msg.x, msg.y, msg.z)
    return (msg.x, msg.y, msg.z)


def to_quat(msg, name):
    x, y, z, w = msg.x, msg.y, msg.z, msg.w
    _finite(x, y, z, w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)  # identity
    return (x / norm, y / norm, z / norm, w / norm)


def build_collision_shape(cid, shape, *, for_link=False):
    """Create a pybullet collision shape from a Shape message on client
    `cid`. Returns the collision shape unique id. Raises PhysicsInputError
    on any invalid shape definition — never hands malformed geometry to
    the native engine."""
    t = (shape.type or "").strip().lower()
    if t not in _SHAPE_TYPES:
        raise PhysicsInputError(
            "INVALID_SHAPE", f"shape.type must be one of {sorted(_SHAPE_TYPES)}, got {shape.type!r}"
        )

    if t == "box":
        hx, hy, hz = to_vec3(shape.half_extents, "half_extents")
        if hx <= 0 or hy <= 0 or hz <= 0:
            raise PhysicsInputError("INVALID_SHAPE", "box shape.half_extents must all be > 0")
        return p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cid)

    if t == "sphere":
        _finite(shape.radius)
        if shape.radius <= 0:
            raise PhysicsInputError("INVALID_SHAPE", "sphere shape.radius must be > 0")
        return p.createCollisionShape(p.GEOM_SPHERE, radius=shape.radius, physicsClientId=cid)

    if t == "capsule":
        _finite(shape.radius, shape.height)
        if shape.radius <= 0 or shape.height <= 0:
            raise PhysicsInputError(
                "INVALID_SHAPE", "capsule shape.radius and shape.height must be > 0"
            )
        return p.createCollisionShape(
            p.GEOM_CAPSULE, radius=shape.radius, height=shape.height, physicsClientId=cid
        )

    if t == "cylinder":
        _finite(shape.radius, shape.height)
        if shape.radius <= 0 or shape.height <= 0:
            raise PhysicsInputError(
                "INVALID_SHAPE", "cylinder shape.radius and shape.height must be > 0"
            )
        return p.createCollisionShape(
            p.GEOM_CYLINDER, radius=shape.radius, height=shape.height, physicsClientId=cid
        )

    if t == "plane":
        if for_link:
            raise PhysicsInputError(
                "INVALID_SHAPE", "plane shapes are not supported on articulated-chain links"
            )
        normal = _unit_plane_normal(shape)
        _finite(shape.plane_constant)
        # pybullet's createCollisionShape does not accept a planeConstant
        # kwarg for GEOM_PLANE (it always creates a plane through its own
        # local origin) — the offset is applied in build_body instead, by
        # translating the body along the normal (see plane_position_offset).
        return p.createCollisionShape(p.GEOM_PLANE, planeNormal=normal, physicsClientId=cid)

    if t == "mesh":
        verts = list(shape.mesh_vertices)
        if len(verts) < 4:
            raise PhysicsInputError("INVALID_SHAPE", "mesh shape.mesh_vertices needs at least 4 points")
        pts = []
        for v in verts:
            _finite(v.x, v.y, v.z)
            pts.append([v.x, v.y, v.z])
        return p.createCollisionShape(p.GEOM_MESH, vertices=pts, physicsClientId=cid)

    raise PhysicsInputError("INVALID_SHAPE", f"unhandled shape type {t!r}")  # pragma: no cover


def _unit_plane_normal(shape):
    nx, ny, nz = to_vec3(shape.plane_normal, "plane_normal")
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-12:
        return [0.0, 0.0, 1.0]
    return [nx / norm, ny / norm, nz / norm]


def plane_position_offset(shape):
    """A plane collision shape always passes through its own local origin
    — pybullet has no plane-constant kwarg to shift it. To honor
    shape.plane_constant (the signed distance of the plane from the
    origin along its normal), the offset is instead added to the body's
    basePosition: translating a plane by `normal * constant` along its
    own normal produces exactly the plane {x : dot(x, normal) = constant},
    since the local-frame plane equation dot(x_local, normal) = 0 becomes
    dot(x_world - offset, normal) = 0 => dot(x_world, normal) =
    dot(offset, normal) = constant (normal is unit length). Returns
    (0, 0, 0) for every non-plane shape."""
    if (shape.type or "").strip().lower() != "plane":
        return (0.0, 0.0, 0.0)
    _finite(shape.plane_constant)
    nx, ny, nz = _unit_plane_normal(shape)
    c = shape.plane_constant
    return (nx * c, ny * c, nz * c)


def build_body(cid, body):
    """Create a free rigid body (a pybullet multi-body with no links) from
    a Body message on client `cid`. Returns the pybullet body unique id."""
    if not body.id:
        raise PhysicsInputError("INVALID_ARGUMENT", "every body must have a non-empty id")
    _finite(body.mass)
    if body.mass < 0:
        raise PhysicsInputError("INVALID_ARGUMENT", f"body {body.id!r}: mass must be >= 0")

    col = build_collision_shape(cid, body.shape)
    pos = to_vec3(body.position, "position")
    offset = plane_position_offset(body.shape)
    pos = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
    orn = to_quat(body.orientation, "orientation")

    uid = p.createMultiBody(
        baseMass=body.mass,
        baseCollisionShapeIndex=col,
        basePosition=list(pos),
        baseOrientation=list(orn),
        physicsClientId=cid,
    )

    lin_v = to_vec3(body.linear_velocity, "linear_velocity")
    ang_v = to_vec3(body.angular_velocity, "angular_velocity")
    p.resetBaseVelocity(
        uid, linearVelocity=list(lin_v), angularVelocity=list(ang_v), physicsClientId=cid
    )

    _finite(body.restitution, body.friction)
    p.changeDynamics(
        uid,
        -1,
        restitution=body.restitution,
        lateralFriction=body.friction,
        linearDamping=0.0,
        angularDamping=0.0,
        physicsClientId=cid,
    )
    return uid


def build_scene(cid, scene):
    """Populate client `cid` with every body in a Scene message and set
    gravity. Returns {body_id: pybullet_uid}. Does NOT step the
    simulation and does NOT touch scene.timestep/steps/solver_iterations
    — those are meaningful only to a node that actually advances time
    (SimulateScene), which validates and applies them itself via
    configure_stepping() below. The read-only query nodes (RayCast,
    ContactPoints, ClosestPoints, ComputeAABB) call this directly and must
    not be gated on an unrelated field they never use."""
    gx, gy, gz = to_vec3(scene.gravity, "gravity")
    p.setGravity(gx, gy, gz, physicsClientId=cid)

    ids = {}
    for body in scene.bodies:
        if body.id in ids:
            raise PhysicsInputError("DUPLICATE_ID", f"duplicate body id {body.id!r}")
        ids[body.id] = build_body(cid, body)
    return ids


def configure_stepping(cid, scene):
    """Validate and apply scene.timestep/steps/solver_iterations — called
    only by SimulateScene, the one node that actually advances time."""
    if scene.timestep <= 0 or not math.isfinite(scene.timestep):
        raise PhysicsInputError(
            "INVALID_ARGUMENT", f"scene.timestep must be > 0, got {scene.timestep!r}"
        )
    if scene.steps < 0:
        raise PhysicsInputError("INVALID_ARGUMENT", f"scene.steps must be >= 0, got {scene.steps!r}")

    iterations = scene.solver_iterations if scene.solver_iterations > 0 else DEFAULT_SOLVER_ITERATIONS
    p.setPhysicsEngineParameter(
        fixedTimeStep=scene.timestep,
        numSolverIterations=iterations,
        numSubSteps=0,
        physicsClientId=cid,
    )


def build_chain(cid, chain):
    """Create an articulated multi-body from a Chain message on client
    `cid` (fixed base plus an ordered list of links, each attached to an
    earlier link or the base by one joint). Returns
    (pybullet_body_uid, movable_link_indices) where movable_link_indices
    lists the 0-based indices into chain.links whose joint_type is not
    "fixed", in order — pybullet's IK/inverse-dynamics calls take one
    value per movable degree of freedom only (fixed joints excluded), not
    one per link, so callers built on top of this need that mapping.
    Every link's joint is set to zero damping so kinematics/dynamics
    queries are not perturbed by hidden defaults."""
    if len(chain.links) == 0:
        raise PhysicsInputError("INVALID_ARGUMENT", "chain.links must contain at least one link")

    base_pos = to_vec3(chain.base_position, "base_position")
    base_orn = to_quat(chain.base_orientation, "base_orientation")
    base_col = p.createCollisionShape(p.GEOM_SPHERE, radius=1e-6, physicsClientId=cid)

    link_masses = []
    link_collision = []
    link_visual = []
    link_positions = []
    link_orientations = []
    link_inertial_pos = []
    link_inertial_orn = []
    link_parent = []
    link_joint_type = []
    link_joint_axis = []
    movable_link_indices = []

    seen_ids = set()
    for i, link in enumerate(chain.links):
        if not link.id:
            raise PhysicsInputError("INVALID_ARGUMENT", f"link at index {i} must have a non-empty id")
        if link.id in seen_ids:
            raise PhysicsInputError("DUPLICATE_ID", f"duplicate link id {link.id!r}")
        seen_ids.add(link.id)

        _finite(link.mass)
        if link.mass < 0:
            raise PhysicsInputError("INVALID_ARGUMENT", f"link {link.id!r}: mass must be >= 0")

        if link.parent_index < -1 or link.parent_index >= i:
            raise PhysicsInputError(
                "INVALID_ARGUMENT",
                f"link {link.id!r}: parent_index must be -1 or reference an earlier "
                f"link (< {i}), got {link.parent_index}",
            )

        jt = (link.joint_type or "").strip().lower()
        if jt not in _JOINT_TYPES:
            raise PhysicsInputError(
                "INVALID_JOINT",
                f"link {link.id!r}: joint_type must be one of {sorted(_JOINT_TYPES)}, "
                f"got {link.joint_type!r}",
            )

        col = build_collision_shape(cid, link.shape, for_link=True)
        pos = to_vec3(link.position, "position")
        orn = to_quat(link.orientation, "orientation")
        axis = to_vec3(link.joint_axis, "joint_axis")
        if jt != "fixed":
            axis_norm = math.sqrt(sum(c * c for c in axis))
            if axis_norm < 1e-12:
                raise PhysicsInputError(
                    "INVALID_JOINT",
                    f"link {link.id!r}: joint_axis must be non-zero for joint_type {jt!r}",
                )

        link_masses.append(link.mass)
        link_collision.append(col)
        link_visual.append(-1)
        link_positions.append(list(pos))
        link_orientations.append(list(orn))
        link_inertial_pos.append([0.0, 0.0, 0.0])
        link_inertial_orn.append([0.0, 0.0, 0.0, 1.0])
        # pybullet's parent-index convention: 0 = base, N = the N-th link (1-based).
        link_parent.append(0 if link.parent_index == -1 else link.parent_index + 1)
        link_joint_type.append(_JOINT_TYPES[jt])
        link_joint_axis.append(list(axis))
        if jt != "fixed":
            movable_link_indices.append(i)

    uid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=base_col,
        basePosition=list(base_pos),
        baseOrientation=list(base_orn),
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision,
        linkVisualShapeIndices=link_visual,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_pos,
        linkInertialFrameOrientations=link_inertial_orn,
        linkParentIndices=link_parent,
        linkJointTypes=link_joint_type,
        linkJointAxis=link_joint_axis,
        physicsClientId=cid,
    )

    for i in range(len(chain.links)):
        p.changeDynamics(
            uid, i, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0, physicsClientId=cid
        )

    return uid, movable_link_indices


def point3(x, y, z):
    return {"x": x, "y": y, "z": z}


def quat(x, y, z, w):
    return {"x": x, "y": y, "z": z, "w": w}


def read_body_state(cid, body_id, uid):
    """Read a free rigid body's current world-space state back out of
    client `cid` as a BodyState-shaped kwargs dict."""
    from gen.messages_pb2 import BodyState

    pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=cid)
    lin_v, ang_v = p.getBaseVelocity(uid, physicsClientId=cid)
    return BodyState(
        id=body_id,
        position=point3(*pos),
        orientation=quat(*orn),
        linear_velocity=point3(*lin_v),
        angular_velocity=point3(*ang_v),
    )


@contextmanager
def physics_client():
    """A headless DIRECT-mode pybullet client, guaranteed to disconnect
    even if the body raises. Yields the client id — every subsequent
    pybullet call in the node body must pass physicsClientId=<that id>
    explicitly (see module docstring for why)."""
    cid = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=cid)
        yield cid
    finally:
        p.disconnect(physicsClientId=cid)
