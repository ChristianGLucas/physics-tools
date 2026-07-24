import pybullet as p

from gen.messages_pb2 import ClosestPointsInput, ClosestPointsOutput
from gen.axiom_context import AxiomContext
from nodes._physics import build_scene, physics_client, point3, err, PhysicsInputError


def closest_points(ax: AxiomContext, input: ClosestPointsInput) -> ClosestPointsOutput:
    """Find the nearest pair of points between two named bodies in a scene
    (within a caller-supplied search radius) and the distance between
    them.
    """
    try:
        if input.max_distance <= 0:
            return ClosestPointsOutput(
                error=err("INVALID_ARGUMENT", "max_distance must be > 0")
            )
        with physics_client() as cid:
            ids = build_scene(cid, input.scene)
            if input.body_a_id not in ids:
                return ClosestPointsOutput(
                    error=err("NOT_FOUND", f"body_a_id {input.body_a_id!r} not found in scene")
                )
            if input.body_b_id not in ids:
                return ClosestPointsOutput(
                    error=err("NOT_FOUND", f"body_b_id {input.body_b_id!r} not found in scene")
                )

            raw = p.getClosestPoints(
                ids[input.body_a_id],
                ids[input.body_b_id],
                distance=input.max_distance,
                physicsClientId=cid,
            )
            if not raw:
                return ClosestPointsOutput(found=False)

            best = min(raw, key=lambda c: c[8])
            pt_a, pt_b, distance = best[5], best[6], best[8]
            return ClosestPointsOutput(
                found=True,
                point_on_a=point3(*pt_a),
                point_on_b=point3(*pt_b),
                distance=distance,
            )
    except PhysicsInputError as e:
        return ClosestPointsOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.ClosestPoints: unexpected failure", error=str(e))
        return ClosestPointsOutput(error=err("CLOSEST_POINTS_FAILED", str(e)))
