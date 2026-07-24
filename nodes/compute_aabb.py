import pybullet as p

from gen.messages_pb2 import ComputeAABBInput, ComputeAABBOutput, AABB
from gen.axiom_context import AxiomContext
from nodes._physics import build_scene, physics_client, point3, err, PhysicsInputError


def compute_aabb(ax: AxiomContext, input: ComputeAABBInput) -> ComputeAABBOutput:
    """Compute the world-space axis-aligned bounding box of one, several, or
    every body in a scene.
    """
    try:
        with physics_client() as cid:
            ids = build_scene(cid, input.scene)

            requested = list(input.body_ids) if len(input.body_ids) > 0 else list(ids.keys())
            for bid in requested:
                if bid not in ids:
                    return ComputeAABBOutput(
                        error=err("NOT_FOUND", f"body_id {bid!r} not found in scene")
                    )

            boxes = []
            for bid in requested:
                lo, hi = p.getAABB(ids[bid], physicsClientId=cid)
                boxes.append(AABB(body_id=bid, min=point3(*lo), max=point3(*hi)))
            return ComputeAABBOutput(boxes=boxes)
    except PhysicsInputError as e:
        return ComputeAABBOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.ComputeAABB: unexpected failure", error=str(e))
        return ComputeAABBOutput(error=err("AABB_FAILED", str(e)))
