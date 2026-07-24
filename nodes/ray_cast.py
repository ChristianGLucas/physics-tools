import pybullet as p

from gen.messages_pb2 import RayCastInput, RayCastOutput, RayHit
from gen.axiom_context import AxiomContext
from nodes._physics import build_scene, physics_client, point3, err, PhysicsInputError


def ray_cast(ax: AxiomContext, input: RayCastInput) -> RayCastOutput:
    """Cast one or more rays against a scene of bodies placed exactly as
    given (not simulated), returning for each ray whether it hit, which
    body, the hit point and surface normal, and the hit fraction along the
    ray.
    """
    try:
        with physics_client() as cid:
            ids = build_scene(cid, input.scene)
            uid_to_id = {uid: bid for bid, uid in ids.items()}

            if len(input.rays) == 0:
                return RayCastOutput(hits=[])

            froms = [[r.from_point.x, r.from_point.y, r.from_point.z] for r in input.rays]
            tos = [[r.to_point.x, r.to_point.y, r.to_point.z] for r in input.rays]
            results = p.rayTestBatch(froms, tos, physicsClientId=cid)

            hits = []
            for ray, result in zip(input.rays, results):
                obj_uid, link_index, fraction, hit_pos, hit_normal = result
                if obj_uid < 0:
                    hits.append(RayHit(ray_id=ray.id, hit=False))
                    continue
                hits.append(
                    RayHit(
                        ray_id=ray.id,
                        hit=True,
                        body_id=uid_to_id.get(obj_uid, ""),
                        point=point3(*hit_pos),
                        normal=point3(*hit_normal),
                        fraction=fraction,
                    )
                )
            return RayCastOutput(hits=hits)
    except PhysicsInputError as e:
        return RayCastOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.RayCast: unexpected failure", error=str(e))
        return RayCastOutput(error=err("RAYCAST_FAILED", str(e)))
