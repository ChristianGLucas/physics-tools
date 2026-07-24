import pybullet as p

from gen.messages_pb2 import ContactPointsInput, ContactPointsOutput, Contact
from gen.axiom_context import AxiomContext
from nodes._physics import build_scene, physics_client, point3, err, PhysicsInputError

# Bullet's own contact-detection pipeline (performCollisionDetection +
# getContactPoints) silently skips any pair where BOTH bodies are static
# (mass 0) — a broadphase optimization that assumes static-static pairs
# never need a dynamics response. That is wrong for this node's contract
# (report every touching/overlapping pair regardless of mass), so contacts
# are instead computed via a pairwise getClosestPoints sweep with a tiny
# margin — which does not have that static-static blind spot — rather than
# the more obvious getContactPoints call.
_CONTACT_MARGIN = 1e-4


def contact_points(ax: AxiomContext, input: ContactPointsInput) -> ContactPointsOutput:
    """Detect collisions between the bodies in a scene placed exactly as
    given (not simulated), returning each contact's points on both bodies,
    the contact normal, and the separation distance (negative means
    penetration, and by how much).
    """
    try:
        with physics_client() as cid:
            ids = build_scene(cid, input.scene)
            body_ids = list(ids.items())

            contacts = []
            for i in range(len(body_ids)):
                for j in range(i + 1, len(body_ids)):
                    a_id, a_uid = body_ids[i]
                    b_id, b_uid = body_ids[j]
                    raw = p.getClosestPoints(
                        a_uid, b_uid, distance=_CONTACT_MARGIN, physicsClientId=cid
                    )
                    for c in raw:
                        pt_a, pt_b, normal_on_b, distance = c[5], c[6], c[7], c[8]
                        contacts.append(
                            Contact(
                                body_a_id=a_id,
                                body_b_id=b_id,
                                point_on_a=point3(*pt_a),
                                point_on_b=point3(*pt_b),
                                normal_on_b=point3(*normal_on_b),
                                distance=distance,
                            )
                        )
            return ContactPointsOutput(contacts=contacts)
    except PhysicsInputError as e:
        return ContactPointsOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.ContactPoints: unexpected failure", error=str(e))
        return ContactPointsOutput(error=err("CONTACTS_FAILED", str(e)))
