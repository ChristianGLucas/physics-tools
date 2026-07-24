import pybullet as p

from gen.messages_pb2 import ForwardKinematicsInput, ForwardKinematicsOutput, LinkPose
from gen.axiom_context import AxiomContext
from nodes._physics import build_chain, physics_client, point3, quat, err, PhysicsInputError


def forward_kinematics(ax: AxiomContext, input: ForwardKinematicsInput) -> ForwardKinematicsOutput:
    """Given an inline-defined articulated chain and a joint value for
    every link, compute the resulting world-space position and
    orientation of every link.
    """
    try:
        chain = input.chain
        n_links = len(chain.links)
        joint_positions = list(input.joint_positions)
        if len(joint_positions) != n_links:
            return ForwardKinematicsOutput(
                error=err(
                    "INVALID_ARGUMENT",
                    f"joint_positions must have exactly {n_links} values (one per "
                    f"chain.links entry, 0 for fixed joints), got {len(joint_positions)}",
                )
            )

        with physics_client() as cid:
            uid, _movable = build_chain(cid, chain)

            for i, value in enumerate(joint_positions):
                # resetJointState is a documented no-op on a fixed joint, so
                # this is safe to call unconditionally in chain.links order.
                p.resetJointState(uid, i, value, physicsClientId=cid)

            links = []
            for i, link in enumerate(chain.links):
                state = p.getLinkState(uid, i, computeForwardKinematics=1, physicsClientId=cid)
                pos, orn = state[0], state[1]
                links.append(LinkPose(link_id=link.id, position=point3(*pos), orientation=quat(*orn)))

            return ForwardKinematicsOutput(links=links)
    except PhysicsInputError as e:
        return ForwardKinematicsOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.ForwardKinematics: unexpected failure", error=str(e))
        return ForwardKinematicsOutput(error=err("FK_FAILED", str(e)))
