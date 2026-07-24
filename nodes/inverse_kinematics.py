import pybullet as p

from gen.messages_pb2 import InverseKinematicsInput, InverseKinematicsOutput
from gen.axiom_context import AxiomContext
from nodes._physics import build_chain, physics_client, point3, to_vec3, to_quat, err, PhysicsInputError


def inverse_kinematics(ax: AxiomContext, input: InverseKinematicsInput) -> InverseKinematicsOutput:
    """Solve for the joint values of an inline-defined articulated chain
    (built entirely from caller-supplied links; no URDF/file/URL) that
    bring a chosen end-effector link as close as possible to a target
    position (and, optionally, orientation).
    """
    try:
        chain = input.chain
        n_links = len(chain.links)
        if input.end_effector_link_index < 0 or input.end_effector_link_index >= n_links:
            return InverseKinematicsOutput(
                error=err(
                    "INVALID_ARGUMENT",
                    f"end_effector_link_index must be in [0, {n_links}), "
                    f"got {input.end_effector_link_index}",
                )
            )

        with physics_client() as cid:
            uid, movable = build_chain(cid, chain)

            target_pos = to_vec3(input.target_position, "target_position")
            kwargs = {}
            if input.use_orientation:
                kwargs["targetOrientation"] = list(to_quat(input.target_orientation, "target_orientation"))

            solution = p.calculateInverseKinematics(
                uid,
                input.end_effector_link_index,
                list(target_pos),
                physicsClientId=cid,
                **kwargs,
            )

            # `solution` has one value per movable DOF only (fixed joints
            # excluded); expand back out to one value per chain.links entry.
            joint_positions = [0.0] * n_links
            for dof_value, link_index in zip(solution, movable):
                joint_positions[link_index] = dof_value
                p.resetJointState(uid, link_index, dof_value, physicsClientId=cid)

            state = p.getLinkState(
                uid, input.end_effector_link_index, computeForwardKinematics=1, physicsClientId=cid
            )
            achieved_position = state[0]

            return InverseKinematicsOutput(
                joint_positions=joint_positions,
                achieved_position=point3(*achieved_position),
            )
    except PhysicsInputError as e:
        return InverseKinematicsOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.InverseKinematics: unexpected failure", error=str(e))
        return InverseKinematicsOutput(error=err("IK_FAILED", str(e)))
