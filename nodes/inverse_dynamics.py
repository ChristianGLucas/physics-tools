import pybullet as p

from gen.messages_pb2 import InverseDynamicsInput, InverseDynamicsOutput
from gen.axiom_context import AxiomContext
from nodes._physics import build_chain, physics_client, to_vec3, err, PhysicsInputError


def _require_length(values, n_links, field_name):
    if len(values) != n_links:
        raise PhysicsInputError(
            "INVALID_ARGUMENT",
            f"{field_name} must have exactly {n_links} values (one per chain.links "
            f"entry, ignored for fixed joints), got {len(values)}",
        )


def inverse_dynamics(ax: AxiomContext, input: InverseDynamicsInput) -> InverseDynamicsOutput:
    """Given an inline-defined articulated chain and, for every joint, its
    position, velocity, and desired acceleration, compute the joint
    torques (or forces, for prismatic joints) required to produce those
    accelerations under the given gravity, via the Recursive Newton-Euler
    algorithm.
    """
    try:
        chain = input.chain
        n_links = len(chain.links)
        positions = list(input.joint_positions)
        velocities = list(input.joint_velocities)
        accelerations = list(input.joint_accelerations)
        _require_length(positions, n_links, "joint_positions")
        _require_length(velocities, n_links, "joint_velocities")
        _require_length(accelerations, n_links, "joint_accelerations")

        with physics_client() as cid:
            gx, gy, gz = to_vec3(input.gravity, "gravity")
            p.setGravity(gx, gy, gz, physicsClientId=cid)
            uid, movable = build_chain(cid, chain)

            if len(movable) == 0:
                return InverseDynamicsOutput(joint_torques=[0.0] * n_links)

            mv_pos = [positions[i] for i in movable]
            mv_vel = [velocities[i] for i in movable]
            mv_acc = [accelerations[i] for i in movable]

            torques = p.calculateInverseDynamics(uid, mv_pos, mv_vel, mv_acc, physicsClientId=cid)

            joint_torques = [0.0] * n_links
            for value, link_index in zip(torques, movable):
                joint_torques[link_index] = value

            return InverseDynamicsOutput(joint_torques=joint_torques)
    except PhysicsInputError as e:
        return InverseDynamicsOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.InverseDynamics: unexpected failure", error=str(e))
        return InverseDynamicsOutput(error=err("INVERSE_DYNAMICS_FAILED", str(e)))
