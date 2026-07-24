import pybullet as p

from gen.messages_pb2 import SimulateSceneInput, SimulateSceneOutput, SimulationStep
from gen.axiom_context import AxiomContext
from nodes._physics import (
    build_scene,
    configure_stepping,
    physics_client,
    read_body_state,
    err,
    PhysicsInputError,
)


def simulate_scene(ax: AxiomContext, input: SimulateSceneInput) -> SimulateSceneOutput:
    """Advance a scene of free rigid bodies under gravity for a fixed number
    of fixed-timestep steps and return each body's final position,
    orientation, linear velocity, and angular velocity (or the full
    per-step trajectory, if requested). Deterministic: connects headless
    (DIRECT), builds the scene from scratch, steps scene.steps times, reads
    out state, and disconnects — the same scene always produces
    byte-identical output.
    """
    scene = input.scene
    try:
        with physics_client() as cid:
            configure_stepping(cid, scene)
            ids = build_scene(cid, scene)
            trajectory = []
            for step_i in range(1, scene.steps + 1):
                p.stepSimulation(physicsClientId=cid)
                if scene.record_trajectory:
                    bodies = [read_body_state(cid, bid, uid) for bid, uid in ids.items()]
                    trajectory.append(
                        SimulationStep(
                            step_index=step_i,
                            time=step_i * scene.timestep,
                            bodies=bodies,
                        )
                    )
            final_bodies = [read_body_state(cid, bid, uid) for bid, uid in ids.items()]
            return SimulateSceneOutput(final_bodies=final_bodies, trajectory=trajectory)
    except PhysicsInputError as e:
        return SimulateSceneOutput(error=e.error)
    except Exception as e:  # noqa: BLE001 - never leak a raw traceback to the caller
        ax.log.error("physics-tools.SimulateScene: unexpected failure", error=str(e))
        return SimulateSceneOutput(error=err("SIMULATION_FAILED", str(e)))
