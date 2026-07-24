# physics-tools

Deterministic, offline rigid-body physics for AI agents: simulate a caller-supplied scene of free rigid bodies under gravity for N fixed-timestep steps, cast rays, detect contacts and penetration depth, find the closest points between two bodies, compute axis-aligned bounding boxes, and solve inverse/forward kinematics and inverse dynamics on an inline articulated chain — built for the [Axiom](https://axiomide.com) marketplace.

## Use it from your agent or app

Every node in this package is a **live, auto-scaling API endpoint** on the
[Axiom](https://axiomide.com) marketplace — call it from an AI agent or your
own code, with nothing to self-host.

**📦 See it on the marketplace:**
https://dev.axiomide.com/marketplace/christiangeorgelucas/physics-tools@0.1.0

**Hook it up to an AI agent (MCP).** Add Axiom's hosted MCP server to any MCP
client and every node becomes a typed tool your agent can call — search the
catalog, inspect a schema, and invoke it directly.

```bash
# Claude Code
claude mcp add --transport http axiom https://api.axiomide.com/mcp \
  --header "Authorization: Bearer $AXIOM_API_KEY"
```

Claude Desktop, Cursor, or any config-based client:

```json
{
  "mcpServers": {
    "axiom": {
      "type": "http",
      "url": "https://api.axiomide.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_AXIOM_API_KEY" }
    }
  }
}
```

**Call it from the CLI.**

```bash
axiom invoke christiangeorgelucas/physics-tools/SimulateScene --input '{"scene":{"bodies":[{"id":"ball","shape":{"type":"sphere","radius":0.5},"mass":1.0,"position":{"x":0,"y":0,"z":5}},{"id":"ground","shape":{"type":"plane","planeNormal":{"x":0,"y":0,"z":1}},"mass":0}],"gravity":{"x":0,"y":0,"z":-9.8},"timestep":0.0041666667,"steps":240}}'
```

**Call it over HTTP.**

```bash
curl -X POST https://api.axiomide.com/invocations/v1/nodes/christiangeorgelucas/physics-tools/0.1.0/SimulateScene \
  -H "Authorization: Bearer $AXIOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"scene":{"bodies":[{"id":"ball","shape":{"type":"sphere","radius":0.5},"mass":1.0,"position":{"x":0,"y":0,"z":5}},{"id":"ground","shape":{"type":"plane","planeNormal":{"x":0,"y":0,"z":1}},"mass":0}],"gravity":{"x":0,"y":0,"z":-9.8},"timestep":0.0041666667,"steps":240}}'
```

### Get started free

Install the CLI:

```bash
# macOS / Linux — Homebrew
brew install axiomide/tap/axiom

# macOS / Linux — install script
curl -fsSL https://raw.githubusercontent.com/AxiomIDE/axiom-releases/main/install.sh | sh
```

**Windows:** download the `windows/amd64` `.zip` from the
[releases page](https://github.com/AxiomIDE/axiom-releases/releases), unzip
it, and put `axiom.exe` on your `PATH`.

Then `axiom version` to verify, `axiom login` (GitHub or Google) to
authenticate, and create an API key under **Console → API Keys**. Docs and
sign-up at **[axiomide.com](https://axiomide.com)**.

## Why this package

[pybullet](https://github.com/bulletphysics/bullet3) is the official Python
interface to the Bullet Physics SDK (zlib license) — the physics engine
behind countless robotics, RL, and simulation projects. This package wraps
it as a set of **pure, stateless, deterministic functions**: every node
connects to a fresh headless (`DIRECT` mode) pybullet client, builds the
scene or articulated chain from the request, computes, reads the result
back out, and disconnects. No process-level state ever survives between
calls. Every body simulates with zero linear/angular damping, so undamped
closed-form physics (free fall, projectile motion, elastic-collision
momentum conservation) applies directly and is what this package's test
suite checks each node against.

## Nodes

- **SimulateScene** — advance a scene of free rigid bodies under gravity
  for a fixed number of fixed-timestep steps; returns each body's final
  (or full per-step trajectory) position, orientation, and velocities.
- **RayCast** — cast one or more rays against a scene of placed (not
  simulated) bodies; returns hit body, point, normal, and fraction per ray.
- **ContactPoints** — detect collisions between placed bodies; returns
  contact points, normal, and separation distance (negative = penetration
  depth). Reports every touching/overlapping pair regardless of mass,
  including two static bodies — unlike pybullet's own contact-detection
  pipeline, which silently skips static-static pairs.
- **ClosestPoints** — nearest points and distance between two named
  bodies, within a caller-supplied search radius.
- **ComputeAABB** — world-space axis-aligned bounding box of one, several,
  or every body in a scene.
- **InverseKinematics** — solve joint values of an inline articulated
  chain that bring a chosen end-effector link close to a target
  position/orientation.
- **ForwardKinematics** — given a chain and every joint's value, compute
  the resulting world-space pose of every link.
- **InverseDynamics** — given a chain, joint positions/velocities/desired
  accelerations, and gravity, compute the joint torques/forces required
  (Recursive Newton-Euler).

Shapes: box, sphere, capsule, cylinder, plane, and convex mesh (from
inline vertices). Chains are always built inline from caller-supplied
links — **no URDF, mesh file, or URL is ever loaded**, so a node can never
be made to dereference a path or fetch a remote resource.

Two real modeling notes carried over from the test suite (see
`messages/messages.proto` for the full detail):

- A **link's own mass sits at its own joint frame** — to model a mass at a
  distance from a joint (a pendulum bob, an end-effector payload), give
  the joint its own near-massless link and attach a child link with
  `joint_type: "fixed"` at the desired offset carrying the real mass.
- A **joint's own position is fixed relative to its parent**, independent
  of that joint's own rotation — a segment that should rotate *with* a
  joint has to be the *next* link's position offset, not that joint's own.
  A textbook 2-segment arm is therefore 3 links (joint, joint, fixed tip).

## Out of scope

- **Soft-body / cloth / deformable simulation** — pybullet supports this,
  but it is a materially different (and much larger) domain; a dedicated
  package would do it justice.
- **GUI / rendering** — not applicable to a headless server-side node.
- **Loading a URDF/mesh by file path or URL** — capability-safety: a node
  never dereferences caller-controlled paths or fetches remote resources.
  Every shape and chain is specified inline in the request.
- **Forward dynamics as a standalone node** (torque → resulting
  acceleration) — pybullet does not expose a single-call pure-function
  form of this the way it does `calculateInverseDynamics`; producing it
  requires time-stepping under motor control, which is really a variant
  of `SimulateScene`, not a distinct clean primitive.

## License

MIT. Wraps [pybullet](https://github.com/bulletphysics/bullet3) (zlib
license) and [numpy](https://numpy.org) (BSD-3-Clause) — both fully
permissive, no copyleft anywhere in the dependency tree.
