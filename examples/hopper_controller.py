"""Replay the hopping controller PySR evolved against MuJoCo rollouts.

Every candidate expression was scored by simulating the Gymnasium Hopper-v5 and
accumulating its reward, so the search optimised locomotion directly. There was
no expert policy and no imitation data. The search is a cluster job; this script
replays what it found, against the twenty held-out start states that were frozen
before the search and never scored during it.

Requires `gymnasium[mujoco]`.
"""

import gymnasium as gym
import numpy as np

HELDOUT_SEEDS = range(20261001, 20261021)
RESET_NOISE = 0.01

# Eight seconds at the Hopper's 0.008 s control step. Passing means staying
# upright for all of it, travelling five metres, and completing four
# contact/liftoff cycles.
STEPS = 1000
DISTANCE = 5.0
CYCLES = 4

# The three vector constants the search chose, one component per motor.
C0 = np.array([-0.24771142177432096, -4.22684590927177286, 8.8397066748254275])
C1 = np.array([0.06452662644513463, 0.6180986305329657, 0.3968791265257595])
C2 = np.array([0.49735344234321904, -1.1920811500492086, 1.1808251376288732])


def action(qpos, qvel):
    """The evolved controller: hopper state in, three motor commands out."""
    angles = qpos[3:6]
    rates = np.clip(qvel[3:6], -10.0, 10.0) / 10.0
    pitch = qpos[2]
    forward, vertical, spin = qvel[0] / 5.0, qvel[1] / 5.0, qvel[2] / 5.0
    rolled_angles = angles[[1, 2, 0]]
    rolled_rates = rates[[1, 2, 0]]
    d = angles - 2 * rolled_angles + rolled_rates - 2 * (forward + spin) + C0
    left = -angles - (pitch + spin * (rolled_rates - spin)) * d - C1
    right = -rolled_angles - vertical + spin + C2
    return np.tanh(rolled_angles - spin + left * right)


def contact_geoms(env):
    model = env.unwrapped.model
    named = [(g, (model.geom(g).name or "").lower()) for g in range(model.ngeom)]
    return (
        {g for g, name in named if "foot" in name},
        {g for g, name in named if "floor" in name},
    )


def touching(data, feet, floor):
    pairs = ((int(c.geom1), int(c.geom2)) for c in data.contact[: data.ncon])
    return any(a in feet and b in floor or b in feet and a in floor for a, b in pairs)


def cycles(contact):
    """Completed liftoff-to-landing cycles, which is what makes a gait a gait."""
    count = 0
    landed = airborne = False
    for down in contact:
        if down:
            count += airborne
            landed, airborne = True, False
        elif landed:
            airborne = True
    return count


def trial(env, feet, floor, seed):
    env.reset(seed=seed)
    data = env.unwrapped.data
    start = float(data.qpos[0])
    contact = []
    upright = True
    for _ in range(STEPS):
        _, _, terminated, truncated, _ = env.step(action(data.qpos, data.qvel))
        contact.append(touching(data, feet, floor))
        upright = not terminated
        if terminated or truncated:
            break
    hops = cycles(contact)
    distance = float(data.qpos[0]) - start
    return {
        "seed": seed,
        "steps": len(contact),
        "distance": distance,
        "cycles": hops,
        "success": (
            len(contact) == STEPS
            and upright
            and distance >= DISTANCE
            and hops >= CYCLES
        ),
    }


def replay():
    env = gym.make("Hopper-v5", reset_noise_scale=RESET_NOISE)
    feet, floor = contact_geoms(env)
    try:
        return [trial(env, feet, floor, seed) for seed in HELDOUT_SEEDS]
    finally:
        env.close()


def main():
    results = replay()
    for r in results:
        outcome = "passed" if r["success"] else "fell"
        line = f"seed {r['seed']}  {outcome:<6}  {r['distance']:6.2f} m"
        print(f"{line}  {r['cycles']:2d} hops  {r['steps']:4d} steps")
    print(f"\n{sum(r['success'] for r in results)} of {len(results)} held-out starts")


if __name__ == "__main__":
    main()
