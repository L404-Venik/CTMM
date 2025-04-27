import multiprocessing as mp
import numpy as np
from os import cpu_count

def verlet_worker(start, end, n, dt, steps, G, masses, shared_positions, shared_velocities, positions_out_base, barrier):
    local_positions = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((n, 2))
    local_velocities = np.frombuffer(shared_velocities.get_obj(), dtype=np.float32).reshape((n, 2))
    positions_out = np.frombuffer(positions_out_base.get_obj(), dtype=np.float32).reshape((steps, n, 2))

    for s in range(steps):
        positions_copy = local_positions.copy()

        for i in range(start, end):
            if masses[i] == 0.0:
                continue

            pos = positions_copy[i]
            vel = local_velocities[i]

            acc = np.zeros(2, dtype=np.float32)
            for j in range(n):
                if i == j or masses[j] == 0.0:
                    continue
                r = positions_copy[j] - pos
                dist_sqr = r[0]*r[0] + r[1]*r[1] + 1e-7
                inv_r3 = 1.0 / np.sqrt(dist_sqr * dist_sqr * dist_sqr)
                acc += G * masses[j] * r * inv_r3

            pos_new = pos + vel * dt + 0.5 * acc * dt * dt

            new_acc = np.zeros(2, dtype=np.float32)
            for j in range(n):
                if i == j or masses[j] == 0.0:
                    continue
                r = positions_copy[j] - pos_new
                dist_sqr = r[0]*r[0] + r[1]*r[1] + 1e-7
                inv_r3 = 1.0 / np.sqrt(dist_sqr * dist_sqr * dist_sqr)
                new_acc += G * masses[j] * r * inv_r3

            vel = vel + 0.5 * (acc + new_acc) * dt

            local_positions[i] = pos_new
            local_velocities[i] = vel

            positions_out[s, i] = pos_new

        barrier.wait()

def verlet_integration_threaded(n, masses, positions, velocities, dT, steps, G):
    positions = positions.copy()
    velocities = velocities.copy()

    num_processes = min(n, cpu_count())

    shared_positions = mp.Array('f', positions.flatten())
    shared_velocities = mp.Array('f', velocities.flatten())
    positions_out_base = mp.Array('f', steps * n * 2)

    # Create a barrier
    barrier = mp.Barrier(num_processes)

    # Split work
    chunk = (n + num_processes - 1) // num_processes
    ranges = [(i*chunk, min((i+1)*chunk, n)) for i in range(num_processes)]

    processes = []
    for r in ranges:
        p = mp.Process(target=verlet_worker, args=(r[0], r[1], n, dT, steps, G, masses, shared_positions, shared_velocities, positions_out_base, barrier))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    positions_out = np.frombuffer(positions_out_base.get_obj(), dtype=np.float32).reshape((steps, n, 2))

    return positions_out
