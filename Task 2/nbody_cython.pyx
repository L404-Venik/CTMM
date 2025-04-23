# nbody_cython.pyx

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef double G = 39.478  # Gravitaional constant in AU^3 / (mass_of_the_Sun * year^2)

cdef double merge_threshold = 0.015

def merge_bodies(np.ndarray[double, ndim=2] positions,
                 np.ndarray[double, ndim=2] velocities,
                 np.ndarray[double, ndim=1] masses):

    cdef int n = masses.shape[0]
    cdef int i, j
    cdef double dx, dy, dist, total_mass

    for i in range(n):
        if masses[i] == 0:
            continue
        for j in range(i + 1, n):
            if masses[j] == 0:
                continue

            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = sqrt(dx*dx + dy*dy)

            if dist < merge_threshold:
                total_mass = masses[i] + masses[j]
                positions[i, 0] = (masses[i] * positions[i, 0] + masses[j] * positions[j, 0]) / total_mass
                positions[i, 1] = (masses[i] * positions[i, 1] + masses[j] * positions[j, 1]) / total_mass
                velocities[i, 0] = (masses[i] * velocities[i, 0] + masses[j] * velocities[j, 0]) / total_mass
                velocities[i, 1] = (masses[i] * velocities[i, 1] + masses[j] * velocities[j, 1]) / total_mass
                velocities[i, 0] = 0
                velocities[i, 1] = 0
                masses[i] = total_mass
                masses[j] = 0
                break

    return positions, velocities, masses


def compute_accelerations(np.ndarray[double, ndim=2] positions,
                          np.ndarray[double, ndim=1] masses):

    cdef int n = masses.shape[0]
    cdef np.ndarray[double, ndim=2] accels = np.zeros_like(positions)
    cdef int i, j
    cdef double dx, dy, r, inv_r3

    for i in range(n):
        for j in range(n):
            if i != j and masses[j] > 0:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                r = sqrt(dx*dx + dy*dy)
                if r > 0:
                    inv_r3 = 1.0 / (r*r*r)
                    accels[i, 0] += G * masses[j] * dx * inv_r3
                    accels[i, 1] += G * masses[j] * dy * inv_r3

    return accels


def verlet_integration_cython(int n,
                       np.ndarray[double, ndim=1] masses,
                       np.ndarray[double, ndim=2] positions,
                       np.ndarray[double, ndim=2] velocities,
                       double dt, int steps):

    cdef np.ndarray[double, ndim=3] positions_history = np.zeros((steps, n, 2), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] accelerations = compute_accelerations(positions, masses)
    cdef int i

    for i in range(steps):
        new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
        new_accelerations = compute_accelerations(new_positions, masses)
        new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt

        positions, velocities, masses = merge_bodies(new_positions, new_velocities, masses)
        accelerations = compute_accelerations(positions, masses)
        positions_history[i] = positions.copy()

    return positions_history
