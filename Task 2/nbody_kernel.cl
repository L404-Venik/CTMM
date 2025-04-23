// nbody_kernel.cl


__kernel void verlet_nbody(__global float2* positions,
    __global float2* velocities,
    __global float* masses,
    __global float2* positions_out,
    const int n,
    const float dt,
    const int steps,
    const float G)
{
    const int i = get_global_id(0);
    if (i >= n || masses[i] == 0.0f) return;

    float2 pos = positions[i];
    float2 vel = velocities[i];

    for (int s = 0; s < steps; ++s)
    {
        // Compute acceleration
        float2 acc = (float2)(0.0f, 0.0f);
        for (int j = 0; j < n; ++j)
        {
            if (i == j || masses[j] == 0.0f) continue;
            float2 r = positions[j] - pos;
            float dist_sqr = r.x*r.x + r.y*r.y + 1e-7f;
            float inv_r3 = native_rsqrt(dist_sqr * dist_sqr * dist_sqr);
            acc += G * masses[j] * r * inv_r3;
        }

    // Verlet integration
        pos = pos + vel * dt + 0.5f * acc * dt * dt;

        // Compute acceleration at new position
        float2 new_acc = (float2)(0.0f, 0.0f);
        for (int j = 0; j < n; ++j)
        {
            if (i == j || masses[j] == 0.0f) continue;
            float2 r = positions[j] - pos;
            float dist_sqr = r.x*r.x + r.y*r.y + 1e-7f;
            float inv_r3 = native_rsqrt(dist_sqr * dist_sqr * dist_sqr);
            new_acc += G * masses[j] * r * inv_r3;
        }

        // Update velocity
        vel = vel + 0.5f * (acc + new_acc) * dt;

        // Synchronize all threads at the end of this step
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Update shared position buffer for next step
        positions[i] = pos;

        // Optionally store position history
        positions_out[s * n + i] = pos;
    }

    velocities[i] = vel;
}