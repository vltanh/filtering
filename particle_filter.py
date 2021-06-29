import numpy as np
from numpy.random import randn


def particle_filter(dynamic, measurement, noise_pdf, n_particles, X, z0, P0):
    N = X.shape[0]

    particles = z0 + np.sqrt(P0) * randn(n_particles)
    particles_pred = np.zeros(n_particles)
    particles_weight = np.ones(n_particles) / n_particles

    Z_est = np.zeros(N)
    Z_est[0] = particles.mean()

    P_est = np.zeros(N)
    P_est[0] = particles.std()

    for t in range(1, N):
        particles_pred = dynamic(particles, t)
        innov = X[t] - measurement(particles_pred)
        particles_weight = noise_pdf(innov)
        particles_weight /= particles_weight.sum()

        particles = np.random.choice(
            particles_pred, size=n_particles, p=particles_weight)
        Z_est[t] = particles.mean()
        P_est[t] = particles.std()
    return Z_est, P_est
