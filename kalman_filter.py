import numpy as np
import matplotlib.pyplot as plt


def visualize_kalman_filter_1d(T, X, Z, Z_est, P_est, X_pred, gain):
    L = T.max()

    fig, ax = plt.subplots(2, 2, figsize=(20, 10), dpi=100)

    ax[0, 0].fill_between(T,
                          Z_est + np.sqrt(P_est),
                          Z_est - np.sqrt(P_est),
                          alpha=0.3)
    ax[0, 0].plot(T, Z_est, label='Estimation')
    ax[0, 0].plot(T, Z, label='True')
    ax[0, 0].set_xlim(0, L)
    ax[0, 0].legend()

    ax[0, 1].plot((T[1:], T[1:]), (X[1:], X_pred[1:]), c='red', alpha=0.3)
    ax[0, 1].scatter(T[1:], X_pred[1:], marker='.',
                     label='Predicted Measurement')
    ax[0, 1].scatter(T, X, marker='.', label='Measurement')

    ax[0, 1].set_xlim(0, L)
    ax[0, 1].legend()

    est_err = Z - Z_est
    top_std = 2 * np.sqrt(P_est)
    bot_std = - top_std
    ax[1, 0].plot(T, est_err, label='State estimation error')
    ax[1, 0].plot(T, top_std, label='$+2\sqrt{P_{est}}$')
    ax[1, 0].plot(T, bot_std, label='$-2\sqrt{P_{est}}$')
    ax[1, 0].set_xlim(0, L)
    ax[1, 0].legend()

    ax[1, 1].plot(T[1:], gain[1:], label='Kalman gain')
    ax[1, 1].set_xlim(0, L)
    ax[1, 1].legend()

    plt.show()


def kalman_filter_1d(a, b, c, d, sw, sv, X, U, z0=None, p0=None):
    N = X.shape[0]
    Z_pred, P_pred, Z_est, P_est, X_pred, gain = np.zeros((6, N))

    Z_est[0] = z0 if z0 is not None else (X[0] - d[0]) / c[0]
    P_est[0] = p0 if p0 is not None else 1.
    for t in range(1, N):
        # Prediction
        # Prediction of state
        Z_pred[t] = a[t-1] * Z_est[t-1] + b[t-1] * U[t-1]
        P_pred[t] = a[t-1] ** 2 * P_est[t-1] + sw

        # Prediction of measurement
        X_pred[t] = c[t] * Z_pred[t] + d[t]

        # Update
        gain[t] = c[t] * P_pred[t] / (sv + c[t]**2 * P_pred[t])
        Z_est[t] = Z_pred[t] + gain[t] * (X[t] - X_pred[t])
        P_est[t] = P_pred[t] * (1 - c[t] * gain[t])

    return Z_pred, P_pred, Z_est, P_est, X_pred, gain


def extended_kalman_filter_1d(process, d_process,
                              measurement, d_measurement,
                              control,
                              sw, sv, X,
                              z0=None, p0=None):
    N = X.shape[0]
    Z_pred, P_pred, Z_est, P_est, X_pred, gain = np.zeros((6, N))

    Z_est[0] = z0 if z0 is not None else 0.
    P_est[0] = p0 if p0 is not None else 1.
    for t in range(1, N):
        # Prediction
        # Prediction of state
        Z_pred[t] = process(Z_est[t-1]) + control(t)
        a = d_process(Z_pred[t])
        P_pred[t] = a ** 2 * P_est[t-1] + sw

        # Prediction of measurement
        X_pred[t] = measurement(Z_pred[t])
        c = d_measurement(Z_pred[t])

        # Update
        gain[t] = c * P_pred[t] / (sv + c**2 * P_pred[t])
        Z_est[t] = Z_pred[t] + gain[t] * (X[t] - X_pred[t])
        P_est[t] = P_pred[t] * (1 - c * gain[t])

    return Z_pred, P_pred, Z_est, P_est, X_pred, gain
