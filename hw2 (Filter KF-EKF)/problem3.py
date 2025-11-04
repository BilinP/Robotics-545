import matplotlib.pyplot as plt
import numpy as np


class ExtendedKalmanFilter():
    """
    Implementation of an Extended Kalman Filter.
    """
    def __init__(self, mu, sigma, R=0., Q=0.):
        self.mu = mu
        self.sigma = sigma
        self.R = R
        self.Q = Q

    def predict(self):
        x, a = self.mu

        # predicted state
        x_pred = a * x
        a_pred = a
        self.mu = np.array([x_pred, a_pred])

        # Jacobian of dynamics derivative w.r.t. state
        F = np.array([[a, x],
                      [0, 1]])

        # process noise covariance 
        Qt = np.array([[self.R, 0],
                       [0, 0]])

        # predicted covariance  
        self.sigma = F @ self.sigma @ F.T + Qt

    def update(self, z):
        x, a = self.mu

        # predicted measurement
        z_pred = np.sqrt(x**2 + 1)

        # measurement Jacobian
        H = np.array([[x / np.sqrt(x**2 + 1), 0]])

        # innovation
        y = z - z_pred
        # innovation covariance
        S = H @ self.sigma @ H.T + self.Q
        #kalman gain
        K = self.sigma @ H.T @ np.linalg.inv(S)

        # state update
        self.mu = self.mu + (K.flatten() * y)

        # covariance update
        I = np.eye(len(self.mu))
        self.sigma = (I - K @ H) @ self.sigma




def plot_prediction(t, ground_truth, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    gt_x, gt_a = ground_truth[:, 0], ground_truth[:, 1]
    pred_x, pred_a = predict_mean[:, 0], predict_mean[:, 1]
    pred_x_std = np.sqrt(predict_cov[:, 0, 0])
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
    plt.fill_between(
        t,
        pred_x-pred_x_std,
        pred_x+pred_x_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
    plt.fill_between(
        t,
        pred_a-pred_a_std,
        pred_a+pred_a_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")

    plt.show()


def problem3():
    T=20
    Q=1.0
    R=0.5
     # true parameters
    alpha_true = 0.1
    x_true = 2.0

    # storage
    ground_truth,z_measurements = [],[]

    # simulate
    for t in range(T):
        ground_truth.append([x_true, alpha_true])
        z = np.sqrt(x_true**2 + 1) + np.random.normal(0, np.sqrt(Q))
        z_measurements.append(z)
        x_true = alpha_true * x_true + np.random.normal(0, np.sqrt(R))

    ground_truth = np.array(ground_truth)
    z_measurements = np.array(z_measurements)

    # initial estimates
    mu0 = np.array([1.0, 2.0])  
    P0 = np.diag([2.0, 2.0])

    ekf = ExtendedKalmanFilter(mu0, P0, R, Q)

    # storage for predictions
    predictions,covariances = [],[]

    for t in range(T):
        ekf.predict()
        ekf.update(np.array([z_measurements[t]]))
        predictions.append(ekf.mu.copy())
        covariances.append(ekf.sigma.copy())

    predictions = np.array(predictions)
    covariances = np.array(covariances)

    # plot results
    t = np.arange(T)
    plot_prediction(t, ground_truth, predictions, covariances)


if __name__ == '__main__':
    problem3()
