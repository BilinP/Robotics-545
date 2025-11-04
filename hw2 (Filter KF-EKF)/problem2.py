import matplotlib.pyplot as plt
import numpy as np


class KalmanFilter():
    """
    Implementation of a Kalman Filter.
    """
    def __init__(self, mu, sigma, A, C, R=0., Q=0.):
        self.mu=mu
        self.sigma=sigma
        self.A=A
        self.C=C
        self.R=R
        self.Q=Q
        
    def predict(self):
        # Prediction step
        self.mu = self.A @ self.mu
        self.sigma = self.A @ self.sigma @ self.A.T + self.Q
        return self.mu, self.sigma
    
    def update(self, z):
        # Innovation
        y = z - (self.C @ self.mu)
        S = self.C @ self.sigma @ self.C.T + self.R
        K = self.sigma @ self.C.T @ np.linalg.inv(S)

        # Update step
        self.mu = self.mu + K @ y
        self.sigma = (np.eye(len(self.mu)) - K @ self.C) @ self.sigma
        return self.mu, self.sigma

def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param measurement: Tx1 array of sensor values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    predict_pos_mean = predict_mean[:, 0]
    predict_pos_std = np.sqrt(predict_cov[:, 0, 0])

    plt.figure()
    plt.plot(t, ground_truth, color='k')
    plt.plot(t, measurement, color='r')
    plt.plot(t, predict_pos_mean, color='g')
    plt.fill_between(
        t,
        predict_pos_mean-predict_pos_std,
        predict_pos_mean+predict_pos_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground truth", "measurements", "predictions"))
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Predicted Values")
    plt.show()


def plot_mse(t, ground_truth, predict_means):
    """
    Plot MSE of your KF over many trials.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_means: NxTxD array of T mean vectors over N trials
    """
    predict_pos_means = predict_means[:, :, 0]
    errors = ground_truth.squeeze() - predict_pos_means
    mse = np.mean(errors ** 2, axis=0)

    plt.figure()
    plt.plot(t, mse)
    plt.xlabel("time (s)")
    plt.ylabel("position MSE (m^2)")
    plt.title("Prediction Mean-Squared Error")
    plt.show()




def problem2a():
    T = 100
    t = np.arange(T) * 0.1 


    A = np.array([
        [1, 0.1, 0, 0],
        [0, 1, 0.1, 0],
        [0, 0, 1, 0.1],
        [0, 0, 0, 1]
    ])
    C = np.array([[1, 0, 0, 0]])

    # Initial
    mu0 = np.array([5, 1, 0, 0])
    Sigma0 = np.diag([10, 10, 10, 10])

    # Ground truth    
    t = np.arange(0, T * 0.1, 0.1)
    ground_truth = np.sin(0.1 * t)
    
    # first figure
    z = ground_truth + np.random.normal(0, 1.0, size=T)
    kf = KalmanFilter(mu0, Sigma0, A, C, R=1.0)
    means, covs = [], []
    for i in range(T):
        kf.predict()
        mu_update, cov_upd = kf.update(z[i])
        means.append(mu_update)
        covs.append(cov_upd)

    means, covs = np.array(means), np.array(covs)
    plot_prediction(t, ground_truth, z, means, covs)

    # 10k trails for MSE 2nd figure

    N = 10000
    all_means_N = []
    for trails in range(N):
        z = ground_truth + np.random.normal(0, 1.0, size=T)
        kf = KalmanFilter(mu0, Sigma0, A, C, R=1.0)
        means = []
        for i in range(T):
            kf.predict()
            mu_update, _ = kf.update(z[i])
            means.append(mu_update)
        all_means_N.append(means)

    all_means_N = np.array(all_means_N)
    plot_mse(t, ground_truth, all_means_N)

def problem2b():
    T = 100
    t = np.arange(T) * 0.1 


    A = np.array([
        [1, 0.1, 0, 0],
        [0, 1, 0.1, 0],
        [0, 0, 1, 0.1],
        [0, 0, 0, 1]
    ])
    C = np.array([[1, 0, 0, 0]])

    # Initial
    mu0 = np.array([5, 1, 0, 0])
    Sigma0 = np.diag([10, 10, 10, 10])

    # Ground truth    
    t = np.arange(0, T * 0.1, 0.1)
    ground_truth = np.sin(0.1 * t)
    
    # process noise
    R_proc = np.diag([0.1, 0.1, 0.1, 0.1])
    
    # first figure
    z = ground_truth + np.random.normal(0, 1.0, size=T)
    kf = KalmanFilter(mu0, Sigma0, A, C, R=1.0,Q=R_proc)
    means, covs = [], []
    for i in range(T):
        kf.predict()
        mu_update, cov_upd = kf.update(z[i])
        means.append(mu_update)
        covs.append(cov_upd)

    means, covs = np.array(means), np.array(covs)
    plot_prediction(t, ground_truth, z, means, covs)

    # 10k trails for MSE 2nd figure

    N = 10000
    all_means_N = []
    for trails in range(N):
        z = ground_truth + np.random.normal(0, 1.0, size=T)
        kf = KalmanFilter(mu0, Sigma0, A, C, R=1.0,Q=R_proc)
        means = []
        for i in range(T):
            kf.predict()
            mu_update, _ = kf.update(z[i])
            means.append(mu_update)
        all_means_N.append(means)

    all_means_N = np.array(all_means_N)
    plot_mse(t, ground_truth, all_means_N)



if __name__ == '__main__':
    problem2a()
    problem2b()
