import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def nse(targets: np.ndarray, predictions: np.ndarray):
    return 1-(np.sum(np.square(targets-predictions))/np.sum(np.square(targets-np.mean(targets))))

def normalize(x):
    return 1/(2 - x)


def plot_quantiles(T: np.ndarray, Q: np.ndarray, Q_hat: np.ndarray, quantiles: list, threshold: float, ax=None):

    T = np.array(list(map(dt.datetime.fromtimestamp, T)))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(T, Q, color='black', label='obs', alpha=1.0)
    for i, q in enumerate(quantiles):
        if q == 0.5:
            ax.plot(T, Q_hat[:, i], label=f'pred-{q:.2f}', alpha=0.75, color='red')
    
    ax.fill_between(T, Q_hat[:, 0], Q_hat[:, -1], alpha=0.5, color='green')
    
    ax.axhline(y=threshold, color='blue',
                linestyle='--', label='flooding_threshold')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Flow (mm/day)')
    
    plt.legend()

    return ax.get_figure()


def evaluate(Q: np.ndarray, Q_hat:np.ndarray, quantiles:list=None):

    # Calculate NSE score
    if quantiles is not None:
        nse_score = nse(Q, Q_hat[:, int(len(quantiles)/2)])
    else:
        nse_score = nse(Q, Q_hat)
    nnse_score = normalize(nse_score)

    return nse_score, nnse_score


def out_of_interval(targets, predictions, low_idx=0, high_idx=None):
    
    # assert (len(predictions.shape)==3), "Expect 3 dimensions in predictions"
    
    if high_idx is None:
        high_idx = predictions.shape[-1] - 1
    
    out_of_interval = (targets < predictions[:, low_idx]) | (targets > predictions[:, high_idx])
    
    return out_of_interval.sum()/out_of_interval.shape[0]