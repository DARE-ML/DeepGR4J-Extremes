import numpy as np
import matplotlib.pyplot as plt

def confidence_score(targets: np.ndarray, predictions: np.ndarray):
    lowlim = predictions[:, 0]
    uplim = predictions[:, -1]
    targets = targets.flatten()
    flag = (targets>lowlim) & (targets<uplim)
    return flag.sum()/len(targets)


def nse(targets: np.ndarray, predictions: np.ndarray):
    return 1-(np.sum(np.square(targets-predictions))/np.sum(np.square(targets-np.mean(targets))))

def normalize(x):
    return 1/(2 - x)

def evaluate(P: np.ndarray, E: np.ndarray, Q: np.ndarray, Q_hat:np.ndarray, quantiles:list=None, plot:bool=True):

    # Calculate NSE score
    if quantiles is not None:
        nse_score = nse(Q, Q_hat[:, int(len(quantiles)/2)])
    else:
        nse_score = nse(Q, Q_hat)
    nnse_score = normalize(nse_score)

    # Plot hydrograph
    if plot:
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(Q, color='black', label='obs', alpha=1.0)
        if quantiles is not None:
            for i, q in enumerate(quantiles):
                if q == 0.5:
                    ax.plot(Q_hat[:, i], label=f'pred-{q:.2f}', alpha=0.75, color='red')
            ax.fill_between(range(len(Q_hat)), Q_hat[:, 0], Q_hat[:, -1], alpha=0.5, color='green')
        else:
            ax.plot(Q_hat, color='red', label=f'pred', alpha=0.75)
            ax.plot(P, 'g--', label='precip', alpha=0.40)
            ax.plot(E, 'y--', label='etp', alpha=0.30)

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Flow (mm/day)')

        ax.annotate(f'NSE: {nse_score:.4f}',
                xy=(0.90, 0.92), xycoords='figure fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12)
        ax.set_title('Streamflow prediction')

        plt.legend()

        return nse_score, nnse_score, fig

    return nse_score, nnse_score