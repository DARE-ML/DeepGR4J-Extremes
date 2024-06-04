import numpy as np
import matplotlib.pyplot as plt


def nse(targets: np.ndarray, predictions: np.ndarray):
    return 1-(np.sum(np.square(targets-predictions))/np.sum(np.square(targets-np.mean(targets))))

def normalize(x):
    return 1/(2 - x)

def evaluate(P: np.ndarray, E: np.ndarray, Q: np.ndarray, Q_hat:np.ndarray, quantiles: list, plot:bool = True):

    # Calculate NSE score
    nse_score = nse(Q, Q_hat[:, int(len(quantiles)/2)])
    nnse_score = normalize(nse_score)

    # Plot hydrograph
    if plot:
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(Q, color='black', label='obs', alpha=1.0)
        for i, q in enumerate(quantiles):
            ax.plot(Q_hat[:, i], label=f'pred-{q:.2f}', alpha=0.75)
        # ax.plot(P, 'g--', label='precip', alpha=0.40)
        # ax.plot(E, 'y--', label='etp', alpha=0.30)

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