import numpy as np
import time
from lib import fft_convolve2d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from matplotlib import animation

from pathlib import Path

from project.custom_images import convert_to_binary

plt.ion()

def conway(state, k=None):
    """
    Conway's game of life state transition
    """

    # set up kernel if not given
    if k == None:
        m, n = state.shape
        k = np.zeros((m, n))
        k[m//2-1 : m//2+2, n//2-1 : n//2+2] = np.array([[1,1,1],[1,0,1],[1,1,1]])

    # computes sums around each pixel
    b = fft_convolve2d(state,k).round()
    c = np.zeros(b.shape)

    c[np.where((b == 2) & (state == 1))] = 1
    c[np.where((b == 3) & (state == 1))] = 1

    c[np.where((b == 3) & (state == 0))] = 1

    # return new state
    return c


class Simulation:
    def __init__(self, path: str, runs: int = 30, iters: int = 40):
        self.path = path
        self.runs = runs
        self.iters = iters

        self.m, self.n = convert_to_binary(path)[0].shape
        self.expectation = np.zeros([self.iters, self.m, self.n])
        self.prob: np.ndarray = None

    def run(self):
        for r in range(self.runs):
            print(f'Run {r + 1}/{self.runs}...')

            # Initialize board
            board, p = convert_to_binary(self.path)
            if self.prob is None:
                self.prob = p

            # Initialize expectation
            self.expectation[0] += board.astype(float) / self.runs
            # Run
            for i in range(1, self.iters):
                print(f'\t{i + 1}/{self.iters}...')
                board = conway(board)
                self.expectation[i] += board.astype(float) / self.runs

    def save_vid(self, save_name: str = 'expectation'):
        def animate(frame: int):
            img.set_data(self.expectation[frame])
            return img

        fig, ax = plt.subplots()
        img = ax.imshow(self.expectation[0], cmap='inferno')
        fig.set_size_inches(15, 20)
        ani = animation.FuncAnimation(fig, animate, interval=100, frames=self.iters)
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save(self._save_path(save_name) + '.mp4', writer=FFwriter)
        print(f'Successfully saved   {self._save_path(save_name)}.mp4')

    def save_data(self, save_name: str = 'expectation'):
        results = {
            'exp': self.expectation,
            'prob': self.prob,
        }
        np.save(self._save_path(save_name), results, allow_pickle=True)
        print(f'Successfully saved   {self._save_path(save_name)}.npy')

    def _save_path(self, name: str):
        return f'{Path(__file__).parent}/project/{name} - {self.runs} runs {self.iters} iters'


def load_sim(
        path: Path,
        exp_smooth_shape: tuple = (5, 5), slope_shift: int = 15, slope_ignore: int = 5, exp_floor: float = 0.1
):
    # Normalized (& locally-smoothed) mean expectation
    results = np.load(path, allow_pickle=True).tolist()
    exp, prob = results['exp'], results['prob']
    exp_normed = exp / prob[None]
    exp_smoothed = uniform_filter(exp_normed, [1, exp_smooth_shape[0], exp_smooth_shape[1]])

    # # Vectorized linear regression - todo Fix it
    # log_exp = np.log(np.maximum(exp_smoothed, 0.1))
    # n = exp_smoothed.shape[0]
    # t = np.arange(n, dtype=float)
    # numer = (n * np.einsum('t, tij -> ij', t, log_exp) - t.sum() * log_exp.sum(axis=0))
    # denom = n * (np.linalg.norm(log_exp, axis=0) ** 2) - (log_exp.sum(axis=0) ** 2)
    # slope = np.minimum(numer / denom, 0)
    # relax = -1 / slope

    # Dumb-down linear regression
    log_exp = np.log(np.maximum(exp_smoothed, exp_floor))
    starts = np.arange(slope_ignore, n - slope_shift)
    stops = starts + slope_shift
    slopes = (log_exp[stops] - log_exp[starts]) / slope_shift
    # slope = slopes.mean(axis=0)
    slope = np.mean(slopes, axis=0)
    relax = -1 / slope

    breakpoint()

    """ Time series of expectation """
    frames = [0, 3, 10, 20, 29]

    fig, ax = plt.subplots(1, len(frames), sharex=True, sharey=True)
    fig.suptitle('Relaxation of mean cell value')
    for i, frame in enumerate(frames):
        ax[i].set_title(f'Generation {frame}')
        im = ax[i].imshow(log_exp[frame], cmap='inferno', vmax=0)
        ax[i].set_xlabel('j')
    ax[i].set_ylabel('i')
    # plt.colorbar(mappable=im, label='Log mean')
    fig.set_size_inches(15, 7)
    plt.tight_layout()


    """ Showing that the fall is exponential-like """
    n_chosen = 10
    chosen_i = np.random.randint(0, prob.shape[0], size=n_chosen)
    chosen_j = np.random.randint(0, prob.shape[1], size=n_chosen)

    fig, ax = plt.subplots()
    for i, j in zip(chosen_i, chosen_j):
        ax.plot(uniform_filter(log_exp[:, i, j], 3), linewidth=0.2, color='k')


    """ Relaxation map """
    fig, ax = plt.subplots()
    fig.suptitle('Relaxation time map')
    max_relax = 3
    replace_zeros = lambda x, val: np.where(x <= 0, val, x)
    im = ax.imshow(replace_zeros(np.log10(np.maximum(relax, 1)), max_relax), vmax=max_relax, cmap='RdBu')
    plt.colorbar(mappable=im)
    fig.set_size_inches(15, 10)

    fig, ax = plt.subplots()
    fig.suptitle('Growth map')
    im = ax.imshow(slope, cmap='RdBu', vmax=0)
    plt.colorbar(mappable=im)
    fig.set_size_inches(15, 10)


    """ Dependence of relaxation time with initial ALIVE probability """
    #log_relax = replace_zeros(np.log10(np.maximum(relax, 1)), max_relax)
    hist, xedges, yedges = np.histogram2d(prob.flatten(), slope.flatten(), bins=100)

    # Normalize the histogram over the x-axis (axis=1)
    hist_norm = hist / hist.sum(axis=1, keepdims=True)
    hist_norm = np.nan_to_num(hist_norm, 1e-5)

    plt.imshow(np.log10(np.maximum(hist_norm.T, 1e-5)), cmap='RdBu', origin='lower', aspect='auto',
               extent=[prob.min(), prob.max(), slope.min(), slope.max()])
    plt.colorbar()



    n1 = 4
    n2 = 29
    slope = (log_exp[n2] - log_exp[n1]) / (n2 - n1)


    # from d175d1ef5d45c9cc4662d3e82e20bc96.vectorized_linear_regression import LinearRegression
    #
    # relax = np.zeros(prob.shape, dtype=float)
    # for i in range(300):
    #     print(f'{i}...')
    #     lin = LinearRegression(iterations=500, learning_rate=1e-5)
    #     W = lin.fit(log_exp[:, i, :], np.arange(n))
    #     slope = lin.predict(W, 1).squeeze()
    #     relax[i] = np.maximum(np.minimum(-1 / slope, 300), 0)[1:]


    #todo
    #   Graphs to plot:
    #       Time series of mean relaxation
    #       Rate map (convert to relaxation time)
    #       Correlations


    # Correlation with prob (do it conditioned on prob)
    plt.hist2d(prob.flatten()[::10], slope.flatten()[::10], bins=50, cmap='RdBu')
    plt.colorbar()

    hist, xedges, yedges = np.histogram2d(prob.flatten(), slope.flatten(), bins=50)

    # Normalize the histogram over the x-axis (axis=1)
    hist_norm = hist / hist.sum(axis=1, keepdims=True)

    plt.imshow(np.log10(np.maximum(hist_norm.T, 1e-5)), cmap='RdBu', origin='lower', aspect='auto',
               extent=[prob.min(), prob.max(), slope.min(), slope.max()])
    plt.colorbar()
    #       --- Appears to not much correlate with density!





    # Correlation with prob variance


    breakpoint()
    return




# """ Estimate expectation """
#
# if __name__ == '__main__':
#     path = f'{Path(__file__).parent}/project/test - 20 runs 30 iters.npy'
#     load_sim(path)
#
#
# breakpoint()

""" Turn a photo into a game of life video """

if __name__ == "__main__":

    #todo
    #   Separate function for density obtainment, so we can normalize the curves
    #   Analyze the curves - are they exponential?

    path = '/mnt/c/Users/along/Downloads/test_image.jpg'

    sim = Simulation(path, runs=20, iters=30)
    sim.run()
    sim.save_vid('test')
    sim.save_data('test')

    breakpoint()

    #todo
    #   Save expectation time series numpy.
    #   Get normalized curves of the relaxation (divide by the maximum), place them all together (perhaps mean & STD)
    #   Infer relaxation time by thresholding and assuming this exponential model -> Relaxation map
    #   .
    #   LATER
    #   Find dependence curve between initial density, relaxation time.






breakpoint()

""" Typical run """
if __name__ == "__main__":
    path = '/mnt/c/Users/along/Downloads/test_image.jpg'

    from project.custom_images import convert_to_binary


    # Initialize board

    board = convert_to_binary(path)
    m, n = board.shape

    iters = 100
    board_timeseries = np.zeros([iters, board.shape[0], board.shape[1]], dtype=bool)
    board_timeseries[0] = board

    # breakpoint()

    # # set up board
    # m,n = 1000,1000
    # A = np.random.random(m*n).reshape((m, n)).round()

    # plot each frame
    #plt.figure()
    #img_plot = plt.imshow(A, interpolation="nearest", cmap = plt.cm.gray)
    #plt.show(block=False)
    #while True:

    # Run
    for i in range(1, iters):
        print(f'{i + 1}/{iters}...')
        board = conway(board)
        board_timeseries[i] = board

    from scipy.ndimage import uniform_filter

    smooth_timeseries = np.ones(board_timeseries.shape, dtype=float) * board_timeseries
    smooth_timeseries = uniform_filter(smooth_timeseries, [5, 10, 10])

    def animate(frame: int):
        #todo
        #   Include zooming in.

        # img.set_data(board_timeseries[frame])
        img.set_data(smooth_timeseries[frame])
        return img

    fig, ax = plt.subplots()
    img = ax.imshow(board, cmap='inferno')
    fig.set_size_inches(20, 20)
    ani = animation.FuncAnimation(fig, animate, interval=20)
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(f'{Path(__file__).parent}/project/density.mp4', writer=FFwriter)