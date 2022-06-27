"""
A script for testing performance of a function using cupy vs numpy
"""
import cupy as cp

# Edit the following

# Change name for saving on different files
basename = "add_"

# Edit the content of the function
def fnc(x, y):
    "A function of x and y"
    return x + y


# Implement the same function here
kfnc = cp.ElementwiseKernel("T x, T y", "T z", "z = x + y", "kernel",)

# You can ignore the rest, if you want
import numpy as np
from timeit import timeit
from pandas import DataFrame, Series
from cupyx.profiler import benchmark


def run(iters=100, size_exp2=25):
    "The main function that runs the benchmark"
    times = DataFrame(index=Series([2 ** i for i in range(size_exp2)], name="Size"))

    for size in times.index:
        x, y = np.random.rand(2, size)
        times.at[size, "Numpy"] = timeit(lambda: fnc(x, y), number=iters) / iters

        x, y = cp.random.rand(2, size)
        times.at[size, "Cupy, inline"] = (
            timeit_gpu(lambda: fnc(x, y), number=iters) / iters
        )

        x, y = cp.random.rand(2, size)
        times.at[size, "Cupy, kernel"] = (
            timeit_gpu(lambda: kfnc(x, y), number=iters) / iters
        )

    save_times(times, basename=basename)


def timeit_gpu(fnc, number=100):
    "A time fnc for gpu"
    bench = benchmark(fnc, n_repeat=number)
    cpu = bench.cpu_times.sum()
    gpu = bench.gpu_times.sum()
    return cpu + gpu


def save(data, ylabel, filename, hline=None):
    "Saves plot and data"

    ax = data.plot(logx=True, logy=True, ylabel=ylabel, figsize=(8, 5))
    if hline:
        ax.axhline(hline, ls="--", color="black")

    fig = filename + ".pdf"
    ax.figure.savefig(fig, bbox_inches="tight")  # Save figure

    csv = filename + ".txt"
    data.to_csv(csv)  # Save data to csv format


def save_times(times, basename="", bandwidth=900, datasize=3 * 8):
    """
    This function displays some derived quantities from the provided timings.
    
    Namely:
    - the times themselves
    - the speedup compared to the first entry
    - the performance measured in GFLOP/s
    - the bandwidth measured in GB/s
    """

    # Timings
    save(times, "Time [s]", basename + "time")

    # Speedup
    key = next(iter(times))
    speedup = times.apply(lambda x: times[key] / x)
    save(speedup, f"Speed-up vs {key}", basename + "speedup")

    # Perfomance
    perf = times.apply(lambda x: np.array(times.index) / x / 1e9)
    save(perf, "Performance [GFLOP/s]", basename + "perf")

    # Bandwidth
    band = times.apply(lambda x: np.array(times.index) * datasize / x / 1e9)
    save(band, "Bandwidth [GB/s]", basename + "band", hline=bandwidth)


if __name__ == "__main__":
    run()
