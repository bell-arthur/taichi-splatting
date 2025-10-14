import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_all_logs():
    csv_files = glob.glob("/csse/users/abe118/Documents/SENG402/taichi-splatting/taichi_splatting/examples/outputs/mlp_hash/*.csv")
    plt.figure(figsize=(10, 8))
    for file in csv_files:
        df = pd.read_csv(file)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['iteration'], df['psnr'], label=label)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.title('PSNR vs. Iteration')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    for file in csv_files:
        df = pd.read_csv(file)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['time'], df['psnr'], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('PSNR')
    plt.title('PSNR vs. Time')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_all_logs()
