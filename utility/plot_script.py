import matplotlib.pyplot as plt
import numpy as np

import os


def plot_predictions(
        all_preds,
        all_labels_x,
        all_labels_u,
        plot_idx,
        save_dir,
        epoch = None
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_rows = len(plot_idx)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 20))
    _, steps, _ = all_preds.shape
    steps = np.arange(0, steps)
    for row in range(n_rows):
        axes[row, 0].plot(steps, all_preds[plot_idx[row], :, 0], color='blue', linestyle='-.', label='pred_x1')
        axes[row, 0].plot(steps, all_preds[plot_idx[row], :, 1], color='purple', linestyle='-.', label='pred_x2')
        axes[row, 0].plot(steps, all_preds[plot_idx[row], :, 2], color='black', linestyle='-.', label='pred_x3')

        axes[row, 0].plot(steps, all_labels_x[plot_idx[row], :, 0], color='green', label='true_x1')
        axes[row, 0].plot(steps, all_labels_x[plot_idx[row], :, 1], color='yellow', label='true_x2')
        axes[row, 0].plot(steps, all_labels_x[plot_idx[row], :, 2], color='pink', label='true_x3')
        axes[row, 0].legend()
        axes[row, 0].set_xlabel('Steps')
        axes[row, 0].set_ylabel('States')

        axes[row, 1].plot(steps, all_labels_u[plot_idx[row], :, 0], color='blue', label='input_force_1')
        axes[row, 1].plot(steps, all_labels_u[plot_idx[row], :, 1], color='red', label='input_force_2')
        axes[row, 1].plot(steps, all_labels_u[plot_idx[row], :, 2], color='green', label='input_force_3')
        axes[row, 1].legend()
        axes[row, 1].set_xlabel('Steps')
        axes[row, 1].set_ylabel('Force')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    # plt.show()
    fig.savefig(save_dir + f'/plot_predictions_{epoch}.png', dpi=300)
    plt.close(fig)


def plot_contour(
        all_preds,
        all_labels_x,
        all_labels_u,
        plot_idx,
        save_dir,
        epoch = None
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_rows = len(plot_idx)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    _, steps, dim = all_preds.shape
    steps = np.arange(0, steps)
    dim = np.linspace(-np.pi, np.pi, dim)
    y, x = np.meshgrid(dim, steps)
    for row in range(n_rows):
        axes[row, 0].pcolormesh(x, y, all_preds[plot_idx[row], :, :], shading='auto')
        axes[row, 0].set_xlabel('Steps')
        axes[row, 0].set_ylabel('States')
        axes[row, 0].set_title('Predicted states')

        axes[row, 1].pcolormesh(x, y, all_labels_x[plot_idx[row], :, :], shading='auto')
        axes[row, 1].set_xlabel('Steps')
        axes[row, 1].set_ylabel('States')
        axes[row, 1].set_title('True states')

        axes[row, 2].plot(steps, all_labels_u[plot_idx[row], :, 0], color='blue', label='input_force')
        axes[row, 2].legend()
        axes[row, 2].set_xlabel('Steps')
        axes[row, 2].set_ylabel('Force')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    # plt.show()
    fig.savefig(save_dir + f'/plot_predictions_contour_{epoch}.png', dpi=300)
    plt.close(fig)

