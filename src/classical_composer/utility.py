import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import uuid

# Import your trained models and utilities
from classical_composer.data import (
    generate_piano_roll,
    get_fixed_frame_indicies,
    generate_random_frame_indicies,
)
from classical_composer.frames import extract_frames
from scipy.stats import mode


def create_piano_roll_image(piano_roll, extracted_frames, frame_indices, output_folder, fs=100):
    """
    Plots the entire piano roll with shaded regions for frames and a 3x3 grid of extracted frames.

    Args:
        piano_roll: 2D NumPy array representing the piano roll.
        extracted_frames: List of 2D NumPy arrays for the extracted frames.
        frame_indices: List of tuples (start_index, end_index) for each frame.
        fs: Frames per second for the piano roll.
    """
    fig, ax_main = plt.subplots(figsize=(15, 10))
    canvas = FigureCanvas(fig)  # Attach the canvas for non-interactive backend

    # Plot the full piano roll at the top
    ax_main.imshow(piano_roll, aspect="auto", origin="lower", cmap="hot")
    ax_main.set_title("Piano Roll with Highlighted Frames")
    ax_main.set_ylabel("Pitch")
    ax_main.set_xlabel(f"Time (1/{fs}s)")

    # Add shaded areas and labels for each frame
    colors = plt.cm.tab10(np.linspace(0, 1, len(frame_indices)))  # Generate unique colors
    for idx, (start, end) in enumerate(frame_indices):
        # Add a shaded rectangle
        ax_main.add_patch(
            patches.Rectangle(
                (start, 0),  # Bottom-left corner
                end - start,  # Width
                piano_roll.shape[0] / 10,  # Height
                linewidth=1,
                edgecolor=colors[idx],
                facecolor=colors[idx],
                alpha=0.3,  # Semi-transparent shading
            )
        )

        # Add a rotated label
        label_x = (start + end) / 2
        label_y = -15  # Bottom of the plot
        ax_main.text(
            label_x,
            label_y,
            f"Frame {idx+1}",
            color=colors[idx],
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,  # Rotate 90 degrees
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # Generate a random UUID
    file_UUID = uuid.uuid4()
    main_filename = f"{output_folder}/{file_UUID}.png"
    canvas.print_figure(main_filename)  # Save using the canvas
    plt.close(fig)  # Close the figure explicitly

    frame_files = []
    # Create each frame image
    for idx, frame in enumerate(extracted_frames):
        fig_frame, ax_frame = plt.subplots(figsize=(5, 3))
        canvas_frame = FigureCanvas(fig_frame)

        ax_frame.imshow(frame, aspect="auto", origin="lower", cmap="hot")
        ax_frame.axis("off")  # Turn off the axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove whitespace

        frame_filename = f"{output_folder}/{file_UUID}_frame_{idx}.png"
        canvas_frame.print_figure(frame_filename)
        plt.close(fig_frame)  # Close the frame figure explicitly

        frame_files.append(
            {"indicies": frame_indices[idx], "file_name": f"{file_UUID}_frame_{idx}.png"}
        )

    return {"file_name": f"{file_UUID}.png", "frames": frame_files}
