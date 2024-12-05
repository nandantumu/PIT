import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_data(data_dict, axs=None, color='b'):
    axs_specified = True if axs is not None else False
    if axs is None:
        fig, axs = plt.subplots(6, 2, figsize=(15, 10), constrained_layout=True)
    time = data_dict["time"]

    # Plot x y position separately
    axs[0, 0].plot(time, data_dict["x"], color=color)
    axs[0, 0].set_title("Position X")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].grid()

    axs[0, 1].plot(time, data_dict["y"], color=color)
    axs[0, 1].set_title("Position Y")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Position (m)")
    axs[0, 1].grid()

    # Plot steering angle
    axs[1, 0].plot(time, data_dict["steering_angle"], color=color)
    axs[1, 0].set_title("Steering Angle")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Angle (rad)")
    axs[1, 0].grid()

    # Plot velocity
    axs[1, 1].plot(time, data_dict["velocity"], color=color)
    axs[1, 1].set_title("Velocity")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Velocity (m/s)")
    axs[1, 1].axhline(y=0.1, color="r", linestyle="--")
    axs[1, 1].grid()

    # Plot yaw
    axs[2, 0].plot(time, data_dict["yaw"], color=color)
    axs[2, 0].set_title("Yaw")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Yaw (rad)")
    axs[2, 0].grid()

    # Plot yaw rate
    axs[2, 1].plot(time, data_dict["yaw_rate"], color=color)
    axs[2, 1].set_title("Yaw Rate")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Yaw Rate (rad/s)")
    axs[2, 1].grid()

    # Plot slip angle
    axs[3, 0].plot(time, data_dict["slip_angle"], color=color)
    axs[3, 0].set_title("Slip Angle")
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].set_ylabel("Slip Angle (rad)")
    axs[3, 0].axhline(y=-0.26, color="r", linestyle="--")
    axs[3, 0].axhline(y=0.26, color="r", linestyle="--", label="Small Angle Bound")
    axs[3, 0].grid()

    # Plot acceleration
    axs[3, 1].plot(time, data_dict["acceleration"], color=color)
    axs[3, 1].set_title("Acceleration")
    axs[3, 1].set_xlabel("Time (s)")
    axs[3, 1].set_ylabel("Acceleration (m/s^2)")
    axs[3, 1].grid()
    
    try:
        # Plot steering velocity
        axs[4, 0].plot(time, data_dict["steering_velocity"], color=color)
        axs[4, 0].set_title("Steering Velocity")
        axs[4, 0].set_xlabel("Time (s)")
        axs[4, 0].set_ylabel("Velocity (rad/s)")
        axs[4, 0].grid()
    except KeyError:
        pass

    try:
        # Plot dt
        axs[4, 1].plot(time, data_dict["dt"], color=color)
        axs[4, 1].set_title("Delta Time")
        axs[4, 1].set_xlabel("Time (s)")
        axs[4, 1].set_ylabel("Delta Time (s)")
        axs[4, 1].grid()
    except KeyError:
        pass

    # Plot wheel speeds:
    try:
        axs[5, 0].plot(time, data_dict["omega_f"], color=color)
        axs[5, 0].set_title("Front Wheel Speed")
        axs[5, 0].set_xlabel("Time (s)")
        axs[5, 0].set_ylabel("Speed (rad/s)")
        axs[5, 0].grid()

        axs[5, 1].plot(time, data_dict["omega_r"], color=color)
        axs[5, 1].set_title("Rear Wheel Speed")
        axs[5, 1].set_xlabel("Time (s)")
        axs[5, 1].set_ylabel("Speed (rad/s)")
        axs[5, 1].grid()
    except KeyError:
        pass
    except ValueError:
        pass


    if not axs_specified:
        plt.show()


def position_and_slip_viz(data_dict, FREQ=100):
    x = data_dict["x"]
    y = data_dict["y"]
    yaw = data_dict["yaw"]
    slip_angle = data_dict["slip_angle"]
    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    # Plot the yaw of the car as arrows
    plt.quiver(x[::FREQ], y[::FREQ], np.cos(yaw[::FREQ]), np.sin(yaw[::FREQ]))
    # Plot the slip a10gle of the10car as arrows, and 10enter them at the ya10 of the c10r
    plt.quiver(
        x[::FREQ],
        y[::FREQ],
        np.cos(yaw[::FREQ] + slip_angle[::FREQ]),
        np.sin(yaw[::FREQ] + slip_angle[::FREQ]),
        color="r",
        width=0.005,
        scale=30,
    )
    plt.title("Position of the car")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
