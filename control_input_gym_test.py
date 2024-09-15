import numpy as np

from waypoint_follow import PurePursuitPlanner
from f1tenth_gym.envs.track import Track
import gymnasium as gym
import matplotlib.pyplot as plt


def main():
    """
    Create an empty map with steering angle and acceleration control inputs from an NPZ file.
    """
    # Load control inputs from the NPZ file
    data = np.load('collected_data.npz')
    steer_angles = data['steer_angle']

    # vx = data['vx']
    # vy = data['vy']

    # # vy = -1 * vy
    # # vx = -1 * vx

    # #Get the velocity array from vx and vy
    # v = np.linalg.norm(np.stack([vx, vy], axis=0), axis=0)

    v = data['linear_acceleration_x']  

    # Interpolate velocity data to match the length of steering angles
    steer_len = len(steer_angles)
    v_len = len(v)

    # Create original and new x values for interpolation
    # x_original = np.linspace(0, steer_len - 1, steer_len)
    # x_new = np.linspace(0, steer_len - 1, v_len)

    # # Interpolate steering angles to fit the length of the velocity data
    # steer_interp = np.interp(x_new, x_original, steer_angles)

    x_original = np.linspace(0, v_len - 1, v_len)
    x_new = np.linspace(0, v_len - 1, steer_len)
    v_interp = np.interp(x_new, x_original, v)

    # Plotting steering angles and velocity after interpolation
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot interpolated steering angles
    axs[0].plot(steer_angles, label='Interpolated Steering Angle', color='blue')
    axs[0].set_title('Steering Angles Over Time (Interpolated to Match Acceleration)')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Steering Angle (rad)')
    axs[0].legend()

    # Plot velocity
    axs[1].plot(v_interp, label='Acceleration', color='green')
    axs[1].set_title('Acceleration Over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Acceleration (m/s^2)') 
    axs[1].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()


    # create sinusoidal reference line with custom velocity profile
    xs = np.linspace(0, 100, 200)
    ys = np.sin(xs / 2.0) * 5.0
    velxs = 4.0 * (1 + (np.abs(np.cos(xs / 2.0))))

    # Initialize a list to store the x and y coordinates
    car_x_positions = []
    car_y_positions = []

    # create track from custom reference line
    track = Track.from_refline(x=xs, y=ys, velx=velxs)

    params_dict = {'mu': 0.7,
               'C_Sf': 4.718,
               'C_Sr': 5.4562,
               'lf': 0.02735,
               'lr': 0.02585,
               'h': 0.1875,
               'm': 15.374,
               'I': 0.64332,
               's_min': -0.4189,
               's_max': 0.4189,
               'sv_min': -3.2,
               'sv_max': 3.2,
               'v_switch':7.319,
               'a_max': 9.51,
               'v_min':-5.0,
               'v_max': 20.0,
               'width': 0.8,
               'length': 0.55}


    # env and planner
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        params=params_dict,
        config={
            "map": track,
            "timestep": 0.1,
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
            "control_input": ["accl", "steering_angle"],
            "scale": 5.0,
        },
        render_mode="human",
    )
    # planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)

    # rendering callbacks
    env.add_render_callback(track.raceline.render_waypoints)
    # env.add_render_callback(planner.render_lookahead_point)

    # simulation
    obs, info = env.reset()
    done = False
    env.render()

    num_steps = min(len(steer_angles), len(v_interp))
    print(f"Number of steps: {num_steps}")

    for i in range(num_steps):
        steer = steer_angles[i]
        speed = v_interp[i]
        # action: steering angle and velocity
        action = np.array([[steer, speed]])
        obs, timestep, terminated, truncated, infos = env.step(action)

         # Save the car's current x and y positions
        car_x_positions.append(obs["agent_0"]["pose_x"])
        car_y_positions.append(obs["agent_0"]["pose_y"])

        done = terminated or truncated
        # print("current step: ", i)
        env.render()

        if done:
            break

    env.close()

    # Plot the car's x and y trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(car_x_positions, car_y_positions, label='Car Trajectory', color='red')
    plt.title('Car Trajectory Over Time')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()