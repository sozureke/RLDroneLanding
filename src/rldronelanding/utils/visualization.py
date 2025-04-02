import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(trajectories, platform_pos=(0, 0)):
    plt.figure(figsize=(8, 8))

    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Episode {i+1}")
        plt.scatter(traj[0, 0], traj[0, 1], marker="o", color="gray", label=f"Start {i+1}" if i == 0 else "")
        plt.scatter(traj[-1, 0], traj[-1, 1], marker="x", color="red", label=f"End {i+1}" if i == 0 else "")

    plt.scatter(*platform_pos, color="green", s=200, label="Platform")

    plt.title("Drone Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()
