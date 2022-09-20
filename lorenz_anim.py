from matplotlib.animation import FuncAnimation
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def edo(r: npt.NDArray[np.float64], s: float = 10,
        p: float = 28, b: float = 2.667):
  """
  Given:
     x, y, z: a point of interest in three dimensional space
     s, p, b: parameters defining the lorenz attractor
  Returns:
     x_dot, y_dot, z_dot: values of the lorenz attractor's partial
         derivatives at the point x, y, z
  """
  x, y, z = r[0], r[1], r[2]

  x_dot = s * (y - x)
  y_dot = p * x - y - x * z
  z_dot = x * y - b * z
  return np.array([x_dot, y_dot, z_dot])


dt = 0.01
num_steps = 10000

# Need one more for the initial values

r = np.zeros((num_steps + 1, 3))


# Set initial values
r[0] = [0., 1., 1.05]

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
# for i in range(num_steps):
#   k1 = lorenz(r[i])
#   k2 = lorenz(r[i] + dt * k1 / 2)
#   k3 = lorenz(r[i] + dt * k2 / 2)
#   k4 = lorenz(r[i] + dt * k3)
#   r[i + 1] = r[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(r, f, num_steps, dt):  # runge-kutta method for autonomous systems
  for i in range(num_steps):
    k1 = f(r[i])
    k2 = f(r[i] + dt * k1 / 2)
    k3 = f(r[i] + dt * k2 / 2)
    k4 = f(r[i] + dt * k3)
    r[i + 1] = r[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


rk4(r, edo, num_steps, dt)


# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlim(-25, 25)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 50)


# ax.plot(r[:, 0], r[:, 1], r[:, 2], lw=0.5)  # lw = linewidth
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

ball, = ax.plot([], [], [], 'o', lw=2)  # lw = linewidth
trace, = ax.plot([], [], [], '-', lw=1, ms=1)  # ms = markersize
# history_x, history_y, history_z = deque(
#     maxlen=num_steps), deque(
#     maxlen=num_steps), deque(
#     maxlen=num_steps)


# history_x, history_y, history_z = np.zeros(
#     num_steps), np.zeros(num_steps), np.zeros(num_steps)


def animation_frame(i):
  x_data, y_data, z_data = r[i, 0], r[i, 1], r[i, 2]

  ball.set_data_3d(x_data, y_data, z_data)
  trace.set_data_3d(r[:i, 0], r[:i, 1], r[:i, 2])

  return ball, trace


animation = FuncAnimation(
    fig,
    func=animation_frame,
    # frames=np.arange(
    #     0,
    #     num_steps),
    frames=np.arange(len(r)),
    interval=1, blit=True)

plt.show()
