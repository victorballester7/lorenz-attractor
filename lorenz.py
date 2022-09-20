import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def lorenz(r: npt.NDArray[np.float64], s: float = 10,
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

r = np.empty((num_steps + 1, 3))


# Set initial values
r[0] = [0., 1., 1.05]

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
  k1 = lorenz(r[i])
  k2 = lorenz(r[i] + dt * k1 / 2)
  k3 = lorenz(r[i] + dt * k2 / 2)
  k4 = lorenz(r[i] + dt * k3)
  r[i + 1] = r[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(r[:, 0], r[:, 1], r[:, 2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
