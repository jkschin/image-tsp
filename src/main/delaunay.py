import pickle
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import src.solver.googleortools as gortools


def draw(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = [x2, x1, x1, x2, x2]
    b = [y2, y2, y1, y1, y2]
    return a, b


data = pickle.load(open("data/pickles/TSP200.pickle", "rb"))
coords = data["train"][0]["delivery_locations"]
coords = np.array(coords)
np.random.seed(100)
tri = Delaunay(coords)

points = coords
fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig1.set_size_inches(18.5, 10.5)
ax1.triplot(points[:,0], points[:,1], tri.simplices)
ax1.plot(points[:,0], points[:,1], 'o')
ax1.set_title("Delaunay Triangulation")
ax1.set_aspect(1)

for points in tri.simplices:
    points = list(points)
    points.append(points[0]) 
    for i in range(1, len(points)):
        x_values, y_values = draw(coords[points[i]], coords[points[i-1]])
        ax2.plot(x_values, y_values, color="blue")
ax2.set_title("Reconstructed Pruned Network from Delaunay Triangulation")
ax2.plot(coords[:,0], coords[:,1], 'o', color="green")
ax2.set_aspect(1)

plt.savefig("test2.png")

dist_matrix = np.ones((200, 200))
for points in tri.simplices:
    points = list(points)
    points.append(points[0])
    for i in range(1, len(points)):
        a, b = points[i], points[i-1]
        dist_matrix[a][b] = 0
        dist_matrix[b][a] = 0

routes = gortools.main(dist_matrix)
print(routes)

