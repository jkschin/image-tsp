import os
import random
import pickle
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
import src.solver.heldkarp as hk
import src.solver.googleortools as gortools


def generate_coordinates(n, delta, size):
    """
    Args:
        n (int): number of vertical and horizontal roads
        delta (int): offset from edge of the city map
        size (tuple): shape tuple of city image. Example: (512, 512, 3)

    Returns:
        coords (list): a nested list of coordinates. Number of coordinates is
        exactly n.

    Raises:
        Exception: Generic exception with a message.

    Algorithm:
    1. Generates a random (x, y) coordinate.
    2. Tests if it is at least 10 pixels away from what has already been
    generated.
    3. Append to list.
    4. When n coordinates are generated, returns the coordinates.
    """
    xs = []
    ys = []
    coords = []
    for _ in range(n):
        i = 0
        while True:
            if i == 100000:
                raise Exception(
                    "Tried too long in generating a point. Check the size "
                    "tuple")
            x = random.randint(delta, size[0] - delta)
            y = random.randint(delta, size[1] - delta)
            if len(xs) != 0:
                delta_x = min(map(lambda z: abs(z - x), xs))
                delta_y = min(map(lambda z: abs(z - y), ys))
                if delta_x > 10 and delta_y > 10:
                    pass
                else:
                    continue
            coord = (x, y)
            if coord not in coords:
                xs.append(x)
                ys.append(y)
                break
            i += 1
    for i in range(len(xs)):
        for j in range(len(ys)):
            coords.append([xs[i], ys[j]])
    return coords


def generate_manhattan_distances(coords):
    """
    Args:
        coords (list): a list of the delivery points.

    Returns:
        dists (list): a nested list. The symmetric distance matrix of size n*n.

    Algorithm:
    1. Generates a random (x, y) coordinate.
    2. Tests if it is at least 10 pixels away from what has already been
    generated.
    3. Append to list.
    4. When n coordinates are generated, returns the coordinates.
    """
    n = len(coords)
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            point_a = coords[i]
            point_b = coords[j]
            x_dist = abs(point_a[0] - point_b[0])
            y_dist = abs(point_a[1] - point_b[1])
            dist = x_dist + y_dist
            dists[i][j] = dists[j][i] = dist
    return dists


def generate_tsp_sol(subset):
    dists = generate_manhattan_distances(subset)
    print("started")
    tsp_sol = hk.held_karp(dists)
    print("ended")
    return tsp_sol

if __name__ == "__main__":
    n = 20
    delta = 10
    size = (512, 512, 3)
    coords = generate_coordinates(n, delta, size)
    subset = random.sample(coords, 10)
    dists = hk.generate_manhattan_distances(subset)
    tsp_sol = generate_tsp_sol(subset)
    routes = gortools.main(dists)

# def compute_tsp_sol():
#     start = timer()
#     size_name = "%dx%d" %(size[0], size[0])
#     num_roads = 20
#     num_cities = 20
#     roads_name = "%d" %num_roads
#     train_input = os.path.join(size_name, roads_name, "train", "input")
#     train_output = os.path.join(size_name, roads_name, "train", "output")
#     test_input = os.path.join(size_name, roads_name, "test", "input")
#     test_output = os.path.join(size_name, roads_name, "test", "output")
#     folders = [train_input, train_output, test_input, test_output]
#     for folder in folders:
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#
#     random.seed(1)
#     coords = generate_coordinates(num_roads)
#     values = [generate_tuples(coords, train_input, train_output, num_cities, i) for i in range(10000)]
#     values = [i[1] for i in values]
#     print(f'starting computations on {cpu_count()} cores')
#
#     with Pool(cpu_count()) as pool:
#         res = pool.map(generate_tsp_sol, values)
#     end = timer()
#     dic = {}
#     dic["coords"] = coords
#     for i, r in enumerate(res):
#         dic[i] = {
#             "results": r,
#             "subset": values[i]
#         }
#     with open("tspdump.pickle", "wb") as f:
#         pickle.dump(dic, f, protocol=2)
#     print(f'elapsed time: {end - start}')
