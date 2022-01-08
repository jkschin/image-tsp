import os
import random
import pickle
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
import src.solver.heldkarp as hk
import src.solver.googleortools as gortools
import src.visualization.visualization as viz
import numpy as np
import cv2
import sys


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


def TSP20():
    random.seed(1)
    num_train = 10000
    num_test = 1000
    total = num_train + num_test
    coords = generate_coordinates(num_roads, delta, size)
    values = [random.sample(coords, num_cities) for _ in range(total)]
    num_cores = cpu_count() // 2 - 1
    print(f'starting computations on {num_cores} cores')

    with Pool(num_cores) as pool:
        res = pool.map(generate_hk_tsp_sol, values)
    dic = {}
    dic["coords"] = coords
    dic["train"] = {}
    dic["test"] = {}
    for i in range(num_train):
        dic["train"][i] = {
            "results": res[i][0],
            "time": res[i][1],
            "delivery_locations": values[i]
        }
    for i in range(num_train, total):
        dic["test"][i] = {
            "results": res[i][0],
            "time": res[i][1],
            "delivery_locations": values[i]
        }
    with open("TSP20.pickle", "wb") as f:
        pickle.dump(dic, f, protocol=2)


def TSP():
    random.seed(1)
    num_train = 10000
    num_test = 1000
    total = num_train + num_test
    coords = generate_coordinates(num_roads, delta, size)
    values = [random.sample(coords, num_cities) for _ in range(total)]
    num_cores = cpu_count() 
    print(f'starting computations on {num_cores} cores')

    with Pool(num_cores) as pool:
        res = pool.map(generate_gortools_tsp_sol, values)
    dic = {}
    dic["coords"] = coords
    dic["train"] = {}
    dic["test"] = {}
    for i in range(num_train):
        dic["train"][i] = {
            "results": res[i][0],
            "time": res[i][1],
            "delivery_locations": values[i]
        }
    for i in range(num_train, total):
        dic["test"][i] = {
            "results": res[i][0],
            "time": res[i][1],
            "delivery_locations": values[i]
        }
    with open("TSP%d.pickle" %num_cities, "wb") as f:
        pickle.dump(dic, f, protocol=2)


def generate_hk_tsp_sol(subset):
    dists = generate_manhattan_distances(subset)
    start = timer()
    print("started")
    tsp_sol = hk.held_karp(dists)
    end = timer()
    elapsed = end - start
    print("ended: ", elapsed)
    return tsp_sol, elapsed


def generate_gortools_tsp_sol(subset):
    dists = generate_manhattan_distances(subset)
    start = timer()
    print("started")
    tsp_sol = gortools.main(dists)
    end = timer()
    elapsed = end - start
    print("ended: ", elapsed)
    return tsp_sol, elapsed


def generate_tsp_images(picklefile):
    cwd = os.getcwd()
    filenames = picklefile.split("/")
    assert len(filenames) == 3
    num_cities = filenames[-1].split(".")[0][3:]
    data = pickle.load(open(picklefile, "rb"))
    assert len(data.keys()) == 3
    assert "coords" in data.keys()
    assert "train" in data.keys()
    assert "test" in data.keys()
    assert len(data["coords"]) == 400
    assert len(data["train"]) == 10000
    assert len(data["test"]) == 1000
    LOCAL = "local"
    CLOUD = "cloud"
    if cwd.startswith("/Users"):
        ENV = LOCAL
    elif cwd.startswith("/home"):
        ENV = CLOUD
    else:
        raise Exception("Current working directory neither starts with /Users"
                        "or /home.")
    print("Generating TSP images on %s environment for %s cities" %(ENV, num_cities))


    city_coords = data["coords"]
    for data_class in ["train", "test"]:
        i = 0
        for k, v in data[data_class].items():
            tour = v["results"][1]
            delivery_locations = v["delivery_locations"]
            for image_type in ["input", "output"]:
                img = np.zeros(size, np.uint8)
                path = "data/images/%s/%s/%s" %(num_cities, data_class, image_type)
                if not os.path.exists(path):
                    os.makedirs(path)
                if image_type == "input":
                    img = viz.draw_input_data(img, city_coords,
                                              delivery_locations,
                                              size, delta, viz.RED, False)
                    img_path = os.path.join(path, "input_%05d.png" %k)
                    cv2.imwrite(img_path, img)
                elif image_type == "output":
                    img = viz.draw_output_data(img, delivery_locations,
                                               tour, delta, viz.RED, False)
                    img_path = os.path.join(path, "output_%05d.png" %k)
                    cv2.imwrite(img_path, img)
            i += 1
            if ENV == LOCAL and i == 100:
                break
            if i % 1000 == 0:
                print(i)


if __name__ == "__main__":
    num_roads = 20
    num_cities = 200
    delta = 5
    size = (512, 512, 3)
    command = sys.argv[1]
    if command == "generate":
        print("Num Cities: ", num_cities)
        TSP()
    elif command == "draw":
        picklefile = sys.argv[2]
        generate_tsp_images(picklefile)
    elif command == "drawall":
        d = sys.argv[2]
        for filename in os.listdir(d):
            picklefile = os.path.join(d, filename)
            generate_tsp_images(picklefile)
    else:
        raise Exception("Invalid Command")

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
