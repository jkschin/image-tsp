import itertools
import random
import sys
import math
import numpy as np
import cv2
import os
import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
import pickle

size = (256, 256, 3)
size = (224, 224, 3)
size = (128, 128, 3)
size = (512, 512, 3)
delta = 5
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.

    Parameters:
        dists: distance matrix

    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        # Tracks the DP and memory usage.
        # print(n, subset_size, sys.getsizeof(C)/1000000, len(C.keys()))
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state and close the loop
    path.append(0)
    path = list(reversed(path))
    path.append(0)

    return opt, path


def generate_distances(coords):
    n = len(coords)
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            pointA = coords[i]
            pointB = coords[j]
            x_dist = (pointA[0] - pointB[0])**2
            y_dist = (pointA[1] - pointB[1])**2
            dist = math.sqrt(x_dist + y_dist)
            dists[i][j] = dists[j][i] = dist
    return dists

def generate_manhattan_distances(coords):
    n = len(coords)
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            pointA = coords[i]
            pointB = coords[j]
            x_dist = abs(pointA[0] - pointB[0])
            y_dist = abs(pointA[1] - pointB[1])
            dist = x_dist + y_dist
            dists[i][j] = dists[j][i] = dist
    return dists

def read_distances(filename):
    dists = []
    with open(filename, 'rb') as f:
        for line in f:
            # Skip comments
            if line[0] == '#':
                continue

            dists.append(map(int, map(str.strip, line.split(','))))

    return dists

def draw_dots(img, coords, color):
    for coord in coords:
        r = delta // 2
        x, y = coord
        # OpenCV has BGR instead of RGB
        img[y-r: y+r+1, x-r:x+r+1] = color
    return img


def draw_path(img, coords, path):
    for i in range(1, len(path)):
        a = coords[path[i]]
        b = coords[path[i-1]]
        img = cv2.line(img, a, b, (2), 1)
    a = coords[path[0]]
    b = coords[path[-1]]
    img = cv2.line(img, a, b, (2), 1)
    return img

def draw_sol(img, coords, path):
    img = draw_path(img, coords, path)
    img = draw_dots(img, coords)
    return img

def generate_pair(n):
    coords = generate_coordinates(n)
    dists = generate_distances(coords)
    tsp_sol = held_karp(dists)
    inp = np.zeros((100, 100), np.uint8)
    inp = draw_dots(inp, coords)
    out = np.zeros((100, 100), np.uint8)
    out = draw_sol(out, coords, tsp_sol[1])
    return inp, out

def generate_city(coords):
    img = np.zeros(size, np.uint8)
    for coord in coords:
        x, y = coord
        xa = (x, 0)
        xb = (x, size[0])
        ya = (0, y)
        yb = (size[1], y)
        img = cv2.line(img, xa, xb, WHITE, delta//2)
        img = cv2.line(img, ya, yb, WHITE, delta//2)
    # img = draw_dots(img, coords, (0, 0, 255))
    min_x = min(map(lambda x: x[0], coords))
    max_x = max(map(lambda x: x[0], coords))
    min_y = min(map(lambda x: x[1], coords))
    max_y = max(map(lambda x: x[1], coords))
    # new_img = np.zeros(size, np.uint8)
    # new_img[min_y: max_y+1, min_x:max_x+1, :] = img[min_y: max_y+1, min_x:max_x+1, :]
    return img

def generate_manhattan_solution(img, coords, path):
    for i in range(1, len(path)):
        ax, ay = coords[path[i]]
        bx, by = coords[path[i-1]]
        img = cv2.line(img, (ax, ay), (ax, by), GREEN, delta//2)
        img = cv2.line(img, (ax, by), (bx, by), GREEN, delta//2)
    ax, ay = coords[path[0]]
    bx, by = coords[path[-1]]
    img = cv2.line(img, (ax, ay), (ax, by), GREEN, delta//2)
    img = cv2.line(img, (ax, by), (bx, by), GREEN, delta//2)
    return img

def generate_input(coords, subset, train_input, i):
    city = generate_city(coords)
    city = draw_dots(city, subset, RED)
    cv2.imwrite(os.path.join(train_input, "input_%05d.png" %i), city)

def generate_image_pair(coords, subset, train_input, train_output, i, solver):
    city = generate_city(coords)
    dists = generate_manhattan_distances(subset)
    tsp_sol = solver(dists)
    city = draw_dots(city, subset, RED)
    cv2.imwrite(os.path.join(train_input, "input_%05d.png" %i), city)
    soln = np.zeros(size)
    soln = generate_manhattan_solution(soln, subset, tsp_sol[0])
    soln = draw_dots(soln, subset, RED)
    cv2.imwrite(os.path.join(train_output, "output_%05d.png" %i), soln)

def generate_tsp_sol(subset):
    dists = generate_manhattan_distances(subset)
    print("started")
    tsp_sol = held_karp(dists)
    print("ended")
    return tsp_sol

def generate_tuples(coords, train_input, train_output, num_cities, i, solver):
    subset = random.sample(coords, num_cities)
    return (coords, subset, train_input, train_output, i, solver)

def main():
    start = timer()
    size_name = "%dx%d" %(size[0], size[0])
    num_roads = 20
    num_cities = 30
    solver = googleortools.main
    roads_name = "%d" %num_cities
    train_input = os.path.join(size_name, roads_name, "train", "input")
    train_output = os.path.join(size_name, roads_name, "train", "output")
    test_input = os.path.join(size_name, roads_name, "test", "input")
    test_output = os.path.join(size_name, roads_name, "test", "output")
    folders = [train_input, train_output, test_input, test_output]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    random.seed(1)
    coords = generate_coordinates(num_roads)
    values = [generate_tuples(coords, train_input, train_output, num_cities, i, solver) for i in range(10000)]

    print(f'starting computations on {cpu_count()} cores')

    with Pool() as pool:
        res = pool.starmap(generate_image_pair, values)
        print(res)

    end = timer()
    print(f'elapsed time: {end - start}')

    values = [generate_tuples(coords, test_input, test_output, i) for i in range(1000)]
    print(f'starting computations on {cpu_count()} cores')

    with Pool() as pool:
        res = pool.starmap(generate_image_pair, values)
        print(res)

    end = timer()
    print(f'elapsed time: {end - start}')



if __name__ == '__main__':
    # compute_tsp_sol()
    main()
    # a = open("tspdump.pickle", "rb")
    # b = pickle.load(a)
    # i = 0
    # generate_image_pair(b["coords"], b[i][1], ".", ".", i)
    # tsp_sol = generate_tsp_sol(subset)
    # start = time.time()
    # generate_image_pair(*values[0])
    # end = time.time()
    # print(end - start)

    # random.seed(1)
    # num_roads = 20
    # num_cities = 100
    # train_input = "."
    # train_output = "."
    # coords = generate_coordinates(num_roads)
    # size_name = "512x512"
    # roads_name = "100"
    # train_input = os.path.join(size_name, roads_name, "train", "input")
    # train_output = "."
    # if not os.path.exists(train_input):
    #     os.makedirs(train_input)
    # values = [generate_tuples(coords, train_input, train_output, num_cities, i) for i in range(10000)]
    # for value in values:
    #     coords, subset, train_input, train_output, i = value
    #     generate_input(coords, subset, train_input, i)

    