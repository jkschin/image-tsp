import pickle
from src.main.generate_data import generate_manhattan_distances
from src.main.postprocess import *
import src.solver.googleortools as gortools
import src.visualization.visualization as viz
import cv2
import os
from multiprocessing import Pool, cpu_count

def draw_solution(size, subset, tour, delta):
    img = np.zeros(size, np.uint8)
    img = viz.draw_manhattan_solution(img, subset, tour, delta)
    img = viz.draw_dots(img, subset, delta, viz.RED, True)
    return img


def get_adjacency_matrix(dic):
    num_cities = len(dic)
    matrix = np.zeros((num_cities, num_cities), np.uint8)
    for k, v in dic.items():
        nodes = dic[k].keys()
        for node in nodes:
            matrix[k][node] = 1
            matrix[node][k] = 1
    return matrix


def analyze(test_dir, id, data, output_path):
    delta = 5
    size = (512, 512, 3)
    filepath = os.path.join(test_dir, "output_%05d.png" %id)
    subset = data["test"][id]["delivery_locations"]
    orig_z = data["test"][id]["results"][0]
    pred_img = cv2.imread(filepath)
    pred_img[np.where((pred_img == viz.RED).all(axis=2))] = [1, 1, 0]
    pred_img[np.where((pred_img == viz.GREEN).all(axis=2))] = [1, 0, 0]
    pred_img = pred_img[:, :, 0]
    city_img = get_one_pixel_city_image(size, subset)
    dic = get_edge_dictionary(subset, pred_img, city_img)
    adj_matrix = get_adjacency_matrix(dic)
    dists = generate_manhattan_distances(subset)
    dists = dists * adj_matrix
    dists[dists == 0] = 10000
    routes = gortools.main(dists)
    print("Orig-Z: %d NN-Z: %d" %(orig_z, routes[0]))
    gortools_img = draw_solution(size, subset, routes[1], delta)
    output_path = os.path.join(output_path, "decoded_%05d.png" %id)
    cv2.imwrite(output_path, gortools_img)


def run():
    picklefile = "data/pickles/TSP200.pickle"
    data = pickle.load(open(picklefile, "rb"))
    test_dir = "/Users/samuelchin/Desktop/test-200-1120"
    id = 10000
    output_path = "expt/20-1105-20220108-230102"
    values = [(test_dir, id, data) for id in range(10000, 11000)]