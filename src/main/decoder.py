import pickle
from src.main.generate_data import generate_manhattan_distances
from src.main.postprocess import *
import src.solver.googleortools as gortools
import src.visualization.visualization as viz
import cv2
import os
from multiprocessing import Pool, cpu_count
import sys


def get_adjacency_matrix(dic):
    num_cities = len(dic)
    matrix = np.zeros((num_cities, num_cities), np.uint8)
    for k, v in dic.items():
        nodes = dic[k].keys()
        for node in nodes:
            matrix[k][node] = 1
            matrix[node][k] = 1
    return matrix


def analyze(test_dir, id, data, naive=False):
    """
    Args:
        test_dir: relative directory of image outputs from CNN.
        Example: "expt/1-1150-20220110-140212/test-200"
        This folder contains all the images from running the model 
        1-1150-20220110-140212 on the 200 city TSP.

        id (int): ID of image to run test on. Range from 10000 to 10999.

        data (dict): The data dictionary for the specific id.
        Example:
        data = "data/pickles/TSP200.pickle"
        data = pickle.load(open(data, "rb"))
        data["test"][id]

        naive (boolean): Whether to run the naive version or not.

    Returns:
        (tour distance, tour)

    Raises:

    """ 

    # Get Prediction Image and City Image
    size = (512, 512, 3)
    filepath = os.path.join(test_dir, "output_%05d.png" %id)
    subset = data["delivery_locations"]
    orig_z = data["results"][0]
    pred_img = cv2.imread(filepath)
    print(filepath)
    pred_img[np.where((pred_img == viz.RED).all(axis=2))] = [1, 1, 0]
    pred_img[np.where((pred_img == viz.GREEN).all(axis=2))] = [1, 0, 0]
    pred_img = pred_img[:, :, 0]
    city_img = get_one_pixel_city_image(size, subset)

    # If not naive, use pred_img obtained above.
    if not naive:
        dic = get_edge_dictionary(subset, pred_img, city_img)
    # If naive, just send in an image of ones. It's as good as 
    # "all paths" can be used.
    else:
        pred_img = np.ones(pred_img.shape)
        dic = get_edge_dictionary(subset, pred_img, city_img)

    # Get adjacency matrix and print the shapes. It gives an indication of whether naive is run or not.
    adj_matrix = get_adjacency_matrix(dic)
    print("Naive: %s | Adj Matrix Size: %d | Adj Matrix Shape: %s" %(naive, np.sum(adj_matrix), adj_matrix.shape))
    orig_dists = generate_manhattan_distances(subset)
    dists = orig_dists * adj_matrix
    dists[dists == 0] = 1000000
    routes = gortools.main(dists)
    pred_z = get_tour_length(routes[1], orig_dists)
    print("ID: %d Orig-Z: %d NN-Z: %d 10K: %d" %(id, orig_z, pred_z, routes[0]))
    return pred_z, routes, dic


def run(expt_path, num_cities):
    test_dir = os.path.join(expt_path, "test-%d" %num_cities)
    output_path = os.path.join(expt_path, "decode-%d" %num_cities)
    picklefile = "data/pickles/TSP%d.pickle" %num_cities
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = pickle.load(open(picklefile, "rb"))
    values = [(test_dir, id, data["test"][id]) for id in range(10000, 11000)]

    num_cores = cpu_count() // 2 - 1
    print(f'starting computations on {num_cores} cores')

    with Pool(num_cores) as pool:
        res = pool.starmap(analyze, values)
    with open(os.path.join(expt_path, "testcnn60s-%d.pickle" %num_cities), "wb") as f:
        pickle.dump(res, f, protocol=2)

def run_naive(expt_path, num_cities):
    test_dir = os.path.join(expt_path, "test-%d" %num_cities)
    picklefile = "data/pickles/TSP%d.pickle" %num_cities
    data = pickle.load(open(picklefile, "rb"))
    values = [(test_dir, id, data["test"][id], True) for id in range(10000, 11000)]

    num_cores = cpu_count() // 2 - 1
    print(f'starting computations on {num_cores} cores')

    with Pool(num_cores) as pool:
        res = pool.starmap(analyze, values)
    with open(os.path.join(expt_path, "testnaive60s-%d.pickle" %num_cities), "wb") as f:
        pickle.dump(res, f, protocol=2)


def debug(expt_path, num_cities, id):
    test_path = os.path.join(expt_path, "test-%d" %num_cities)
    output_path = os.path.join(expt_path, "decode-%d" %num_cities)
    gt_path = "data/images/%d/test/output" %num_cities
    paths = [gt_path, test_path]
    imgs = []
    for path in paths:
        img_path = os.path.join(path, "output_%05d.png" %id)
        imgs.append(cv2.imread(img_path))
        print(img_path)
    img_path = os.path.join(output_path, "output_%05d.png" %id)
    print(img_path)
    imgs.append(cv2.imread(img_path))
    out = np.concatenate(imgs, axis=1)
    cv2.imwrite("debug.png", out)


def opt_gap(picklefile):
    data = pickle.load(open(picklefile, "rb"))
    total_z = 0
    total_diff = 0
    same = 0
    for a, b, c in data:
        if b == c:
            same += 1
        else:
            diff = abs(b-c)
            total_diff += diff
            total_z += b
        if c >= 10000:
            raise Exception("Above 10K detected!")
    return total_diff / total_z, same


if __name__ == "__main__":
    expt1 = "expt/1-1150-20220110-140212"
    expt20 = "expt/20-1150-20220109-144909"
    expt30 = "expt/30-1150-20220109-144910"
    expt100= "expt/100-1150-20220109-144921"
    expt200= "expt/200-1150-20220109-144932"
    # folders = [expt1, expt20, expt30, expt100, expt200]
    folders = [expt20, expt30, expt100, expt200]

    expt_path = expt20

    if sys.argv[1] == "runall":
        for expt_path in folders:
            for num_cities in [20, 30, 100, 200]:
                run(expt_path, num_cities)
    elif sys.argv[1] == "runone":
        num_cities = 200
        expt_path = expt1
        run_naive(expt_path, num_cities)
    elif sys.argv[1] == "debug":
        num_cities = 200
        id = int(sys.argv[2])
        debug(expt_path, num_cities, id)
    elif sys.argv[1] == "optgap":
        picklefile = sys.argv[2]
        gap, num_same = opt_gap(picklefile)
        print(gap, num_same)
    elif sys.argv[1] == "analyzeone":
        num_cities = 200
        test_dir = os.path.join(expt1, "test-%d" %num_cities)
        picklefile = "data/pickles/TSP%d.pickle" %num_cities
        data = pickle.load(open(picklefile, "rb"))
        id = 10258
        ans = cv2.imread("data/images/%d/test/output/output_%05d.png" %(num_cities, id))
        print("CNN")
        analyze(test_dir, id, data["test"][id], "analysis", False, ans)
        print("Naive")
        analyze(test_dir, id, data["test"][id], "analysis", True, ans)
    elif sys.argv[1] == "naive":
        num_cities = 200
        expt_path = expt1
        run_naive(expt_path, num_cities)
        # folders = [expt200]
        # for expt_path in folders:
        #     for num_cities in [200]:
        #         run_naive(expt_path, num_cities)

    