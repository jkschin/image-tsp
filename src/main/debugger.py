import os
import cv2
import numpy as np
import decoder
import pickle


def test_overlap():
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img >= 1] = 1
        return img

    def subroutine(img_a, img_b):
        img_a = preprocess(img_a)
        img_b = preprocess(img_b)
        img_c = np.logical_and(img_a, img_b)
        return img_c * 255

    nn_outputs_path = "expt/1-1150-20220110-140212/test-200"
    gt_path = "data/images/200/test/output"
    id = 10144
    nn = os.path.join(nn_outputs_path, "output_%05d.png" %id)
    gt = os.path.join(gt_path, "output_%05d.png" %id)
    img_a = cv2.imread(nn)
    img_b = cv2.imread(gt)
    img_c = subroutine(img_a, img_b)
    img_c = np.stack([img_c]*3, axis=2)
    imgs = [img_a, img_b, img_c]
    img_out = np.concatenate(imgs, axis=1)
    cv2.imwrite("test.png", img_out)


def pruning_analysis(id):
    test_dir = "expt/1-1150-20220110-140212/test-200" 
    data = "data/pickles/TSP200.pickle"
    data = pickle.load(open(data, "rb"))
    print("Running: ", id)
    cnn = decoder.analyze(test_dir, id, data["test"][id], naive=False)
    naive = decoder.analyze(test_dir, id, data["test"][id], naive=True)
    # naive = False
    return cnn, naive

# [(10072, 6214, 6904, 16215), (10088, 6360, 6884, 16132), (10098, 6448, 7096, 16132), (10137, 6572, 6668, 16289), (10413, 6552, 7094, 16412), (10421, 6250, 6552, 16058), (10535, 6684, 7354, 16404), (10552, 6482, 6976, 16243), (10677, 6632, 6920, 16378), (10678, 6428, 6832, 16210), (10691, 6674, 6834, 16188), (10907, 6332, 6650, 16060), (10984, 6618, 7240, 16407)]
  
if __name__ == "__main__":
    ids = [10072, 10088, 10098, 10137, 10413, 10421, 10535, 10552, 10677, 10678, 10691, 10907, 10984]
    ids = [10413, 10677, 10678, 10691, 10984]
    ids = [10678, 10691]
    cnns = []
    naives = []
    for id in ids:
        cnn, naive = pruning_analysis(id)
        cnns.append(cnn)
        naives.append(naive)
    