import plotly.graph_objects as go
import plotly.express as px
import pickle

import numpy as np

def cdf_analysis(cnn, naive):
    data = pickle.load(open(cnn, "rb"))
    x0 = [i[2] for i in data]
    print("CNN")
    print("Mean: %.2f Std: %.2f" %(np.mean(x0), np.std(x0)))

    data = pickle.load(open(naive, "rb"))
    x1 = [i[2] for i in data]
    print("Naive")
    print("Mean: %.2f Std: %.2f" %(np.mean(x1), np.std(x1)))
    fig = px.ecdf({
      "cnn": x0, "naive":x1})
    fig.write_image("test.png")

    print("*************")
    print("Trim Outliers")
    print("*************")

    data = pickle.load(open(cnn, "rb"))
    x0 = [i[2] for i in data if i[3] < 10000]
    print("CNN", len(x0))
    print("Mean: %.2f Std: %.2f" %(np.mean(x0), np.std(x0)))

    data = pickle.load(open(naive, "rb"))
    x1 = [i[2] for i in data if i[3] < 10000]
    print("Naive", len(x1))
    print("Mean: %.2f Std: %.2f" %(np.mean(x1), np.std(x1)))

if __name__ == "__main__":
    # cnn = "expt/1-1150-20220110-140212/testcnn-200.pickle"
    # naive = "expt/1-1150-20220110-140212/testnaive-200.pickle"
    cnn = "expt/1-1150-20220110-140212/testcnn10s-200.pickle"
    naive = "expt/1-1150-20220110-140212/testnaive10s-200.pickle"
    cdf_analysis(cnn, naive)
