import re
import pickle

def build_min_list(l):
    min_val = 100000
    output = []
    for value in l:
        min_val = min(value, min_val)
        output.append(min_val)
    return output


def split_into_experiments(l):
    # Original first
    types = ["orig", "trunc"]
    num_cities = [20, 30, 100, 200]
    i = 0
    interval = 100
    dic = {}
    for t in types:
        for c in num_cities:
            key = "%s-%d" %(t, c)
            dic[key] = l[i:i+interval]
            i += interval
    return dic


def parse_log():
    STEPPING = "STEPPING"
    EXTRACT = "EXTRACT"

    mode = STEPPING
    f = open("out.log", "r")
    all = []
    mins = []
    for line in f:
        if "Start search" in line:
            mode = EXTRACT
            continue
        elif "End search" in line:
            mins = build_min_list(mins)
            all.append(mins)
            mins = []
            mode = STEPPING
            continue
        if mode == EXTRACT:
            if "Solution #" in line:
                res = re.findall(r'\(.*?\)', line)
                # obj_value = int(res[0].split(",")[1].split(" = ")[1])
                obj_value = int(res[0].split(",")[0][1:])
                mins.append(obj_value)
    dic = split_into_experiments(all)
    with open("processed.pickle", "wb") as f:
        pickle.dump(dic, f, protocol=2)


import time
from collections import defaultdict
def parse_log_v2():
    STEPPING = "STEPPING"
    EXTRACT = "EXTRACT"

    mode = STEPPING
    f = open("out.log", "r")
    all = []
    dic = defaultdict(list)
    ss_count = 0
    for line in f:
        if "Start search" in line:
            mode = EXTRACT
            ss_count += 1
            continue
        elif "End search" in line:
            ans = []
            for i in range(30):
                try:
                    value = min(dic[i])
                    if i == 0 or value <= ans[-1]:
                        ans.append(value)
                    else:
                        ans.append(ans[-1])
                except ValueError:
                    ans.append(ans[-1])
            all.append(ans)
            dic = defaultdict(list)
            mode = STEPPING
            continue
        if mode == EXTRACT:
            if "Solution #" in line:
                res = re.findall(r'\(.*?\)', line)
                m = re.findall(r"time = \d+ ms", res[0])
                t = int(m[0].split(" = ")[1].split(" ")[0])
                obj_value = int(res[0].split(",")[0][1:])
                dic[t//1000].append(obj_value)
                # if ss_count == 5:
                #     if t // 1000 == 22:
                #         time.sleep(1)
                #     print(line)
                #     print(mins)
                #     print(obj_value, t)
                    # print(obj_value, tmp, mins, t)
    dic = split_into_experiments(all)
    with open("processed2.pickle", "wb") as f:
        pickle.dump(dic, f, protocol=2)


import numpy as np
def plot():
    data = pickle.load(open("processed2.pickle", "rb"))
    print(data["orig-20"][5])
    for k, v in data.items():
        arr = np.array(data[k])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        upper = mean + std
        lower = mean - std

if __name__ == "__main__":
    parse_log_v2()
    # plot()
    
