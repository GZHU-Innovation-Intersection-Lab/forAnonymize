# /data_huawei/jiakun/AAMAS/src/utils/random.py
import random, numpy as np, os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)