import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle
import GPUtil
import threading
project = sys.argv[1]
card = [0]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {'Time':1, 'Math':1, "Lang":1, "Chart":3, "Mockito":1, "Closure":1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum
lr = 1e-2
seed = 0
batch_size = 60

def print_gpu_memory_usage():
    while True:
        GPUs = GPUtil.getGPUs()
        for gpu in GPUs:
            print(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        time.sleep(5)  # Print every 5 seconds

threading.Thread(target=print_gpu_memory_usage, daemon=True).start()

for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        cardn =int(j / singlenum)
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python3 run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)
        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("python3 sum.py %s %d %f %d"%(project, seed, lr, batch_size), shell=True)
p.wait()
subprocess.Popen("python3 watch.py %s %d %f %d"%(project, seed, lr, batch_size),shell=True)            