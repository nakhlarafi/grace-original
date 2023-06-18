import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle
import pdb
project = sys.argv[1]
card = [0]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {'Time':5, 'Math':2, "Lang":1, "Chart":3, "Mockito":4, "Closure":1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum
lr = 1e-2
seed = 0
batch_size = 60
print('lst------------- ', lst)
print('times',int(len(lst) / totalnum) + 1)
pdb.set_trace()
for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        cardn =int(j / singlenum)
        print("CUDA_VISIBLE_DEVICES="+str(card[cardn]))
        # p = subprocess.Popen("CUDA_VISIBLE_DEVICES=1,2,3" + " python run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)
        # p = subprocess.Popen("CUDA_VISIBLE_DEVICES=" + " torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:quail:127.0.1.1:29400 run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=virya4:29407 run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)

        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("python3 sum.py %s %d %f %d"%(project, seed, lr, batch_size), shell=True)
p.wait()
subprocess.Popen("python3 watch.py %s %d %f %d"%(project, seed, lr, batch_size),shell=True)            