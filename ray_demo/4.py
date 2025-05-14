import ray
import os
import time
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
base_dataset_path='/data/wangchunhao-slurm/workspace/code/projects/data/CIR_dataset/CIRR'
image_path_list=[]

for root,dirs,names in os.walk(base_dataset_path):
    for name in names:
        if name.endswith('.png') or name.endswith('.jpg'):
            image_path=os.path.join(root,name)
            image_path_list.append(image_path)
            if len(image_path_list)>20000:
                break

def concurrent_method():
    def _read_img(path):
        img=Image.open(path)
    t1=time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(tqdm(executor.map(_read_img,image_path_list)))
    print(time.time()-t1)

def normal_method():
    t1=time.time()
    for image_path in tqdm(image_path_list):
        img=Image.open(fp=image_path)
    print(time.time()-t1) 

def ray_method():
    @ray.remote
    def read_img(img_path_list):
        #res=[]
        for i,image_path in tqdm(enumerate(img_path_list)):
            img=Image.open(fp=image_path)
            #res.append(img)
        #return res
    t1=time.time()
    chunk_size = len(image_path_list) // 4

    futures = []
    for i in tqdm(range(4)):
        start = i * chunk_size
        end = (i+1) * chunk_size if i < 3 else len(image_path_list)
        futures.append(read_img.remote(image_path_list[start:end]))

    ray.get(futures)
    print(time.time()-t1)

concurrent_method()

