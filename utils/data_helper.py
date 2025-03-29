from pathlib import Path
import numpy as np



def read_file(file_path):
    path = Path(file_path)



# Reading fvecs
def fvecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view('float32')

# Reading ivecs
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# Writing fvecs
def write_fvecs(filename, vecs):
    with open(filename, "wb") as f:
        dim = vecs.shape[1]
        for i, vec in enumerate(vecs):
            f.write(np.array([dim], dtype='int32').tobytes())
            f.write(vec.astype('float32').tobytes())


def explore_data(dir_path):
    path = Path(dir_path)
    for item in path.iterdir():
        print(item)
        print(fvecs_read(item))
        #print(item.name.endswith(".fvecs"))
        
        

dir_path = "data/siftsmall/siftsmall/"
explore_data(dir_path=dir_path)

#print(fvecs_read("/home/satyam/github_repos/semantic_cache_and_RAG/data/siftsmall/siftsmall/siftsmall_base.fvecs").shape)