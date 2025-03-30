from pathlib import Path
import numpy as np


def read_file(pathlib_path):
    a = np.fromfile(pathlib_path, dtype="int32")
    d = a[0]

    if pathlib_path.name.endswith(".fvecs"):
        return a.reshape(-1, d + 1)[:, 1:].view("float32")
    elif pathlib_path.name.endswith(".ivecs"):
        return a.reshape(-1, d + 1)[:, 1:].copy()
    else:
        return None


# Reading fvecs
def fvecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view("float32")


# Reading ivecs
def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


# Writing fvecs
def write_fvecs(filename, vecs):
    with open(filename, "wb") as f:
        dim = vecs.shape[1]
        for i, vec in enumerate(vecs):
            f.write(np.array([dim], dtype="int32").tobytes())
            f.write(vec.astype("float32").tobytes())


def explore_data(dir_path):
    path = Path(dir_path)
    for item in path.iterdir():
        res = read_file(item)
        print(f"file_name: {item}\n")
        print(f"data_shape: {res.shape}")
        print(f"data_sample: {res[0]}\n\n")


dir_path = "data/siftsmall/siftsmall/"
explore_data(dir_path=dir_path)

# print(fvecs_read("/home/satyam/github_repos/semantic_cache_and_RAG/data/siftsmall/siftsmall/siftsmall_base.fvecs").shape)
