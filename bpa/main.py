import argparse
import os
import pickle

from utils import io
from bpa import BPA_solver

# radius_list_map = {
#     "bunny.obj": [0.2, 0.3, 0.5],
#     "bunny2.obj": [0.1, 0.2],
#     "teapot.obj": [0.3],
#     "cow.obj": [0.04, 0.1, 0.2],
#     "homer.obj": [0.1, 0.2, 0.3],
# }

def main(args):
    mesh = io.read_obj_file(args.in_file)
    # radius_list = radius_list_map[os.path.basename(args.in_file)]
    bpa_solver = BPA_solver(mesh["v"], mesh["vn"])
    resulting_mesh = bpa_solver.solve()
    # with open(args.out_file + ".pkl", "wb") as f:
    #     pickle.dump(bpa_solver.faces, f)
    io.write_obj_file(args.out_file, resulting_mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', '-i', type=str, default='data/bunny.obj')
    parser.add_argument('--out_file', '-o', type=str, default='output')
    args = parser.parse_args()
    if os.path.isdir(args.out_file):
        args.out_file = os.path.join(args.out_file, os.path.basename(args.in_file))
    main(args)