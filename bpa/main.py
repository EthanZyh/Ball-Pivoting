import argparse
import os

from utils import io
from bpa import BPA_solver

def main(args):
    mesh = io.read_obj_file(args.in_file)
    resulting_mesh = BPA_solver(mesh["v"], mesh["vn"]).solve()
    io.write_obj_file(args.out_file, resulting_mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', '-i', type=str, default='data/bunny.obj')
    parser.add_argument('--out_file', '-o', type=str, default='output')
    args = parser.parse_args()
    if os.path.isdir(args.out_file):
        args.out_file = os.path.join(args.out_file, os.path.basename(args.in_file))
    main(args)