import os
import argparse
from tqdm import tqdm
from decalib.deca import DECA
from decalib.datasets import TestData
import torch

def main(input_path, output_path, is_crop=True):
    os.makedirs(output_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    deca = DECA(config=None, device=device)
    testdata = TestData(input_path, iscrop=is_crop)

    for i, batch in enumerate(tqdm(testdata)):
        name = batch['name'][0]
        codedict = deca.encode(batch)
        opdict, visdict = deca.decode(codedict)
        deca.save_obj(os.path.join(output_path, name + '.obj'), opdict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath', type=str, default='frames_biden', help='Path to image frames')
    parser.add_argument('--outputpath', type=str, default='frames_biden/results', help='Where to save meshes')
    parser.add_argument('--iscrop', default=True, type=bool, help='Whether to crop faces')
    args = parser.parse_args()

    main(args.inputpath, args.outputpath, args.iscrop)
