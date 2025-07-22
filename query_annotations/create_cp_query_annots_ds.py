import os
import sys
import argparse
import numpy as np
from PIL import Image
from datasets import DatasetDict
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from query_annotations.create_cp_query_annots_ds_for_area import create_query_annots_ds_for_area

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_proc', default=4, type=int)
    return parser.parse_args()

area_dirs = (
    'area_1',
    'area_2',
    'area_3',
    'area_4',
    'area_5a',
    'area_5b',
    'area_6'
)

if __name__ == "__main__":
    labels_path = 'assets/semantic_labels.json'
    output_dir = 'query_annotations/'
    cubemap_face_mask_array = np.array(
        Image.open('query_annotations/cubemap_face_mask.png')
    )
    
    args = parse_args()
    query_annots_ds = DatasetDict()

    for area_dir in area_dirs:
        tqdm.write(f"Processing {area_dir}...")
        query_annots_ds[area_dir] = create_query_annots_ds_for_area(
            cubemap_face_mask_array=cubemap_face_mask_array,
            area_dir=area_dir,
            num_proc=args.num_proc,
            labels_path=labels_path
        )

    ds_output_dir = os.path.join(output_dir, '2d-3d-semantics_cp_query_annots')
    query_annots_ds.save_to_disk(ds_output_dir)
    print('Saved query annotations dataset at', ds_output_dir)