import argparse
import glob
import os
import sys
import numpy as np
from PIL import Image
from datasets import Dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from assets import utils

CUBEMAP_FACES = {
        "+X": 1,  # Front
        "-X": 2,  # Back
        "+Y": 3,  # Right
        "-Y": 4,  # Left
        "+Z": 5,  # Top
        "-Z": 6,  # Bottom
    }
PIXEL_MASK_MULTIPLIER = 32
CLASSES_TO_IGNORE = [
    '<MIS>',
    '<UNK>',
    'ceiling',
    'clutter',
    'floor',
    'wall',
]

def get_all_labels_vectorized(semantic_image_array, labels_path):
    """
    Retrieves the semantic labels for every pixel in a semantic image using vectorized operations.

    Args:
        semantic_image_path (str): Path to the semantic segmentation image.
        labels_path (str): Path to the semantic_labels.json file.

    Returns:
        numpy.ndarray: A 2D array of strings with the semantic label for each pixel.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    # Load the list of semantic labels and convert to a NumPy array for fast indexing
    labels_list = utils.load_labels(labels_path)
    missing_data_RGB = (13,13,13)
    missing_data_index = utils.get_index(missing_data_RGB)
    labels_array = np.array(labels_list)
    complete_labels_array = np.full(
        (missing_data_index + 1),
        fill_value="<MIS>_0_<MIS>_0_0",
        dtype=labels_array.dtype
    )
    complete_labels_array[:labels_array.shape[0]] = labels_array

    img_array = semantic_image_array
    # Create a multiplier for vectorized conversion from RGB to a single index
    # [R, G, B] * [256*256, 256, 1] = R*256*256 + G*256 + B
    multipliers = np.array([256*256, 256, 1], dtype=np.int64)
    
    index_map = np.dot(img_array, multipliers)

    semantic_map = complete_labels_array[index_map]

    return semantic_map

def extract_cp_query_labels_from_img(semantic_image_path, cubemap_face_mask_array, labels_path, return_ds_dict=True):
    rbg_image_path = semantic_image_path.replace('semantic', 'rgb')
    if not os.path.exists(semantic_image_path):
        raise FileNotFoundError(f"Image file not found at {semantic_image_path}")
    
    with Image.open(semantic_image_path) as img:
        semantic_img_array = np.array(img.convert("RGB"), dtype=np.int64)
    
    all_labels = get_all_labels_vectorized(semantic_img_array, labels_path)
    
    # 1. Find all unique label strings in the entire image and their inverse mapping.
    unique_labels, inverse_indices = np.unique(all_labels, return_inverse=True)
    
    parsed_classes = np.array(
        [utils.parse_label(label)['instance_class'] for label in unique_labels]
    )

    all_classes = parsed_classes[inverse_indices].reshape(all_labels.shape)

    if return_ds_dict:
        img_dict = {
            'image_file': [],
            'query': [],
            'face_id': []
        }
    else:
        img_dict = {
            "img_file": rbg_image_path,
            "proj_type": "cubemap"
        }

    for face_key in CUBEMAP_FACES.keys():
        mask = (cubemap_face_mask_array == CUBEMAP_FACES[face_key] * PIXEL_MASK_MULTIPLIER)
        
        if not mask.any():
            img_dict[face_key] = {}
            continue

        face_classes = all_classes[mask]

        ignore_mask = np.isin(face_classes, CLASSES_TO_IGNORE)
        valid_face_classes = face_classes[~ignore_mask]
        valid_face_unique_classes = np.unique(valid_face_classes)
                
        if return_ds_dict:
            for valid_class in valid_face_unique_classes:
                img_dict['image_file'].append(rbg_image_path)
                img_dict['query'].append(valid_class)
                img_dict['face_id'].append(CUBEMAP_FACES[face_key])
        else:
            img_dict[face_key] = list(valid_face_unique_classes)
    return img_dict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--area_dir')
    parser.add_argument('--num_proc', default=4)
    return parser.parse_args()

def create_query_annots_ds_for_area(
        cubemap_face_mask_array,
        area_dir,
        labels_path,
        num_proc=4
):
    semantic_imgs_dir = os.path.join(area_dir, 'pano/semantic')
    semantic_imgs_files = glob.glob(os.path.join(semantic_imgs_dir, '**', '*.png'), recursive=True)

    img_ds  = Dataset.from_dict({'image_file':semantic_imgs_files})
    img_ds = img_ds.map(
        lambda example: extract_cp_query_labels_from_img(
            semantic_image_path=example['image_file'],
            cubemap_face_mask_array=cubemap_face_mask_array,
            labels_path=labels_path
        ),
        remove_columns=['image_file'], # the function returns the the respective 'image_file' column
        num_proc=num_proc,
        batched=False
    )

    img_ds = img_ds.map(
        lambda x: {
            'image_file': [file
                           for files in x['image_file']
                           for file in files],
            'query': [q
                      for queries in x['query']
                      for q in queries],
            'face_id': [id
                        for ids in x['face_id']
                        for id in ids]
        },
    batched=True,
    num_proc=num_proc)
    
    return img_ds

if __name__ == '__main__':

    labels_path = 'assets/semantic_labels.json'
    output_dir = 'query_annotations/'
    cubemap_face_mask_array = np.array(
        Image.open('query_annotations/cubemap_face_mask.png')
    )
    args = parse_args()
    area_dir = args.area_dir
    img_ds = create_query_annots_ds_for_area(
        cubemap_face_mask_array=cubemap_face_mask_array,
        area_dir=area_dir,
        num_proc=args.num_proc,
        labels_path=labels_path
    )

    img_ds.save_to_disk(
        os.path.join(output_dir, f'2d-3d-semantics_{area_dir}_cp_query_annots')
    )
