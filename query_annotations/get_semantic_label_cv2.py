import argparse
import os
import sys
import json
import numpy as np
import cv2

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from assets import utils

def get_label_for_pixel(semantic_image_path, labels_path, x, y):
    """
    Retrieves the semantic label for a specific pixel using OpenCV.
    """
    if not os.path.exists(semantic_image_path):
        return f"Error: Image file not found at {semantic_image_path}"
    
    if not os.path.exists(labels_path):
        return f"Error: Labels file not found at {labels_path}"

    # Load the list of semantic labels
    labels = utils.load_labels(labels_path)

    # Open the image with OpenCV (it loads as a NumPy array in BGR format)
    img_bgr = cv2.imread(semantic_image_path)
    if img_bgr is None:
        return f"Error: Could not read image at {semantic_image_path}"
    
    # Convert from BGR to RGB to match the label index logic
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    height, width, _ = img_rgb.shape
    if x >= width or y >= height:
        return f"Error: Pixel coordinates ({x}, {y}) are out of bounds for image size ({width}, {height})."
    
    # Get pixel color using NumPy indexing (y, x)
    pixel_color = img_rgb[y, x]

    # Convert the RGB color to an index
    index = utils.get_index(tuple(pixel_color))

    # Check for missing data
    if tuple(pixel_color) == (13, 13, 13):
        return "No label (missing data)"

    # Retrieve the label using the index
    if index < len(labels):
        return labels[index]
    else:
        return f"Error: Calculated index {index} is out of bounds for the labels list (size {len(labels)})."

def get_all_labels_vectorized(semantic_image_array, labels_path):
    """
    This function is library-agnostic as it operates on NumPy arrays. No changes needed.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

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
    multipliers = np.array([256*256, 256, 1], dtype=np.int64)
    index_map = np.dot(img_array, multipliers)
    semantic_map = complete_labels_array[index_map]
    return semantic_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Get semantic labels from an image using OpenCV."
    )
    parser.add_argument(
        '--labels', 
        type=str, 
        default='assets/semantic_labels.json', 
        help="Path to the semantic_labels.json file."
    )
    
    args = parser.parse_args()

    # Hardcoded paths for testing
    args.image_path = 'area_3/pano/semantic/camera_0ccf3c78ef354902b516c62ef8fb7cf1_lounge_2_frame_equirectangular_domain_semantic.png'
    args.x = None
    args.y = None

    output_dir = 'query_annotations/'
    os.makedirs(output_dir, exist_ok=True)
    
    semantic_image_path = args.image_path
    rbg_image_path = semantic_image_path.replace('semantic', 'rgb')
    if not os.path.exists(semantic_image_path):
        raise FileNotFoundError(f"Image file not found at {semantic_image_path}")
    
    # Load semantic image with OpenCV
    semantic_img_bgr = cv2.imread(semantic_image_path)
    if semantic_img_bgr is None:
        raise IOError(f"Could not load image at {semantic_image_path}")
    
    # Convert to RGB and cast to int64 for processing
    semantic_img_rgb = cv2.cvtColor(semantic_img_bgr, cv2.COLOR_BGR2RGB)
    semantic_img_array = semantic_img_rgb.astype(np.int64)
    
    # The rest of the script assumes no cropping, so we use the full image
    # crop_box = (0, 0, semantic_img_array.shape[1], semantic_img_array.shape[0])
    # semantic_img_array = semantic_img_array[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]

    if args.x is not None and args.y is not None:
        label = get_label_for_pixel(args.image_path, args.labels, args.x, args.y)
        print(f"The semantic label at pixel ({args.x}, {args.y}) is: '{label}'")
    else:
        print("Processing all pixels...")
        all_labels = get_all_labels_vectorized(semantic_img_array, args.labels)
        print(f"Successfully generated a {all_labels.shape} map of semantic labels.")

        # --- Optimization: Parse unique labels once ---
        unique_labels, inverse_indices = np.unique(all_labels, return_inverse=True)
        parsed_classes = np.array(
            [utils.parse_label(label)['instance_class'] for label in unique_labels]
        )
        all_classes = parsed_classes[inverse_indices].reshape(all_labels.shape)
        # --- End Optimization ---

        # Load and resize mask using OpenCV
        all_labels_size = (all_labels.shape[1], all_labels.shape[0]) # (width, height)
        mask_path = 'query_annotations/cubemap_face_mask.png'
        cubemap_face_mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if cubemap_face_mask_array is None:
            raise IOError(f"Could not load mask at {mask_path}")
        
        # Resize mask to match image dimensions
        # cubemap_face_mask_array = cv2.resize(
        #     cubemap_face_mask_array, 
        #     dsize=all_labels_size, 
        #     interpolation=cv2.INTER_NEAREST
        # )

        CUBEMAP_FACES = {
            "+X": 1, "-X": 2, "+Y": 3, "-Y": 4, "+Z": 5, "-Z": 6
        }
        MASK_PIXEL_MULTIPLIER = 32
        CLASSES_TO_IGNORE = ['<MIS>', '<UNK>', 'ceiling', 'clutter', 'floor', 'wall']

        img_cls_fracs_dict = {
            "img_file": rbg_image_path,
            "proj_type": "cubemap"
        }

        for face_key, face_val in CUBEMAP_FACES.items():
            print("Starting for face", face_key)
            mask = (cubemap_face_mask_array == face_val * MASK_PIXEL_MULTIPLIER)
            
            if not mask.any():
                img_cls_fracs_dict[face_key] = {}
                continue

            face_classes = all_classes[mask]
            ignore_mask = np.isin(face_classes, CLASSES_TO_IGNORE)
            valid_face_classes = face_classes[~ignore_mask]
            valid_face_unique_classes, counts = np.unique(valid_face_classes, return_counts=True)
            
            img_cls_fracs_dict[face_key] = list(valid_face_unique_classes)
            print("Done\n")

        with open(os.path.join(output_dir, 'cls_fracs_cv2.json'), 'w+') as f:
            json.dump(img_cls_fracs_dict, f, indent=2)