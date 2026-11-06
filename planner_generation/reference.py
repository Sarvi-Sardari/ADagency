"""Select representative reference images using CLIP embeddings."""

from .utils import list_images, preprocess_image_for_embedding
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from pathlib import Path
from .config import DatasetConfig, ModelsConfig
from .models import Encoder

logger = logging.getLogger(__name__)


def reference_image_creator(
    dataconfig: DatasetConfig,
    modelcfg: ModelsConfig,
    ref_img_folder_path: Path,
    dataset_name: str,
    class_labels_filepath: dict[str, Path],
    num_of_ref_img: int,
) -> list[list[str]]:

    """Select reference images and return source paths grouped by class.

    The selected originals remain in the dataset's train/good folder.
    Class names come from class_labels_filepath and are returned explicitly,
    so no intermediate copy is needed to recover the class name from an
    image's parent directory.
    """
    
    # Define a set of valid image extensions

    list_of_cand_img_filepath, _ = list_images(ref_img_folder_path)
    class_label_dict = {}
    for pos, label in enumerate(class_labels_filepath.keys()):
        class_label_dict[pos] = label
    ref_img_tensors = preprocess_image_for_embedding(Encoder, list(class_labels_filepath.values()))
    cand_img_tensors = preprocess_image_for_embedding(Encoder, list_of_cand_img_filepath)
    ref_img_tensors = ref_img_tensors.to(modelcfg.vlm.device)
    cand_img_tensors = cand_img_tensors.to(modelcfg.vlm.device)

    with torch.no_grad():
        ref_emb = Encoder.model.encode_image(ref_img_tensors)
        cand_emb = Encoder.model.encode_image(cand_img_tensors)

    # Normalize embeddings (IMPORTANT)
    ref_emb = ref_emb / ref_emb.norm(dim=1, keepdim=True)
    cand_emb = cand_emb / cand_emb.norm(dim=1, keepdim=True)

    ref_emb = ref_emb.cpu().numpy()
    cand_emb = cand_emb.cpu().numpy()
    distances = pairwise_distances(cand_emb, ref_emb, metric="cosine")
    min_column_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    merged_data = np.column_stack((min_column_indices, min_distances))
    # 2. Create the DataFrame
    dataframe = pd.DataFrame(merged_data, columns=['Predicted_label', 'minimum_distance'])
    df_indexed = dataframe.reset_index().rename(columns={'index': 'original_index'})

    # 2. Take the top N smallest distances per predicted class.  Using
    # groupby.apply here is fragile across pandas versions because newer
    # versions may exclude the grouping column from the applied frame.
    top_3_confident = (
        df_indexed
        .sort_values('minimum_distance')
        .groupby('Predicted_label', sort=False)
        .head(num_of_ref_img)
        .reset_index(drop=True)
    )

    # 3. Select and display the specific columns you requested
    result_display = top_3_confident[['original_index', 'Predicted_label', 'minimum_distance']]

    logger.debug("Reference image selection result:\n%s", result_display.to_string())
    index_list = result_display['original_index'].tolist()
    logger.debug("Selected candidate indices: %s", index_list)
    if len(list_of_cand_img_filepath) == len(min_column_indices):
        selected_paths_by_label = {label: [] for label in class_label_dict.values()}
        for i in index_list:
            label = class_label_dict[int(min_column_indices[i])]
            selected_paths_by_label[label].append(list_of_cand_img_filepath[i])
        logger.debug("Selected reference paths by class: %s", selected_paths_by_label)
        return [selected_paths_by_label[label] for label in class_label_dict.values()]

    else:
        raise ValueError('The total image inputs and the embedding similarity output for each input count does not match')

def reference_image_selector(dataconfig: DatasetConfig, modelcfg: ModelsConfig, num_of_ref_img:int,ref_img_folder_path:Path,dataset_name,is_class:bool,class_labels_filepath=None,) -> list:
    
    """Description: This function if the dataset is multiclass uses the reference_image_creator to separate the good images folder into each of its class labels if its single class take n random images from good or anomalous image folder\n
    Input: num_of_ref_img:number of reference image, ref_img_folder_path: reference image folder path, dataset_name: name of the dataset, is_class: multi or single class dataset, class_labels_filepath: dictioanry containing class labels as key and reference image filepath for each calss label\n
    output: list of reference image file path""" 

    if is_class:
        if class_labels_filepath == None:
            raise ValueError('If is_class set to True pass argument each_class-instance_filepath as dictionary of class labels and its corresponding image filepath')
        else:
            return reference_image_creator(
                dataconfig,
                modelcfg,
                ref_img_folder_path,
                dataset_name,
                class_labels_filepath,
                num_of_ref_img,
            )
    else:
        ref_img_files , ref_img_filenames= list_images(ref_img_folder_path)
        if dataconfig.reference_selection.explicit_images:
            explicit_images = [str(path) for path in dataconfig.reference_selection.explicit_images]
            missing = [path for path in explicit_images if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"Configured explicit reference images do not exist: {missing}")
            return explicit_images
        if num_of_ref_img > len(ref_img_files):
            raise ValueError(
                f"Requested {num_of_ref_img} reference images, but only {len(ref_img_files)} "
                f"were found in {ref_img_folder_path}."
            )
        rng = np.random.default_rng(dataconfig.reference_selection.random_seed)
        return rng.choice(ref_img_files, num_of_ref_img, replace=False).tolist()
