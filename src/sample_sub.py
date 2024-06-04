import gc
import itertools
import sys
from copy import deepcopy
from pathlib import Path
from time import sleep, time
from typing import Any

import h5py
import kornia as K
import kornia.feature as KF
import numpy as np
import pandas as pd
import pycolmap
import torch
import torch.nn.functional as F
from lightglue import ALIKED, LightGlue, match_pair
from torch import Tensor as T
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Data importing into colmap
# Provided by organizers
sys.path.append("/workspace/colmap-db-import")
from database import COLMAPDatabase
from h5_to_db import add_keypoints, add_matches


def arr_to_str(a):
    """Returns ;-separated string representing the input."""
    return ";".join([str(x) for x in a.reshape(-1)])


def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension."""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


def embed_images(
    paths: list[Path],
    model_name: str,
    device: torch.device = torch.device("cpu"),
) -> T:
    """Computes image embeddings.

    Returns a tensor of shape [len(filenames), output_dim]
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    embeddings = []
    for i, path in tqdm(enumerate(paths), desc="Global descriptors"):
        image = load_torch_image(path)
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)  # last_hidden_state and pooled

            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2)

        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list."""
    return list(itertools.combinations(range(len(lst)), 2))


def get_image_pairs(
    paths: list[Path],
    model_name: str,
    similarity_threshold: float = 0.6,
    tolerance: int = 1000,
    min_matches: int = 20,
    exhaustive_if_less: int = 20,
    p: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> list[tuple[int, int]]:
    """Obtains pairs of similar images."""
    if len(paths) <= exhaustive_if_less:
        return get_pairs_exhaustive(paths)

    matches = []

    # Embed images and compute distances for filtering
    embeddings = embed_images(paths, model_name, device=device)
    distances = torch.cdist(embeddings, embeddings, p=p)

    # Remove pairs above similarity threshold (if enough)
    mask = distances <= similarity_threshold
    image_indices = np.arange(len(paths))

    for current_image_index in range(len(paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row]

        # We don't have enough matches below the threshold, we pick most similar ones
        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index])[:min_matches]

        for other_image_index in indices_to_match:
            # Skip an image matching itself
            if other_image_index == current_image_index:
                continue

            # We need to check if we are below a certain distance tolerance
            # since for images that don't have enough matches, we picked
            # the most similar ones (which could all still be very different
            # to the image we are analyzing)
            if distances[current_image_index, other_image_index] < tolerance:
                # Add the pair in a sorted manner to avoid redundancy
                matches.append(tuple(sorted((current_image_index, other_image_index.item()))))

    return sorted(list(set(matches)))


def detect_keypoints(
    paths: list[Path],
    feature_dir: Path,
    num_features: int = 4096,
    resize_to: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Detects the keypoints in a list of images with ALIKED.

    Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5 to
    be used later with LightGlue
    """
    dtype = torch.float32  # ALIKED has issues with float16

    extractor = (
        ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, resize=resize_to)
        .eval()
        .to(device, dtype)
    )

    feature_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, h5py.File(
        feature_dir / "descriptors.h5", mode="w"
    ) as f_descriptors:
        for path in tqdm(paths, desc="Computing keypoints"):
            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=device).to(dtype)
                features = extractor.extract(image)

                f_keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
                f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()


def keypoint_distances(
    paths: list[Path],
    index_pairs: list[tuple[int, int]],
    feature_dir: Path,
    min_matches: int = 15,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Computes distances between keypoints of images.

    Stores output at feature_dir/matches.h5
    """

    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True if "cuda" in str(device) else False,
    }
    matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)

    with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, h5py.File(
        feature_dir / "descriptors.h5", mode="r"
    ) as f_descriptors, h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:
        for idx1, idx2 in tqdm(index_pairs, desc="Computing keypoing distances"):
            key1, key2 = paths[idx1].name, paths[idx2].name

            keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
            keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
            descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
            descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)

            with torch.inference_mode():
                distances, indices = matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            # We have matches to consider
            n_matches = len(indices)
            if n_matches:
                if verbose:
                    print(f"{key1}-{key2}: {n_matches} matches")
                # Store the matches in the group of one image
                if n_matches >= min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices.detach().cpu().numpy().reshape(-1, 2))


def import_into_colmap(
    path: Path,
    feature_dir: Path,
    database_path: str = "colmap.db",
) -> None:
    """Adds keypoints into colmap."""
    """
                    images_dir,
                    feature_dir,
                    str(database_path),
    
    """
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-pinhole", single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()


def parse_sample_submission(
    base_path: Path,
) -> dict[dict[str, list[Path]]]:
    """Construct a dict describing the test data as.

    {"dataset": {"scene": [<image paths>]}}
    """
    data_dict = {}
    with open(base_path / "sample_submission.csv", "r") as f:
        for i, l in enumerate(f):
            # Skip header
            if i == 0:
                print("header:", l)

            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(Path(base_path / image_path))

    print("=== summary: parse_sample_submission ===")
    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
    print("")

    return data_dict


def create_submission(
    results: dict,
    data_dict: dict[dict[str, list[Path]]],
    base_path: Path,
) -> None:
    """Prepares a submission file."""

    with open("submission.csv", "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")

        for dataset in data_dict:
            # Only write results for datasets with images that have results
            if dataset in results:
                res = results[dataset]
            else:
                res = {}

            # Same for scenes
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}

                # Write the row with rotation and translation matrices
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print(image)
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    image_path = str(image.relative_to(base_path))
                    f.write(f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")


class Config:
    base_path: Path = Path("/kaggle/input/image-matching-challenge-2024")
    feature_dir: Path = Path.cwd() / ".feature_outputs"

    device: torch.device = K.utils.get_cuda_device_if_available(0)

    pair_matching_args = {
        "model_name": "/kaggle/input/dinov2/pytorch/base/1",
        "similarity_threshold": 0.3,
        "tolerance": 1000,
        "min_matches": 20,
        "exhaustive_if_less": 20,
        "p": 2.0,
    }

    keypoint_detection_args = {
        "num_features": 4096,
        "resize_to": 1024,
    }

    keypoint_distances_args = {
        "min_matches": 15,
        "verbose": False,
    }

    colmap_mapper_options = {
        "min_model_size": 3,
        # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 2,
    }


def run_from_config(config: Config) -> None:
    results = {}

    data_dict = parse_sample_submission(config.base_path)
    datasets = list(data_dict.keys())

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:
            images_dir = data_dict[dataset][scene][0].parent
            results[dataset][scene] = {}
            image_paths = data_dict[dataset][scene]
            print(f"Got {len(image_paths)} images")

            try:
                feature_dir = config.feature_dir / f"{dataset}_{scene}"
                feature_dir.mkdir(parents=True, exist_ok=True)
                database_path = feature_dir / "colmap.db"
                if database_path.exists():
                    database_path.unlink()

                # 1. Get the pairs of images that are somewhat similar
                index_pairs = get_image_pairs(
                    image_paths,
                    **config.pair_matching_args,
                    device=config.device,
                )
                gc.collect()

                # 2. Detect keypoints of all images
                detect_keypoints(
                    image_paths,
                    feature_dir,
                    **config.keypoint_detection_args,
                    device=config.device,
                )
                gc.collect()

                # 3. Match  keypoints of pairs of similar images
                keypoint_distances(
                    image_paths,
                    index_pairs,
                    feature_dir,
                    **config.keypoint_distances_args,
                    device=config.device,
                )
                gc.collect()

                sleep(1)

                # 4.1. Import keypoint distances of matches into colmap for RANSAC
                import_into_colmap(
                    images_dir,
                    feature_dir,
                    str(database_path),
                )

                output_path = feature_dir / "colmap_rec_aliked"
                output_path.mkdir(parents=True, exist_ok=True)

                # 4.2. Compute RANSAC (detect match outliers)
                # By doing it exhaustively we guarantee we will find the best possible configuration
                pycolmap.match_exhaustive(database_path)

                mapper_options = pycolmap.IncrementalPipelineOptions(**config.colmap_mapper_options)

                # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
                # The process starts from a random pair of images and is incrementally extended by
                # registering new images and triangulating new points.
                maps = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=output_path,
                    options=mapper_options,
                )

                print(maps)
                # clear_output(wait=False)

                # 5.2. Look for the best reconstruction: The incremental mapping offered by
                # pycolmap attempts to reconstruct multiple models, we must pick the best one
                images_registered = 0
                best_idx = None

                print("Looking for the best reconstruction")

                if isinstance(maps, dict):
                    for idx1, rec in maps.items():
                        print(idx1, rec.summary())
                        try:
                            if len(rec.images) > images_registered:
                                images_registered = len(rec.images)
                                best_idx = idx1
                        except Exception:
                            continue

                # Parse the reconstruction object to get the rotation matrix and translation vector
                # obtained for each image in the reconstruction
                if best_idx is not None:
                    for k, im in maps[best_idx].images.items():
                        key = config.base_path / "test" / scene / "images" / im.name
                        results[dataset][scene][key] = {}
                        results[dataset][scene][key]["R"] = deepcopy(
                            im.cam_from_world.rotation.matrix()
                        )
                        results[dataset][scene][key]["t"] = deepcopy(
                            np.array(im.cam_from_world.translation)
                        )

                print(f"Registered: {dataset} / {scene} -> {len(results[dataset][scene])} images")
                print(f"Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
                create_submission(results, data_dict, config.base_path)
                gc.collect()

            except Exception as e:
                print(e)


def main():
    print("import ok !")
    cfg = Config()  # TODO local debug mode or submit
    run_from_config(cfg)
    pass


if __name__ == "__main__":
    main()
