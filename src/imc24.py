import gc
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def rotate_image(image, rotation):
    for i in range(4):
        with torch.no_grad():
            pred = rotation(image[None, ...]).argmax()
        if pred == 0:
            break
        image = image.rot90(dims=[1, 2])
    return image


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


def parse_sample_submission(sub_csv_path: str, data_root_dir: Path):
    data_dict = {}
    with open(sub_csv_path, "r") as f:
        for i, l in enumerate(f):
            if i == 0:
                print("header:", l)

            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(data_root_dir / image_path)

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    return data_dict


def create_submission(results, data_dict, base_path):
    with open("submission.csv", "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")

        for dataset in data_dict:
            if dataset in results:
                res = results[dataset]
            else:
                res = {}

            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}

                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    image_path = str(image.relative_to(base_path))
                    f.write(f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")


def run_from_config(
    get_pairs_func,
    keypoints_matches_func,
    ransac_and_sparse_reconstruction_func,
    config=None,
    verbose=False,
):
    results = {}

    data_dict = parse_sample_submission(
        sub_csv_path=config.sub_csv_path, data_root_dir=config.data_root_dir
    )
    datasets = list(data_dict.keys())

    df_c = pd.read_csv(config.categories_csv_path)
    scene_to_category = dict(zip(df_c["scene"], df_c["categories"]))

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:
            print(f"* run start {dataset}/{scene}")
            category = scene_to_category[scene]
            print(f"* {category=}")

            results[dataset][scene] = {}
            image_paths = data_dict[dataset][scene]
            images_dir = data_dict[dataset][scene][0].parent
            feature_dir = config.feature_dir / f"{dataset}_{scene}"

            feature_dir.mkdir(parents=True, exist_ok=True)

            start = time.time()
            index_pairs = get_pairs_func(image_paths, config)
            gc.collect()
            end = time.time() - start
            if verbose:
                print(f"  * get_pairs_func: {end}s, pairs: {len(index_pairs)}")

            start = time.time()
            keypoints_matches_func(image_paths, index_pairs, feature_dir, config, category)
            gc.collect()
            end = time.time() - start
            if verbose:
                print(f"  * keypoints_matches_func: {end}s")

            time.sleep(1)
            start = time.time()
            maps = ransac_and_sparse_reconstruction_func(images_dir, feature_dir, config)
            gc.collect()
            end = time.time() - start
            if verbose:
                print(f"  * ransac_and_sparse_reconstruction_func: {end}s")

            path = "test" if config.submit else "train"
            images_registered = 0
            best_idx = 0
            for idx, rec in maps.items():
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx

            for k, im in maps[best_idx].images.items():
                key = config.data_root_dir / path / scene / "images" / im.name
                results[dataset][scene][key] = {}
                results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                results[dataset][scene][key]["t"] = deepcopy(
                    np.array(im.cam_from_world.translation)
                )

            create_submission(results, data_dict, config.data_root_dir)
