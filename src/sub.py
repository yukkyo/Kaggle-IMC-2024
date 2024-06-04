import sys
import time
from itertools import combinations
from pathlib import Path

import click
import h5py
import kornia as K
import numpy as np
import pandas as pd
import pycolmap
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lightglue import ALIKED, LightGlue, match_pair
from lightglue.utils import load_image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from imc24 import run_from_config
from metrics import score

# Data importing into colmap, Provided by organizers
# sys.path.append("/kaggle/input/colmap-db-import")  # on kaggle notebook
sys.path.append("/workspace/colmap-db-import")
from database import COLMAPDatabase
from h5_to_db import add_keypoints, add_matches


def get_pairs(images_list, config):
    device = config.device
    args = config.image_pair_matching_args

    n_images = len(images_list)
    if n_images < args["exhaustive_if_less"]:
        return list(combinations(range(n_images), 2))

    processor = AutoImageProcessor.from_pretrained(args["model_name"])
    model = AutoModel.from_pretrained(args["model_name"]).eval().to(device)
    embeddings = []

    for img_path in tqdm(images_list, desc="Global descriptors"):
        image = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
        with torch.inference_mode():
            inputs = processor(
                images=image,
                return_tensors="pt",
                do_rescale=False,
                # TODO check if need
                do_resize=True,
                do_center_crop=True,
                size=224,
            ).to(device)
            outputs = model(**inputs)
            embedding = F.normalize(outputs.last_hidden_state.max(dim=1)[0])
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)
    distances = torch.cdist(embeddings, embeddings, p=args["p"]).cpu()
    distances_ = (distances <= args["similarity_threshold"]).numpy()
    np.fill_diagonal(distances_, False)
    z = distances_.sum(axis=1)
    idxs0 = np.where(z == 0)[0]
    for idx0 in idxs0:
        t = np.argsort(distances[idx0])[1 : args["min_pairs"]]
        distances_[idx0, t] = True

    s = np.where(distances >= args["tolerance"])
    distances_[s] = False

    idxs = []
    for i in range(n_images):
        for j in range(n_images):
            if distances_[i][j]:
                idxs += [(i, j)] if i < j else [(j, i)]

    idxs = list(set(idxs))
    return idxs


def overlap_detection(extractor, matcher, image0, image1, min_matches):
    """overlap している matche の精度を上げる.

    :param extractor:
    :param matcher:
    :param image0:
    :param image1:
    :param min_matches:
    :return:
    """
    feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
    if len(matches01["matches"]) < min_matches:
        return feats0, feats1, matches01
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    left0, top0 = m_kpts0.numpy().min(axis=0).astype(int)
    width0, height0 = m_kpts0.numpy().max(axis=0).astype(int)
    height0 -= top0
    width0 -= left0

    # TODO change min max without outlier
    left1, top1 = m_kpts1.numpy().min(axis=0).astype(int)
    width1, height1 = m_kpts1.numpy().max(axis=0).astype(int)
    height1 -= top1
    width1 -= left1
    crop_box0 = (top0, left0, height0, width0)
    crop_box1 = (top1, left1, height1, width1)
    cropped_img_tensor0 = TF.crop(image0, *crop_box0)
    cropped_img_tensor1 = TF.crop(image1, *crop_box1)
    feats0_c, feats1_c, matches01_c = match_pair(
        extractor, matcher, cropped_img_tensor0, cropped_img_tensor1
    )
    feats0_c["keypoints"][..., 0] += left0
    feats0_c["keypoints"][..., 1] += top0
    feats1_c["keypoints"][..., 0] += left1
    feats1_c["keypoints"][..., 1] += top1
    return feats0_c, feats1_c, matches01_c


def keypoints_matches(images_list, pairs, feature_dir, config, category):
    device = config.device
    args = config.keypoint_matching_args

    extractor = (
        ALIKED(
            max_num_keypoints=args["max_num_keypoints"],
            detection_threshold=args["detection_threshold"],
            resize=args["resize_to"],
        )
        .eval()
        .to(device)
    )
    matcher = (
        LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)
    )
    # rotation = create_model("swsl_resnext50_32x4d").eval().to(DEVICE)

    fd = feature_dir
    with h5py.File(fd / "keypoints.h5", mode="w") as f_keypoints, h5py.File(
        fd / "descriptors.h5", mode="w"
    ) as f_descriptors, h5py.File(fd / "matches.h5", mode="w") as f_matches:
        exist = np.zeros(len(images_list), dtype=bool)
        for pair in pairs:
            key1, key2 = images_list[pair[0]].name, images_list[pair[1]].name
            image1 = load_image(images_list[pair[0]]).to(device)
            image2 = load_image(images_list[pair[1]]).to(device)
            #             image1 = rotate_image(image1,rotation)
            #             image2 = rotate_image(image2,rotation)
            feats1, feats2, matches12 = match_pair(extractor, matcher, image1, image2)
            # feats1, feats2, matches12 = overlap_detection(
            #     extractor, matcher, image1, image2, args["min_matches_overlap"]
            # )
            if not exist[pair[0]]:
                f_keypoints[key1] = feats1["keypoints"].numpy()
                f_descriptors[key1] = feats1["descriptors"].numpy()
            if not exist[pair[1]]:
                f_keypoints[key2] = feats2["keypoints"].numpy()
                f_descriptors[key2] = feats2["descriptors"].numpy()
            exist[pair[0]], exist[pair[1]] = True, True
            if len(matches12["matches"]) >= args["min_matches"]:
                group = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches12["matches"].numpy())


def keypoints_matches2(images_list, pairs, feature_dir, config, category):
    """
    feats1={
        'keypoints': tensor([
            [2017.7068, 1037.6851],
            [2427.2854, 1042.1211],
            [2440.7209, 1042.1312],
            ...,
            [2584.6033, 2406.8167],
            [2597.8928, 2406.8850],
            [2631.9216, 2405.7615]
        ]),
        'descriptors': tensor([
            [ 0.0723, -0.0083, -0.0971,  ...,  0.0976, -0.0779, -0.0510],
            ...,
            [ 0.0447,  0.1414,  0.1791,  ..., -0.0112,  0.0141, -0.0764]
        ]),
        'keypoint_scores': tensor([0.9831, 0.9867, 0.9882,  ..., 0.9318, 0.9902, 0.1400]),
        'image_size': tensor([4608., 3288.])
    }
    matches12={
        'matches0': tensor([   0,   -1,   -1,  ...,   -1,   -1, 1452]),
        'matches1': tensor([   0,   11,   -1,  ..., 1514, 1519,   -1]),
        'matching_scores0': tensor([0.9799, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.3451]),
        'matching_scores1': tensor([0.9799, 0.3638, 0.0000,  ..., 0.3605, 0.3451, 0.0000]),
        'stop': 9,
        'matches': tensor([[   0,    0],
            [   3,   10],
            [   4,   11],
            ...,
            [1513, 1450],
            [1514, 1451],
            [1519, 1452]]),
        'scores': tensor([0.9799, 0.9487, 0.9654,  ..., 0.5584, 0.3605, 0.3451]),
        'prune0': tensor([9., 9., 9.,  ..., 9., 9., 9.]),
        'prune1': tensor([9., 9., 9.,  ..., 9., 9., 9.])
    }
    """
    device = config.device
    args = config.keypoint_matching_args

    torch.set_float32_matmul_precision("high")
    extractor = (
        ALIKED(
            max_num_keypoints=args["max_num_keypoints"],
            detection_threshold=args["detection_threshold"],
            resize=args["resize_to"],
        )
        .eval()
        .to(device)
    )
    matcher = (
        LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)
    )
    # matcher.compile(mode="reduce-overhead")  # not work

    p_keypoints = feature_dir / "keypoints.h5"
    p_matches = feature_dir / "matches.h5"
    with h5py.File(p_keypoints, mode="w") as f_keypoints, h5py.File(
        p_matches, mode="w"
    ) as f_matches:
        exist = np.zeros(len(images_list), dtype=bool)
        for pair in tqdm(pairs, desc="Keypoint matching for each pair"):
            key1, key2 = images_list[pair[0]].name, images_list[pair[1]].name

            image1 = load_image(images_list[pair[0]]).to(device)
            image2 = load_image(images_list[pair[1]]).to(device)

            # keypoints: n_matches x 2, float, original image hw(not 0~1)
            # matches: n_matches x 2, int, index of keypoints
            feats1, feats2, matches12 = match_pair(extractor, matcher, image1, image2)

            # feats1, feats2, matches12 = overlap_detection(
            #     extractor, matcher, image1, image2, args["min_matches_overlap"]
            # )

            # 他のモデルでどうしているのか確認する
            # keypoints 対していっても良いのでは？
            # その場合 matches の index もずらす必要がある
            if not exist[pair[0]]:
                f_keypoints[key1] = feats1["keypoints"].numpy()
            if not exist[pair[1]]:
                f_keypoints[key2] = feats2["keypoints"].numpy()
            exist[pair[0]], exist[pair[1]] = True, True
            if len(matches12["matches"]) >= args["min_matches"]:
                group = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches12["matches"].numpy())


def ransac_and_sparse_reconstruction(images_dir: Path, feature_dir: Path, config):
    # COLMAP database path
    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    # import into colmap(feature_dir needs keypoints.h5, descriptors.h5, matches.h5)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, images_dir, "", "simple-pinhole", single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()

    # RANSAC
    # we can check by help(pycolmap.SiftExtractionOptions)
    # num_threads: Number of threads for feature matching and geometric verification. (int, default: -1)
    pycolmap.match_exhaustive(database_path, sift_options={"num_threads": 1})

    # 3D mapping
    mapper_option = pycolmap.IncrementalPipelineOptions(config.pycolmap_incremental_mapping_args)
    output_path = feature_dir / "colmap_rec_output"
    output_path.mkdir(parents=True, exist_ok=True)
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=output_path,
        options=mapper_option,
    )
    return maps


class Config:
    data_root_dir: Path = Path("/kaggle/input/image-matching-challenge-2024")
    feature_dir: Path = Path.cwd() / ".feature_outputs"
    device: torch.device = K.utils.get_cuda_device_if_available(0)
    sub_csv_path = "/kaggle/input/image-matching-challenge-2024/sample_submission.csv"
    categories_csv_path = "/kaggle/input/image-matching-challenge-2024/test/categories.csv"

    n_samples_for_local_cv = 100  # for cross validation
    print(f"{feature_dir=}")
    print(f"{device=}")
    print(f"{n_samples_for_local_cv=}")

    image_pair_matching_args = dict(
        model_name="/kaggle/input/dinov2/pytorch/base/1",  # TODO change for local
        similarity_threshold=0.3,
        tolerance=500,
        min_pairs=50,
        exhaustive_if_less=50,
        p=2.0,
    )

    keypoint_matching_args = dict(
        max_num_keypoints=4096,
        detection_threshold=0.01,
        resize_to=1280,
        min_matches=100,
        min_matches_overlap=5000,
    )

    pycolmap_incremental_mapping_args = dict(
        min_model_size=5,
        # By default, COLMAP does not generate a reconstruction if less than 10 images are registered.
        # Lower it to 3.
        max_num_models=3,
        num_threads=1,
    )

    @staticmethod
    def image_path(row):
        row["image_path"] = "train/" + row["dataset"] + "/images/" + row["image_name"]
        return row

    def random_sample(self, df):
        groups = df.groupby(["dataset", "scene"])["image_path"]
        image_paths = []

        for g in groups:
            n = min(len(g[1]), self.n_samples_for_local_cv)
            g = g[0], g[1].sample(n, random_state=42).reset_index(drop=True)
            for image_path in g[1]:
                image_paths.append(image_path)

        ret_df = df[df.image_path.isin(image_paths)].reset_index(drop=True)
        return ret_df

    def __init__(self, submit: bool = True, local: bool = False, debug: bool = False):
        self.submit = submit

        if submit:
            return
        if debug:
            self.n_samples_for_local_cv = 15
            print(f"{self.n_samples_for_local_cv=}")

        self.sub_csv_path = "sub_for_localCV.csv"

        if local:
            self.image_pair_matching_args[
                "model_name"
            ] = "/root/.cache/kagglehub/models/metaresearch/dinov2/PyTorch/base/1"
            self.data_root_dir = Path("./input/image-matching-challenge-2024")
            self.sub_csv_path = str(self.data_root_dir / "sub_for_localCV.csv")

        # for local CV
        # sample train df for reduce time of calc metrics
        self.categories_csv_path = str(self.data_root_dir / "train/categories.csv")

        train_df = pd.read_csv(self.data_root_dir / "train/train_labels.csv")
        train_df = train_df.apply(self.image_path, axis=1).drop_duplicates(subset=["image_path"])
        self.gt_df = self.random_sample(train_df)
        pred_df = self.gt_df[
            ["image_path", "dataset", "scene", "rotation_matrix", "translation_vector"]
        ]
        pred_df.to_csv(self.sub_csv_path, index=False)
        print(f"saved {self.sub_csv_path}")


@click.command()
@click.option("--submit", is_flag=True)
@click.option("--local", is_flag=True)
@click.option("--debug", is_flag=True)
def main(submit, local, debug):
    config = Config(submit=submit, local=local, debug=debug)
    if debug:
        config.n_samples_for_local_cv = 15
    start = time.time()
    run_from_config(
        get_pairs_func=get_pairs,
        keypoints_matches_func=keypoints_matches2,
        ransac_and_sparse_reconstruction_func=ransac_and_sparse_reconstruction,
        config=config,
        verbose=True,
    )
    end = time.time() - start
    print("---------------------------")
    print(f"run_from_config: {end}s")
    print("---------------------------")

    # cross validation
    if not submit:
        pred_df = pd.read_csv("submission.csv")
        start = time.time()
        maa = round(score(config.gt_df, pred_df), 4)
        end = time.time() - start
        print("---------------------------")
        print("*** Total mean Average Accuracy ***")
        print(f"mAA: {maa}")
        print(f"calc score time: {end}s")
        print("---------------------------")

    print("complete.")


if __name__ == "__main__":
    main()
