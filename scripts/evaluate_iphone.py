import argparse
import json
import os.path as osp
from glob import glob
from itertools import product

import cv2
import imageio.v3 as iio
import numpy as np
import roma
import torch
from tqdm import tqdm

from flow3d.data.colmap import get_colmap_camera_params
from flow3d.metrics import mLPIPS, mPSNR, mSSIM
from flow3d.transforms import rt_to_mat4, solve_procrustes

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the data directory that contains all the sequences.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    help="Path to the result directory that contains the results."
    "for batch evaluation, result_dir should contain subdirectories for each sequence. (result_dir/seq_name/results)"
    "for single sequence evaluation, result_dir should contain results directly (result_dir/results)",
)
parser.add_argument(
    "--seq_names",
    type=str,
    nargs="+",
    default=[
        "apple",
        "backpack", 
        "block",
        "creeper",
        "handwavy",
        "haru-sit",
        "mochi-high-five",
        "paper-windmill",
        "pillow",
        "spin",
        "sriracha-tree",
        "teddy",
    ],
    help="Sequence names to evaluate.",
)
args = parser.parse_args()


def load_data_dict(data_dir, train_names, val_names):
    """
    加载评估所需的数据字典
    
    Args:
        data_dir: 数据目录路径
        train_names: 训练帧名称列表
        val_names: 验证帧名称列表
    
    Returns:
        包含验证图像、共视区域、深度、相机参数、关键点等数据的字典
    """
    # 加载验证集图像
    val_imgs = np.array(
        [iio.imread(osp.join(data_dir, "rgb/1x", f"{name}.png")) for name in val_names]
    )
    
    # 加载验证集共视区域掩码
    val_covisibles = np.array(
        [
            iio.imread(
                osp.join(
                    data_dir, "flow3d_preprocessed/covisible/1x/val/", f"{name}.png"
                )
            )
            for name in tqdm(val_names, desc="Loading val covisibles")
        ]
    )
    
    # 加载训练集深度图
    train_depths = np.array(
        [
            np.load(osp.join(data_dir, "depth/1x", f"{name}.npy"))[..., 0]
            for name in train_names
        ]
    )
    
    # 获取COLMAP相机参数（内参矩阵K和世界到相机的变换矩阵w2c）
    train_Ks, train_w2cs = get_colmap_camera_params(
        osp.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
        [name + ".png" for name in train_names],
    )
    train_Ks = train_Ks[:, :3, :3]  # 提取3x3内参矩阵
    
    # 加载场景缩放因子并应用到相机位置
    scale = np.load(osp.join(data_dir, "flow3d_preprocessed/colmap/scale.npy")).item()
    train_c2ws = np.linalg.inv(train_w2cs)  # 计算相机到世界的变换矩阵
    train_c2ws[:, :3, -1] *= scale  # 应用缩放因子到平移部分
    train_w2cs = np.linalg.inv(train_c2ws)  # 重新计算世界到相机的变换
    
    # 加载2D关键点数据
    keypoint_paths = sorted(glob(osp.join(data_dir, "keypoint/2x/train/0_*.json")))
    keypoints_2d = []
    for keypoint_path in keypoint_paths:
        with open(keypoint_path) as f:
            keypoints_2d.append(json.load(f))
    keypoints_2d = np.array(keypoints_2d)
    keypoints_2d[..., :2] *= 2.0  # 从2x分辨率缩放到1x分辨率
    
    # 提取时间ID并生成时间对
    time_ids = np.array(
        [int(osp.basename(p).split("_")[1].split(".")[0]) for p in keypoint_paths]
    )
    time_pairs = np.array(list(product(time_ids, repeat=2)))  # 所有时间帧的配对
    index_pairs = np.array(list(product(range(len(time_ids)), repeat=2)))  # 索引配对
    
    # 将2D关键点投影到3D空间
    keypoints_3d = []
    for i, kps_2d in zip(time_ids, keypoints_2d):
        K = train_Ks[i]  # 当前帧的内参矩阵
        w2c = train_w2cs[i]  # 当前帧的外参矩阵
        depth = train_depths[i]  # 当前帧的深度图
        
        # 检查关键点可见性和深度有效性
        is_kp_visible = kps_2d[:, 2] == 1  # 关键点可见性标志
        is_depth_valid = (
            cv2.remap(
                (depth != 0).astype(np.float32),
                kps_2d[None, :, :2].astype(np.float32),
                None,  # type: ignore
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[0]
            == 1
        )
        
        # 获取关键点处的深度值
        kp_depths = cv2.remap(
            depth,  # type: ignore
            kps_2d[None, :, :2].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        
        # 将2D点反投影到相机坐标系
        kps_3d = (
            np.einsum(
                "ij,pj->pi",
                np.linalg.inv(K),
                np.pad(kps_2d[:, :2], ((0, 0), (0, 1)), constant_values=1),
            )
            * kp_depths[:, None]
        )
        
        # 转换到世界坐标系
        kps_3d = np.einsum(
            "ij,pj->pi",
            np.linalg.inv(w2c)[:3],
            np.pad(kps_3d, ((0, 0), (0, 1)), constant_values=1),
        )
        
        # 添加有效性标志并处理无效点
        kps_3d = np.concatenate(
            [kps_3d, (is_kp_visible & is_depth_valid)[:, None]], axis=1
        )
        kps_3d[kps_3d[:, -1] != 1] = 0.0  # 将无效点设为0
        keypoints_3d.append(kps_3d)
    keypoints_3d = np.array(keypoints_3d)
    
    return {
        "val_imgs": val_imgs,
        "val_covisibles": val_covisibles,
        "train_depths": train_depths,
        "train_Ks": train_Ks,
        "train_w2cs": train_w2cs,
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "time_ids": time_ids,
        "time_pairs": time_pairs,
        "index_pairs": index_pairs,
    }


def load_result_dict(result_dir, val_names):
    """
    加载模型预测结果
    
    Args:
        result_dir: 结果目录路径
        val_names: 验证帧名称列表
    
    Returns:
        包含预测图像、深度、相机参数、关键点等结果的字典
    """
    try:
        # 加载预测的验证集图像
        pred_val_imgs = np.array(
            [
                iio.imread(osp.join(result_dir, "rgb", f"{name}.png"))
                for name in val_names
            ]
        )
    except:
        pred_val_imgs = None
    
    try:
        # 加载关键点预测结果
        keypoints_dict = np.load(
            osp.join(result_dir, "keypoints.npz"), allow_pickle=True
        )
        if len(keypoints_dict) == 1 and "arr_0" in keypoints_dict:
            keypoints_dict = keypoints_dict["arr_0"].item()
        pred_keypoint_Ks = keypoints_dict["Ks"]  # 预测的相机内参
        pred_keypoint_w2cs = keypoints_dict["w2cs"]  # 预测的相机外参
        pred_keypoints_3d = keypoints_dict["pred_keypoints_3d"]  # 预测的3D关键点
        pred_train_depths = keypoints_dict["pred_train_depths"]  # 预测的深度图
    except:
        print(
            "No keypoints.npz found, make sure that it's the method itself cannot produce keypoints."
        )
        keypoints_dict = {}
        pred_keypoint_Ks = None
        pred_keypoint_w2cs = None
        pred_keypoints_3d = None
        pred_train_depths = None

    # 加载可见性预测（如果存在）
    if "visibilities" in list(keypoints_dict.keys()):
        pred_visibilities = keypoints_dict["visibilities"]
    else:
        pred_visibilities = None

    return {
        "pred_val_imgs": pred_val_imgs,
        "pred_train_depths": pred_train_depths,
        "pred_keypoint_Ks": pred_keypoint_Ks,
        "pred_keypoint_w2cs": pred_keypoint_w2cs,
        "pred_keypoints_3d": pred_keypoints_3d,
        "pred_visibilities": pred_visibilities,
    }


def evaluate_3d_tracking(data_dict, result_dict):
    """
    评估3D跟踪性能
    
    计算3D关键点跟踪的端点误差(EPE)和精度(PCK)指标
    
    Args:
        data_dict: 真实数据字典
        result_dict: 预测结果字典
    
    Returns:
        epe: 端点误差
        pck_3d_10cm: 10cm阈值下的精度
        pck_3d_5cm: 5cm阈值下的精度
    """
    # 提取相关数据
    train_Ks = data_dict["train_Ks"]
    train_w2cs = data_dict["train_w2cs"]
    keypoints_3d = data_dict["keypoints_3d"]
    time_ids = data_dict["time_ids"]
    time_pairs = data_dict["time_pairs"]
    index_pairs = data_dict["index_pairs"]
    pred_keypoint_Ks = result_dict["pred_keypoint_Ks"]
    pred_keypoint_w2cs = result_dict["pred_keypoint_w2cs"]
    pred_keypoints_3d = result_dict["pred_keypoints_3d"]
    
    # 检查相机内参一致性
    if not np.allclose(train_Ks[time_ids], pred_keypoint_Ks):
        print("Inconsistent camera intrinsics.")
        print(train_Ks[time_ids][0], pred_keypoint_Ks[0])
    
    # 使用Procrustes分析对齐预测和真实的相机轨迹
    keypoint_w2cs = train_w2cs[time_ids]
    q, t, s = solve_procrustes(
        torch.from_numpy(np.linalg.inv(pred_keypoint_w2cs)[:, :3, -1]).to(
            torch.float32
        ),
        torch.from_numpy(np.linalg.inv(keypoint_w2cs)[:, :3, -1]).to(torch.float32),
    )[0]
    
    # 应用对齐变换到预测的3D关键点
    R = roma.unitquat_to_rotmat(q.roll(-1, dims=-1))
    pred_keypoints_3d = np.einsum(
        "ij,...j->...i",
        rt_to_mat4(R, t, s).numpy().astype(np.float64),
        np.pad(pred_keypoints_3d, ((0, 0), (0, 0), (0, 1)), constant_values=1),
    )
    pred_keypoints_3d = pred_keypoints_3d[..., :3] / pred_keypoints_3d[..., 3:]
    
    # 计算3D跟踪指标
    pair_keypoints_3d = keypoints_3d[index_pairs]
    is_covisible = (pair_keypoints_3d[:, :, :, -1] == 1).all(axis=1)  # 共视区域掩码
    target_keypoints_3d = pair_keypoints_3d[:, 1, :, :3]  # 目标关键点
    
    # 计算每帧的端点误差
    epes = []
    for i in range(len(time_pairs)):
        epes.append(
            np.linalg.norm(
                target_keypoints_3d[i][is_covisible[i]]
                - pred_keypoints_3d[i][is_covisible[i]],
                axis=-1,
            )
        )
    
    # 计算平均指标
    epe = np.mean(
        [frame_epes.mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_10cm = np.mean(
        [(frame_epes < 0.1).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_5cm = np.mean(
        [(frame_epes < 0.05).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    print(f"3D tracking EPE: {epe:.4f}")
    print(f"3D tracking PCK (10cm): {pck_3d_10cm:.4f}")
    print(f"3D tracking PCK (5cm): {pck_3d_5cm:.4f}")
    print("-----------------------------")
    return epe, pck_3d_10cm, pck_3d_5cm


def project(Ks, w2cs, pts):
    """
    将3D点投影到2D图像平面
    
    Args:
        Ks: (N, 3, 3) 相机内参矩阵
        w2cs: (N, 4, 4) 相机外参矩阵
        pts: (N, N, M, 3) 3D点坐标
    
    Returns:
        projected_pts: 投影后的2D点坐标
        depths: 对应的深度值
    """
    N = Ks.shape[0]
    pts = pts.swapaxes(0, 1).reshape(N, -1, 3)

    # 转换为齐次坐标
    pts_homogeneous = np.concatenate([pts, np.ones_like(pts[..., -1:])], axis=-1)

    # 应用世界到相机的变换
    pts_homogeneous = np.matmul(w2cs[:, :3], pts_homogeneous.swapaxes(1, 2)).swapaxes(
        1, 2
    )
    
    # 使用内参矩阵投影到图像平面
    projected_pts = np.matmul(Ks, pts_homogeneous.swapaxes(1, 2)).swapaxes(1, 2)

    depths = projected_pts[..., 2:3]
    # 归一化齐次坐标
    projected_pts = projected_pts[..., :2] / np.clip(depths, a_min=1e-6, a_max=None)
    projected_pts = projected_pts.reshape(N, N, -1, 2).swapaxes(0, 1)
    depths = depths.reshape(N, N, -1).swapaxes(0, 1)
    return projected_pts, depths


def evaluate_2d_tracking(data_dict, result_dict):
    """
    评估2D跟踪性能
    
    计算2D关键点跟踪的平均Jaccard指数(AJ)、平均PCK和遮挡准确率
    
    Args:
        data_dict: 真实数据字典
        result_dict: 预测结果字典
    
    Returns:
        AJ: 平均Jaccard指数
        APCK: 平均PCK
        occ_acc: 遮挡准确率
    """
    # 提取相关数据
    train_w2cs = data_dict["train_w2cs"]
    keypoints_2d = data_dict["keypoints_2d"]
    visibilities = keypoints_2d[..., -1].astype(np.bool_)
    time_ids = data_dict["time_ids"]
    num_frames = len(time_ids)
    num_pts = keypoints_2d.shape[1]
    pred_train_depths = result_dict["pred_train_depths"]
    pred_keypoint_Ks = result_dict["pred_keypoint_Ks"]
    pred_keypoint_w2cs = result_dict["pred_keypoint_w2cs"]
    pred_keypoints_3d = result_dict["pred_keypoints_3d"].reshape(
        num_frames, -1, num_pts, 3
    )
    
    # 对齐预测和真实轨迹的缩放因子
    keypoint_w2cs = train_w2cs[time_ids]
    s = solve_procrustes(
        torch.from_numpy(np.linalg.inv(pred_keypoint_w2cs)[:, :3, -1]).to(
            torch.float32
        ),
        torch.from_numpy(np.linalg.inv(keypoint_w2cs)[:, :3, -1]).to(torch.float32),
    )[0][-1].item()

    # 准备目标点和可见性数据
    target_points = keypoints_2d[None].repeat(num_frames, axis=0)[..., :2]
    target_visibilities = visibilities[None].repeat(num_frames, axis=0)

    # 将预测的3D点投影到2D
    pred_points, pred_depths = project(
        pred_keypoint_Ks, pred_keypoint_w2cs, pred_keypoints_3d
    )
    
    # 计算预测的可见性
    if result_dict["pred_visibilities"] is not None:
        pred_visibilities = result_dict["pred_visibilities"].reshape(
            num_frames, -1, num_pts
        )
    else:
        # 基于深度一致性估计可见性
        rendered_depths = []
        for i, points in zip(
            data_dict["index_pairs"][:, -1],
            pred_points.reshape(-1, pred_points.shape[2], 2),
        ):
            rendered_depths.append(
                cv2.remap(
                    pred_train_depths[i].astype(np.float32),
                    points[None].astype(np.float32),  # type: ignore
                    None,  # type: ignore
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )[0]
            )
        rendered_depths = np.array(rendered_depths).reshape(num_frames, -1, num_pts)
        pred_visibilities = (np.abs(rendered_depths - pred_depths) * s) < 0.05

    # 设置评估掩码（排除对角线元素，即同一帧）
    one_hot_eye = np.eye(target_points.shape[0])[..., None].repeat(num_pts, axis=-1)
    evaluation_points = one_hot_eye == 0
    for i in range(num_frames):
        evaluation_points[i, :, ~visibilities[i]] = False
    
    # 计算遮挡准确率
    occ_acc = np.sum(
        np.equal(pred_visibilities, target_visibilities) & evaluation_points
    ) / np.sum(evaluation_points)
    
    # 在不同阈值下计算PCK和Jaccard指数
    all_frac_within = []
    all_jaccard = []

    for thresh in [4, 8, 16, 32, 64]:  # 像素阈值
        within_dist = np.sum(
            np.square(pred_points - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, target_visibilities)
        count_correct = np.sum(is_correct & evaluation_points)
        count_visible_points = np.sum(target_visibilities & evaluation_points)
        frac_correct = count_correct / count_visible_points
        all_frac_within.append(frac_correct)

        # 计算Jaccard指数
        true_positives = np.sum(is_correct & pred_visibilities & evaluation_points)
        gt_positives = np.sum(target_visibilities & evaluation_points)
        false_positives = (~target_visibilities) & pred_visibilities
        false_positives = false_positives | ((~within_dist) & pred_visibilities)
        false_positives = np.sum(false_positives & evaluation_points)
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)
    
    AJ = np.mean(all_jaccard)
    APCK = np.mean(all_frac_within)

    print(f"2D tracking AJ: {AJ:.4f}")
    print(f"2D tracking avg PCK: {APCK:.4f}")
    print(f"2D tracking occlusion accuracy: {occ_acc:.4f}")
    print("-----------------------------")
    return AJ, APCK, occ_acc


def evaluate_nv(data_dict, result_dict):
    """
    评估新视角合成(Novel View synthesis)性能
    
    计算mPSNR、mSSIM和mLPIPS指标
    
    Args:
        data_dict: 真实数据字典
        result_dict: 预测结果字典
    
    Returns:
        mpsnr: 平均PSNR
        mssim: 平均SSIM  
        mlpips: 平均LPIPS
    """
    device = "cuda"
    # 初始化评估指标
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)

    # 准备数据
    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3].to(device)
    val_covisibles = torch.from_numpy(data_dict["val_covisibles"]).to(device)
    pred_val_imgs = torch.from_numpy(result_dict["pred_val_imgs"]).to(device)

    # 逐帧计算指标
    for i in range(len(val_imgs)):
        val_img = val_imgs[i] / 255.0  # 归一化到[0,1]
        pred_val_img = pred_val_imgs[i] / 255.0
        val_covisible = val_covisibles[i] / 255.0
        
        # 更新各项指标
        psnr_metric.update(val_img, pred_val_img, val_covisible)
        ssim_metric.update(val_img[None], pred_val_img[None], val_covisible[None])
        lpips_metric.update(val_img[None], pred_val_img[None], val_covisible[None])
    
    # 计算最终指标
    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mSSIM: {mssim:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    return mpsnr, mssim, mlpips


if __name__ == "__main__":
    """
    主评估流程
    
    对指定的序列进行批量评估，包括：
    1. 3D关键点跟踪性能
    2. 2D关键点跟踪性能  
    3. 新视角合成性能
    
    最后输出所有序列的平均性能指标
    """
    seq_names = args.seq_names

    # 初始化指标列表
    epe_all, pck_3d_10cm_all, pck_3d_5cm_all = [], [], []  # 3D跟踪指标
    AJ_all, APCK_all, occ_acc_all = [], [], []  # 2D跟踪指标
    mpsnr_all, mssim_all, mlpips_all = [], [], []  # 新视角合成指标

    # 逐序列评估
    for seq_name in seq_names:
        print("=========================================")
        print(f"Evaluating {seq_name}")
        print("=========================================")
        
        # 构建数据和结果目录路径
        data_dir = osp.join(args.data_dir, seq_name)
        if not osp.exists(data_dir):
            data_dir = args.data_dir
        if not osp.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} not found.")
        result_dir = osp.join(args.result_dir, seq_name, "results/")
        if not osp.exists(result_dir):
            result_dir = osp.join(args.result_dir, "results/")
        if not osp.exists(result_dir):
            raise ValueError(f"Result directory {result_dir} not found.")

        # 加载训练/验证数据分割
        with open(osp.join(data_dir, "splits/train.json")) as f:
            train_names = json.load(f)["frame_names"]
        with open(osp.join(data_dir, "splits/val.json")) as f:
            val_names = json.load(f)["frame_names"]

        # 加载数据和结果
        data_dict = load_data_dict(data_dir, train_names, val_names)
        result_dict = load_result_dict(result_dir, val_names)
        
        # 评估关键点跟踪性能（如果有预测结果）
        if result_dict["pred_keypoints_3d"] is not None:
            epe, pck_3d_10cm, pck_3d_5cm = evaluate_3d_tracking(data_dict, result_dict)
            AJ, APCK, occ_acc = evaluate_2d_tracking(data_dict, result_dict)
            epe_all.append(epe)
            pck_3d_10cm_all.append(pck_3d_10cm)
            pck_3d_5cm_all.append(pck_3d_5cm)
            AJ_all.append(AJ)
            APCK_all.append(APCK)
            occ_acc_all.append(occ_acc)
            
        # 评估新视角合成性能（如果有验证图像）
        if len(data_dict["val_imgs"]) > 0:
            if result_dict["pred_val_imgs"] is None:
                print("No NV results found.")
                continue
            mpsnr, mssim, mlpips = evaluate_nv(data_dict, result_dict)
            mpsnr_all.append(mpsnr)
            mssim_all.append(mssim)
            mlpips_all.append(mlpips)

    print(f"mean 3D tracking EPE: {np.mean(epe_all):.4f}")
    print(f"mean 3D tracking PCK (10cm): {np.mean(pck_3d_10cm_all):.4f}")
    print(f"mean 3D tracking PCK (5cm): {np.mean(pck_3d_5cm_all):.4f}")
    print(f"mean 2D tracking AJ: {np.mean(AJ_all):.4f}")
    print(f"mean 2D tracking avg PCK: {np.mean(APCK_all):.4f}")
    print(f"mean 2D tracking occlusion accuracy: {np.mean(occ_acc_all):.4f}")
    print(f"mean NV mPSNR: {np.mean(mpsnr_all):.4f}")
    print(f"mean NV mSSIM: {np.mean(mssim_all):.4f}")
    print(f"mean NV mLPIPS: {np.mean(mlpips_all):.4f}")
