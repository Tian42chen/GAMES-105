import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def get_rotation_matrix_y(x, z):
    # 计算 cos(θ) 和 sin(θ)
    len_v2 = np.sqrt(x**2 + z**2)
    cos_theta = z / len_v2
    sin_theta = x / len_v2

    # 构造绕Y轴的旋转矩阵
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    return rotation_matrix

def get_rotation_matrix_y_batch(direction): 
    """ 

    Args: 
        direction: [N, 3] forwardx, forwardy, forwardz

    Returns:
        rotation_matrix: [N, 3, 3] 绕Y轴的, 将 (0, 0, 1) 旋转到 (x, 0, z)

    """
    # 计算 cos(θ) 和 sin(θ)
    len_v2 = (
        np.linalg.norm(direction[:, [0, 2]], ord=2, axis=-1, keepdims=True) + 1e-9
    )
    # print(len_v2)
    norm_traj = direction[:, [0, 2]] / len_v2
    # print(norm_traj)
    sin_theta = norm_traj[:, 0] # x / len
    cos_theta =  norm_traj[:, 1] # z / len
    # print(cos_theta)
    # print(sin_theta)

    # 构造绕Y轴的旋转矩阵
    rotation_matrix = np.repeat(
            np.array([
            [1., 0., 1.],
            [0., 1., 0.],
            [-1., 0., 1.]
        ])[None], direction.shape[0], axis=0
    )

    rotation_matrix[:, 0, 0] = cos_theta
    rotation_matrix[:, 0, 2] = sin_theta
    rotation_matrix[:, 2, 0] = -sin_theta
    rotation_matrix[:, 2, 2] = cos_theta

    return rotation_matrix


def slerp_pose(alpha, pose1, pose2):
    # 由于 Slerp 目前无法直接对多组四元数批量操作, 因此需要对每个四元数对进行逐个插值
    assert pose1.shape == pose2.shape

    time = np.array([0, 1])

    res_pose = np.zeros_like(pose1)

    for i in range(pose1.shape[0]):
        r1 = R.from_quat(pose1[i])
        r2 = R.from_quat(pose2[i])
        slerp = Slerp(time, R.concatenate([r1, r2]))
        res_pose[i] = slerp([alpha]).as_quat()

    return res_pose


def grep_motion(path):
    """
    从指定路径中获取所有的 BVH 文件
    Args:
        path: str 指定路径

    Returns:
        motion_paths: List[str] 所有 BVH 文件的路径
    """
    motion_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.bvh'):
                motion_paths.append(os.path.join(root, file))
    return motion_paths

def feature_norm_with_weight(feature, weight):
    """
    对特征进行归一化
    Args:
        feature: np.ndarray, shape=(N, C)
        weight: np.ndarray

    Returns:
        feature_norm: np.ndarray, shape=(N, C)
    """
    feature_norm = feature * weight
    return feature_norm


weight_foot_pos = 0.75
weight_foot_vel = 1.
weight_hip_vel = 1.2
weight_traj_pos = 1.
weight_traj_dir = 1.5 

def get_features(motion, motion_end):
    """

    Args:
        motion: BVHMotion

    Returns:
        features: np.ndarray, shape=(N, 27), N 为帧数, 27 为特征数
    """
    nfeatures = (
        3 +  # Left Foot Position
        3 +  # Right Foot Position
        3 +  # Left Foot Velocity
        3 +  # Right Foot Velocity
        3 +  # Hip Velocity
        6 +  # Trajectory Positions 2D
        6    # Trajectory Directions 2D
    )

    global_positions, global_rotations = motion.batch_forward_kinematics() 

    leftfoot_idx = motion.joint_name.index('lToeJoint_end')
    rightfoot_idx = motion.joint_name.index('rToeJoint_end')
    hip_idx = motion.joint_name.index('RootJoint')
    root_idx = motion.joint_name.index('sim_bone')
    global_rotations = R.from_quat(global_rotations[:, root_idx]).as_matrix()

    root_pos = global_positions[:, root_idx]
    root_pos[:, 1] = 0
    root_dir = global_rotations @ np.array([0, 0, 1])
    root_rot_y = get_rotation_matrix_y_batch(root_dir)
    root_rot_y_inv = root_rot_y.transpose(0, 2, 1) # (N, 3, 3)

    # 把 pos 和 rot 移到正面原点
    leftfoot_pos = global_positions[:, leftfoot_idx]
    rightfoot_pos = global_positions[:, rightfoot_idx]
    hip_pos = global_positions[:, hip_idx]
    leftfoot_vel = np.diff(leftfoot_pos, axis=0, append=leftfoot_pos[[-1]])
    rightfoot_vel = np.diff(rightfoot_pos, axis=0, append=rightfoot_pos[[-1]])
    hip_vel = np.diff(hip_pos, axis=0, append=hip_pos[[-1]])

    # 先转速度
    leftfoot_vel = (root_rot_y_inv @ leftfoot_vel[:, :, None])[:, :, 0]
    rightfoot_vel = (root_rot_y_inv @ rightfoot_vel[:, :, None])[:, :, 0]
    hip_vel = (root_rot_y_inv @ hip_vel[:, :, None])[:, :, 0]

    # 再转位置
    leftfoot_pos -= root_pos
    leftfoot_pos = (root_rot_y_inv @ leftfoot_pos[:, :, None])[:, :, 0]
    rightfoot_pos -= root_pos
    rightfoot_pos = (root_rot_y_inv @ rightfoot_pos[:, :, None])[:, :, 0]

    # 生成轨迹 index
    traj_futures = [20, 40 ,60]
    traj_idxs = []
    for future in traj_futures:
        traj_idx = np.arange(0, motion.motion_length) + future
        traj_idx[traj_idx >= motion.motion_length] = motion.motion_length - 1
        traj_idxs.append(traj_idx)
    traj_idxs = np.array(traj_idxs).T

    # 生成轨迹 pos 和 dir
    traj_pos = (global_positions[traj_idxs, 0] - root_pos[:, None]) # (N, 3, 3), (N, 3) is index, (..., 3) is pos
    traj_pos = root_rot_y_inv[:, None] @ traj_pos[:, :, :, None] # (N, 3, 3, 1)
    traj_pos = traj_pos[:, :, ::2, 0]
    traj_pos = traj_pos.reshape(-1, len(traj_futures) * 2)

    traj_rot = global_rotations[traj_idxs] # (N, 3, 3, 3), (N, 3) is index, (..., 3, 3) is rot
    traj_rot = root_rot_y_inv[:, None] @ traj_rot # (N, 3, 3, 3)
    traj_dir = traj_rot @ np.array([0, 0, 1]) # (N, 3, 3)
    traj_dir = traj_dir[:, :, ::2].reshape(-1, len(traj_futures) * 2)

    features = np.concatenate([
        feature_norm_with_weight(leftfoot_pos, weight_foot_pos),
        feature_norm_with_weight(rightfoot_pos, weight_foot_pos),
        feature_norm_with_weight(leftfoot_vel, weight_foot_vel),
        feature_norm_with_weight(rightfoot_vel, weight_foot_vel),
        feature_norm_with_weight(hip_vel, weight_hip_vel),
        feature_norm_with_weight(traj_pos, weight_traj_pos),
        feature_norm_with_weight(traj_dir, weight_traj_dir)
    ], axis=-1)

    return features[:-motion_end]

def get_features_hat(motion, frame, desired_pos_list, desired_rot_list):
    """

    Args:
        motion: BVHMotion
        frame: int
        desired_pos_list: np.ndarray, shape=(6, 3) 暂时假设其是全局坐标
        desired_rot_list: np.ndarray, shape=(6, 4)

    Returns:
        features_hat: np.ndarray, shape=(27)

    """
    if frame +1 >= motion.motion_length:
        frame = motion.motion_length - 2

    global_positions, global_rotations = motion.batch_forward_kinematics(index = [frame, frame+1])

    leftfoot_idx = motion.joint_name.index('lToeJoint_end')
    rightfoot_idx = motion.joint_name.index('rToeJoint_end')
    hip_idx = motion.joint_name.index('RootJoint')
    root_idx = motion.joint_name.index('sim_bone')
    global_rotations = R.from_quat(global_rotations[:, root_idx]).as_matrix()

    root_pos = global_positions[:1, root_idx]
    root_pos[:, 1] = 0
    root_dir = global_rotations[:1] @ np.array([0, 0, 1])
    root_rot_y = get_rotation_matrix_y_batch(root_dir)
    root_rot_y_inv = root_rot_y.transpose(0, 2, 1) # (1, 3, 3)

    # 把 pos 和 rot 移到正面原点
    leftfoot_pos = global_positions[:1, leftfoot_idx]
    rightfoot_pos = global_positions[:1, rightfoot_idx]
    hip_pos = global_positions[:1, hip_idx]
    leftfoot_vel = leftfoot_pos[-1:] - leftfoot_pos[:1]
    rightfoot_vel = rightfoot_pos[-1:] - rightfoot_pos[:1]
    hip_vel = hip_pos[-1:] - hip_pos[:1]

    # 先转速度
    leftfoot_vel = (root_rot_y_inv @ leftfoot_vel[:, :, None])[:, :, 0]
    rightfoot_vel = (root_rot_y_inv @ rightfoot_vel[:, :, None])[:, :, 0]
    hip_vel = (root_rot_y_inv @ hip_vel[:, :, None])[:, :, 0]

    # 再转位置
    leftfoot_pos -= root_pos
    leftfoot_pos = (root_rot_y_inv @ leftfoot_pos[:, :, None])[:, :, 0]
    rightfoot_pos -= root_pos
    rightfoot_pos = (root_rot_y_inv @ rightfoot_pos[:, :, None])[:, :, 0]

    target_length = 3 + 1 # 20, 40, 60
    traj_pos = desired_pos_list[1:target_length] - root_pos # (3, 3)
    traj_pos = root_rot_y_inv @ traj_pos[:, :, None] # (3, 3, 1)
    traj_pos = traj_pos[:, ::2, 0]
    traj_pos = traj_pos.reshape(1, -1)

    traj_rot = R.from_quat(desired_rot_list[1:target_length]).as_matrix() # (3, 3, 3)
    traj_rot = root_rot_y_inv @ traj_rot # (3, 3, 3)
    traj_dir = traj_rot @ np.array([0, 0, 1]) # (3, 3)
    traj_dir = traj_dir[:, ::2].reshape(1, -1)

    features_hat = np.concatenate([
        feature_norm_with_weight(leftfoot_pos, weight_foot_pos),
        feature_norm_with_weight(rightfoot_pos, weight_foot_pos),
        feature_norm_with_weight(leftfoot_vel, weight_foot_vel),
        feature_norm_with_weight(rightfoot_vel, weight_foot_vel),
        feature_norm_with_weight(hip_vel, weight_hip_vel),
        feature_norm_with_weight(traj_pos, weight_traj_pos),
        feature_norm_with_weight(traj_dir, weight_traj_dir)
    ], axis=-1)

    return features_hat
