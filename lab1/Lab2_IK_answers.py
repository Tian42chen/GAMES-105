import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def forward_kinematics(meta_data, joint_offsets, root_position, joint_orientations):
    """
    计算正运动学
    输入: 
        meta_data: 其中包含 joint_name, joint_parent, joint_initial_position(T-pose postion), root_joint, end_joint
        joint_offsets: 所有关节相对于父关节的偏移，tensor，shape为(M, 3)，M为关节数
        root_position: 根的位置，tensor，shape为(3,)
        joint_orientations: 所有关节旋转，tensor，shape为(M, 4)，M为关节数
    输出:
        joint_positions: 计算得到的关节位置，tensor，shape为(M, 3)，M为关节数
    """
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    root_idx = joint_name.index(meta_data.root_joint)
    end_idx = joint_name.index(meta_data.end_joint)

    M = len(joint_name)
    joint_positions = torch.zeros(M, 3)

    rotation_matrices = quaternion_to_matrix(joint_orientations)

    for i in range(1, M):
        parent_idx = joint_parent[i]
        joint_positions[i] = joint_positions[parent_idx] + torch.matmul(rotation_matrices[parent_idx], joint_offsets[i])

    delta = root_position - joint_positions[root_idx]
    joint_positions = joint_positions + delta

    # print(f"joint_positions: {joint_positions}")
    # print(f"delta: {delta}")
    # print(f"rotation_matrices: {rotation_matrices}")
    return joint_positions


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    计算逆运动学
    输入: 
        meta_data: 其中包含 joint_name, joint_parent, joint_initial_position(T-pose postion), root_joint, end_joint
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    learning_rate=0.1
    num_iterations=10

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    
    for i in range(len(joint_name)):
        if joint_orientations[i][-1] == 0:
            joint_orientations[i] = [0, 0, 0, 1]

    opt_joint_orientations = torch.tensor(np.array(joint_orientations), dtype=torch.float32, requires_grad=True)
    M = joint_orientations.shape[0]
    mask = torch.zeros(M, 4, dtype=torch.bool)
    mask[path, :] = True

    # rest_indices = list(set(range(M)) - set(path))

    # x_path = joint_orientations[path].clone().requires_grad_(True)
    # x_rest = joint_orientations[rest_indices].clone().requires_grad_(False)

    # opt_joint_orientations = torch.zeros(M, 4)
    # opt_joint_orientations[path] = x_path
    # opt_joint_orientations[rest_indices] = x_rest
    # print(f"joint_orientations: {opt_joint_orientations}")


    T_positions = meta_data.joint_initial_position
    # print(f"T_positions: {T_positions}")
    joint_offsets = [
        T_positions[i] - T_positions[joint_parent[i]]
        for i in range(len(joint_name))
    ]
    joint_offsets[0] = T_positions[0]
    joint_offsets = torch.tensor(np.array(joint_offsets), dtype=torch.float32)
    # print(f"joint_offsets: {joint_offsets}")

    root_idx = joint_name.index(meta_data.root_joint)
    end_idx = joint_name.index(meta_data.end_joint)
    root_position = joint_positions[root_idx]
    root_position = torch.tensor(root_position, dtype=torch.float32)
    target_pose = torch.tensor(target_pose, dtype=torch.float32)
    # print(f"root_position: {root_position}")

    optimizer = torch.optim.Adam([opt_joint_orientations], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        # 正运动学计算当前末端位置
        current_positions = forward_kinematics(meta_data, joint_offsets, root_position, opt_joint_orientations)
        # break

        # 计算误差
        error = target_pose - current_positions[end_idx]
        error = torch.norm(error)

        # 反向传播
        error.backward()

        # opt_joint_orientations.grad[~mask] = 0
        
        # if torch.isnan(opt_joint_orientations.grad).any():
        #     print(f"i: {i}")
        #     print(f"Error: {error.item()}")
        #     print(f"current_positions: {current_positions[end_idx]}")
        #     print(f"target_pose: {target_pose}")
        #     print(f"opt_joint_orientations: {opt_joint_orientations}")
        #     exit(0)

        # 更新关节旋转
        optimizer.step()

        # 打印迭代信息
        if i % 10 == 0:
            pass
            # print(f'Iteration {i}, Error: {error.item()}')
        if error.item() < 1e-2:
            break


    return_positions = forward_kinematics(meta_data, joint_offsets, root_position, opt_joint_orientations)
    return_positions = return_positions.detach().numpy()


    # joint_rotations = [R.from_quat(joint_rotations[i].detach().numpy()) for i in range(len(joint_name))]
    # for i in range(len(joint_name)):
    #     print(f"joint_rotations[{i}]: {joint_rotations[i].as_euler('XYZ', degrees=True)}")

    # return_orientations = [R.from_euler("XYZ", [0,0,0]) for i in range(len(joint_name))]
    # for i in range(1, len(joint_name)):
    #     return_orientations[i] = return_orientations[joint_parent[i]] * joint_rotations[i]
    # for i in range(len(joint_name)):
    #     print(f"return_orientations[{i}]: {return_orientations[i].as_euler('XYZ', degrees=True)}")

    # return_orientations = np.array([r.as_quat() for r in return_orientations])

    # print(f"return_orientations: {return_orientations}, shape: {return_orientations.shape}")

    return_orientations = opt_joint_orientations.detach().numpy()
    # print(return_orientations)
    # for i in range(len(joint_name)):
    #     # print(f"return_positions[{i}]: {return_positions[i]}")
    #     print(f"return_orientations[{i}]: {R.from_quat(return_orientations[i]).as_euler('XYZ', degrees=True)}")

    tmp_orientations = [R.from_euler('XYZ', [0,0,0]) for i in range(len(joint_name))]
    tmp_orientations = np.array([r.as_quat() for r in tmp_orientations])

    return return_positions, return_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    root_position = joint_positions[0]
    target = [root_position[0]+relative_x, target_height, root_position[2]+relative_z]
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target)

    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    learning_rate=0.01
    num_iterations=100

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    
    for i in range(len(joint_name)):
        if joint_orientations[i][-1] == 0:
            joint_orientations[i] = [0, 0, 0, 1]

    opt_joint_orientations = torch.tensor(np.array(joint_orientations), dtype=torch.float32, requires_grad=True)
    M = joint_orientations.shape[0]
    mask = torch.zeros(M, 4, dtype=torch.bool)
    mask[path, :] = True

    T_positions = meta_data.joint_initial_position
    # print(f"T_positions: {T_positions}")
    joint_offsets = [
        T_positions[i] - T_positions[joint_parent[i]]
        for i in range(len(joint_name))
    ]
    joint_offsets[0] = T_positions[0]
    joint_offsets = torch.tensor(np.array(joint_offsets), dtype=torch.float32)
    # print(f"joint_offsets: {joint_offsets}")

    root_idx = joint_name.index(meta_data.root_joint)
    end_idx_l = joint_name.index('lWrist_end')
    end_idx_r = joint_name.index('rWrist_end')
    root_position = joint_positions[root_idx]
    root_position = torch.tensor(root_position, dtype=torch.float32)
    left_target_pose = torch.tensor(left_target_pose, dtype=torch.float32)
    right_target_pose = torch.tensor(right_target_pose, dtype=torch.float32)
    # print(f"root_position: {root_position}")

    optimizer = torch.optim.Adam([opt_joint_orientations], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        # 正运动学计算当前末端位置
        current_positions = forward_kinematics(meta_data, joint_offsets, root_position, opt_joint_orientations)
        # break

        # 计算误差
        error = torch.norm(left_target_pose - current_positions[end_idx_l]) + torch.norm(right_target_pose - current_positions[end_idx_r])

        # 反向传播
        error.backward()

        # 更新关节旋转
        optimizer.step()

        # 打印迭代信息
        if i % 10 == 0:
            pass
            # print(f'Iteration {i}, Error: {error.item()}')
        if error.item() < 1e-2:
            break


    return_positions = forward_kinematics(meta_data, joint_offsets, root_position, opt_joint_orientations)
    return_positions = return_positions.detach().numpy()


    return_orientations = opt_joint_orientations.detach().numpy()


    tmp_orientations = [R.from_euler('XYZ', [0,0,0]) for i in range(len(joint_name))]
    tmp_orientations = np.array([r.as_quat() for r in tmp_orientations])

    return return_positions, return_orientations