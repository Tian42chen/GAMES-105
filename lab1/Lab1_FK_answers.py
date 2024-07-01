import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('HIERARCHY'):
                break
        for j in range(i, len(lines)):
            if lines[j].startswith('MOTION'):
                break

        bvh = lines[i+1:j]
    # print(bvh)
    joint_name = []
    joint_parent = []
    joint_offset = []
    stack =[]
    idx = 0
    for line in bvh:
        items = line.split()
        # print(items)
        if items[0] == 'ROOT':
            joint_name.append(items[1])
            joint_parent.append(-1)
            stack.append(idx)
        if items[0] == 'JOINT':
            joint_name.append(items[1])
            joint_parent.append(stack[-1])
            idx += 1
            stack.append(idx)
        if items[0] == 'End':
            joint_parent.append(stack[-1])
            joint_name.append(f'{joint_name[-1]}_end')
            idx += 1
            stack.append(idx)
        if items[0] == '}':
            stack.pop()
        if items[0] == 'OFFSET':
            joint_offset.append([float(items[1]), float(items[2]), float(items[3])])
    return joint_name, joint_parent, joint_offset

def motion_iterator(motion: np.ndarray):
    """ 迭代一个 np.ndarray，每次三个元素。

    Args:
        motion (np.ndarray): 一个大小为 (N) 的数组。

    Yields:
        tuple: 每次迭代返回的三个元素。
    """
    for i in range(0, len(motion), 3):
        yield motion[i:i + 3]


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion = motion_iterator(motion_data[frame_id])

    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))

    joint_positions[0] = next(motion)
    joint_orientations[0] = R.from_euler('XYZ', next(motion), degrees=True).as_quat()
    # print(len(joint_name))
    # print(len(motion))
    for i in range(1, len(joint_name)):
        if joint_name[i].endswith('_end'):
            continue
        parent_idx = joint_parent[i]
        joint_positions[i] = joint_positions[parent_idx] + R.from_quat(joint_orientations[parent_idx]).apply(joint_offset[i])
        joint_orientations[i] = (R.from_quat(joint_orientations[parent_idx]) * R.from_euler('XYZ', next(motion), degrees=True)).as_quat()
        # print(f"idx: {i}")
        # print(f"name: {joint_name[i]}")


    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    T_name2idx = {name: i for i, name in enumerate(T_joint_name)}
    T_joint_offset = np.array(T_joint_offset)

    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_name2idx = {name: i for i, name in enumerate(A_joint_name)}
    A_joint_offset = np.array(A_joint_offset)

    # rotation, _ = R.align_vectors([p2], [p1])
    # A2T_joint_Q = [
    #     R.align_vectors([T_joint_offset[T_name2idx[A_joint_name[i]]]], [A_joint_offset[i]])[0]
    #     if not A_joint_name[i].endswith('_end') or not np.allclose(A_joint_offset[i], T_joint_offset[T_name2idx[A_joint_name[i]]], atol=1e-5) else R.from_euler('XYZ', [0, 0, 0], degrees=True)
    #     for i in range(len(A_joint_name)) 
    # ]
    A2T_joint_Q = [R.from_euler('XYZ', [0, 0, 0], degrees=True) for _ in range(len(A_joint_name))]
    for i in range(0, len(A_joint_name)):
        parent_idx = A_joint_parent[i]
        A_offet = A_joint_offset[i]
        T_offet = T_joint_offset[T_name2idx[A_joint_name[i]]]
        if not np.allclose(A_offet, T_offet, atol=1e-5):
            rotation , _ = R.align_vectors([T_offet], [A_offet])
            A2T_joint_Q[parent_idx] = rotation
    A2T_joint_Q.append(R.from_euler('XYZ', [0, 0, 0], degrees=True))

    A2T_joint_Q_inv = [q.inv() for q in A2T_joint_Q]

    for i in range(len(A_joint_name)):
        print(f"name: {A_joint_name[i]}")
        print(A2T_joint_Q[i].as_euler('XYZ', degrees=True))
        print(A2T_joint_Q_inv[i].as_euler('XYZ', degrees=True))
    

    motion_data = load_motion_data(A_pose_bvh_path)
    retarget_motion_data = np.zeros_like(motion_data)
    for i in range(motion_data.shape[0]):
        motion = motion_iterator(motion_data[i])
        A_position = next(motion)
        # A_motion = []
        # # A_motion.append((R.from_euler('XYZ', next(motion), degrees=True) * A2T_joint_Q[0]).as_euler('XYZ', degrees=True))
        # for j in range(len(A_joint_name)):
        #     if A_joint_name[j].endswith('_end'):
        #         A_motion.append(np.zeros(3))
        #         continue
        #     # R^B_i = Q^{T\to A}_{p_i} * R^A_i * (Q^{T\to A}_i)^\top
        #     parent_idx = A_joint_parent[j]
        #     R_A = R.from_euler('XYZ', next(motion), degrees=True)
        #     R_B = A2T_joint_Q[parent_idx] * R_A * A2T_joint_Q_inv[j]
        #     A_motion.append(R_B.as_euler('XYZ', degrees=True))
        #     # print(f"name: {A_joint_name[j]}")
        #     # print(f"parent: {A_joint_name[parent_idx]}")
        #     # print(f"R_A: {R_A.as_euler('XYZ', degrees=True)}")
        #     # print(f"Q: {A2T_joint_Q[parent_idx].as_euler('XYZ', degrees=True)}")
        #     # print(f"Q_inv: {A2T_joint_Q_inv[j].as_euler('XYZ', degrees=True)}")
        #     # print(f"R_B: {R_B.as_euler('XYZ', degrees=True)}")
        
        A_motion = [
            (A2T_joint_Q[A_joint_parent[j]] * R.from_euler('XYZ', next(motion), degrees=True) * A2T_joint_Q_inv[j]).as_euler('XYZ', degrees=True)
            if not A_joint_name[j].endswith('_end')
            else np.zeros(3)
            for j in range(len(A_joint_name))
        ]
        
        retarget_motion = [
            A_motion[A_name2idx[T_joint_name[j]]] 
            for j in range(len(T_joint_name)) 
            if not T_joint_name[j].endswith('_end')
        ]
        retarget_motion_data[i] = np.concatenate([A_position] + retarget_motion)
        # print(f"frame: {i}")
        # print(motion_data[i])
        # print(A_joint_R)
        # print(A_position)
        # print(retarget_motion)
        # print(retarget_motion_data[i])
        # print(len(retarget_motion_data[i]))
        # print(len(retarget_motion))
        # break
        if i >5000: break
    return retarget_motion_data
