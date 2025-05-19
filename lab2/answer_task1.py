import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import scipy.signal as signal
from utils import get_rotation_matrix_y, slerp_pose, get_rotation_matrix_y_batch
from smooth_utils import build_loop_motion, decay_spring_implicit_damping_pos, decay_spring_implicit_damping_rot, quat_to_avel

# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
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

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:
        
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的 ndarray, 局部平移
        self.joint_rotation = None # (N,M,4) 的 ndarray, 用四元数表示的局部旋转

        # self.sim_position = None # (N, 3) ndarray, 模拟骨骼的全局位置
        # self.sim_rotation = None # (N, 4) ndarray, 模拟骨骼的全局旋转
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)
        # ['RootJoint', 'lHip', 'lKnee', 'lAnkle', 'lToeJoint', 'lToeJoint_end', 'rHip', 'rKnee', 'rAnkle', 'rToeJoint', 'rToeJoint_end', 'pelvis_lowerback', 'lowerback_torso', 'torso_head', 'torso_head_end', 'lTorso_Clavicle', 'lShoulder', 'lElbow', 'lWrist', 'lWrist_end', 'rTorso_Clavicle', 'rShoulder', 'rElbow', 'rWrist', 'rWrist_end']
        
        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]

        # return
        length = self.joint_position.shape[0]
        pos_window_length = 31 if length > 31 else length
        rot_window_length = 61 if length > 61 else length

        global_positions, global_rotations = self.batch_forward_kinematics(self.joint_position, self.joint_rotation)

        # Specify joints to use for simulation bone 
        sim_position_joint = self.joint_name.index("lowerback_torso")
        sim_rotation_joint = self.joint_name.index("RootJoint")

        # Position comes from spine joint
        sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint]
        sim_position = signal.savgol_filter(sim_position, pos_window_length, 3, axis=0, mode='interp')

        # Direction comes from projected hip forward direction
        sim_direction = np.array([1.0, 0.0, 1.0]) * R.from_quat(global_rotations[:,sim_rotation_joint]).apply(np.array([0.0, 0.0, 1.0]))
        # 这个 global rotation 乘 (0, 1, 0) 对吗? 似乎是对的, (0, 1, 0) 可以作为一个合理的 $l_0$
        # 逆天, 这 (0, 1, 0) 是不对的.  

        # We need to re-normalize the direction after both projection and smoothing
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
        sim_direction = signal.savgol_filter(sim_direction, rot_window_length, 3, axis=0, mode='interp')
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])

        # Extract rotation from direction
        sim_rotation = R.from_matrix(get_rotation_matrix_y_batch(sim_direction)).as_quat()
        # (0, 0, 1) 表示默认面朝方向

        # Transform first joints to be local to sim and append sim as root bone
        self.joint_position[:,0] = R.from_quat(sim_rotation).inv().apply(self.joint_position[:,0] - sim_position)
        self.joint_rotation[:,0] = (R.from_quat(sim_rotation).inv() * R.from_quat(self.joint_rotation[:,0])).as_quat()

        self.joint_position = np.concatenate([sim_position[:, None], self.joint_position], axis=1)
        self.joint_rotation = np.concatenate([sim_rotation[:, None], self.joint_rotation], axis=1)

        self.joint_channel = np.concatenate([[0], self.joint_channel])
        self.joint_name = ["sim_bone"] + self.joint_name
        self.joint_parent = np.concatenate([[-1], np.array(self.joint_parent) + 1])

        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None, index = None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        if index is not None:
            joint_position = joint_position[index]
            joint_rotation = joint_rotation[index]
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:]) 
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation
    
    
    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]
    
    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass
    
    #--------------------- 你的任务 -------------------- #
    
    def decompose_rotation_with_yaxis(self, rotation):
        '''
        输入:
            rotation - 四元数，形状为(4,)的ndarray, 表示完整的旋转
        输出:
            Ry - 绕Y轴旋转的四元数
            Rxz - 剩余旋转的四元数，旋转轴局限在XZ平面
        '''
        r_matrix = R.from_quat(rotation).as_matrix()

        forward = r_matrix[:, 2]
        x, z = forward[0], forward[2]

        # 构造绕Y轴的旋转矩阵
        y_matrix = get_rotation_matrix_y(x, z)

        xz_matrix = np.dot(y_matrix.T, r_matrix)

        # q_y = R.from_matrix(y_matrix).as_quat()
        # q_xz = R.from_matrix(xz_matrix).as_quat()

        return y_matrix, xz_matrix
    
    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''
        
        res = self.raw_copy() # 拷贝一份，不要修改原始数据
        
        target_y_matrix = get_rotation_matrix_y(target_facing_direction_xz[0], target_facing_direction_xz[1])
        y_matrix, xz_matrix = self.decompose_rotation_with_yaxis(res.joint_rotation[frame_num, 0])
        offset_matrix = target_y_matrix @ y_matrix.T

        root_rotation = R.from_quat(res.joint_rotation[:, 0]).as_matrix()
        root_rotation = offset_matrix @ root_rotation

        res.joint_position[:, 0] = (offset_matrix @ res.joint_position[:, 0].reshape(-1, 3, 1)).reshape(-1, 3)

        res.joint_position[:, 0, [0,2]] -= res.joint_position[frame_num, 0, [0,2]]        
        res.joint_rotation[:, 0] = R.from_matrix(root_rotation).as_quat()
        res.joint_position[:, 0, [0,2]] += target_translation_xz

        return res

# part2
def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0

    n1, n2 = bvh_motion1.motion_length, bvh_motion2.motion_length
    n3 = alpha.shape[0] 
    times = np.array([0, 1])

    for i in range(n3):
        j = i * n1 // n3
        k = i * n2 // n3

        res.joint_position[i] = (1-alpha[i]) * bvh_motion1.joint_position[j] + alpha[i] * bvh_motion2.joint_position[k]
        res.joint_rotation[i] = slerp_pose(alpha[i], bvh_motion1.joint_rotation[j], bvh_motion2.joint_rotation[k])
    
    return res

# part3
def part3_build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    
    return build_loop_motion(res)

# part4
def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    fps = 60
    half_life = 0.4
    motion1 = bvh_motion1.raw_copy()
    motion2 = bvh_motion2.raw_copy()
    res = bvh_motion1.raw_copy()

    motion_length = mix_frame1 + motion2.motion_length - 1
    
    # motion1 = motion1.translation_and_rotation(mix_frame1-1, np.array([0,0]), np.array([0,1]))
    now_pos = motion1.joint_position[mix_frame1-1, 0, [0,2]]
    now_dir = R.from_quat(motion1.joint_rotation[mix_frame1-1, 0]).apply(np.array([0,0,1])).flatten()[[0,2]]
    motion2 = motion2.translation_and_rotation(0, now_pos, now_dir)

    rotations = np.concatenate([motion1.joint_rotation[:mix_frame1], motion2.joint_rotation[1:]], axis=0)
    positions = np.concatenate([motion1.joint_position[:mix_frame1], motion2.joint_position[1:]], axis=0)

    avel = quat_to_avel(rotations, 1/60)

    # 计算最后一帧和第一帧的旋转差
    rot_diff = (R.from_quat(rotations[mix_frame1]) * R.from_quat(rotations[mix_frame1-1].copy()).inv()).as_rotvec()
    avel_diff = (avel[mix_frame1] - avel[mix_frame1-2])

    coeff = 0.5

    # 将旋转差均匀分布到每一帧
    for i in range(mix_frame1):
        offset = decay_spring_implicit_damping_rot(
            coeff*rot_diff, coeff*avel_diff, half_life, 1.*(mix_frame1-i-1)/fps
        )
        offset_rot = R.from_rotvec(offset[0])
        rotations[i] = (offset_rot * R.from_quat(rotations[i])).as_quat() 

    for i in range(mix_frame1, motion_length):
        offset = decay_spring_implicit_damping_rot(
            (coeff - 1.)*rot_diff, (coeff - 1.)*avel_diff, half_life, 1.*(i-mix_frame1)/fps
        )
        offset_rot = R.from_rotvec(offset[0])
        rotations[i] = (offset_rot * R.from_quat(rotations[i])).as_quat()

    pos_diff = positions[mix_frame1] - positions[mix_frame1-1]
    pos_diff[:,[0,2]] = 0
    vel1 = positions[mix_frame1-1] - positions[mix_frame1-2]
    vel2 = positions[mix_frame1+1] - positions[mix_frame1]
    vel_diff = (vel1 - vel2)/60

    for i in range(mix_frame1):
        offset = decay_spring_implicit_damping_pos(
            0.5*pos_diff, 0.5*vel_diff, half_life, 1.*(mix_frame1-i-1)/fps
        )
        offset_pos = offset[0]
        positions[i] += offset_pos

    for i in range(mix_frame1, motion_length):
        offset = decay_spring_implicit_damping_pos(
            -0.5*pos_diff, -0.5*vel_diff, half_life, 1.*(i-mix_frame1)/fps
        )
        offset_pos = offset[0]
        positions[i] += offset_pos

    positions = np.concatenate([positions[:mix_frame1], positions[mix_frame1+2:]], axis=0)
    rotations = np.concatenate([rotations[:mix_frame1], rotations[mix_frame1+2:]], axis=0)

    res.joint_position = positions
    res.joint_rotation = rotations
    
    return res



def translation_motions(motion1, motion2, end1, start2, max_length = 60):
    """
    将 motion1 的第 end1 帧和 motion2 的第 start2 帧连接起来
    """
    fps = 60
    half_life = 0.2
    content = 5
    content = content if end1 > content else end1

    res = BVHMotion()
    res.joint_name = motion2.joint_name
    res.joint_parent = motion2.joint_parent
    res.joint_channel = motion2.joint_channel
    res.joint_position = motion2.joint_position[start2:start2+max_length]
    res.joint_rotation = motion2.joint_rotation[start2:start2+max_length]

    now_pos = motion1.joint_position[end1, 0, [0,2]]
    now_dir = R.from_quat(motion1.joint_rotation[end1, 0]).apply(np.array([0,0,1])).flatten()[[0,2]]
    motion2 = res.translation_and_rotation(0, now_pos, now_dir)

    rotations = np.concatenate([motion1.joint_rotation[end1-content:end1+1], motion2.joint_rotation], axis=0)
    positions = np.concatenate([motion1.joint_position[end1-content:end1+1], motion2.joint_position], axis=0)
    motion_length = positions.shape[0]

    avel = quat_to_avel(rotations, 1/60)

    # 计算最后一帧和第一帧的旋转差
    rot_diff = (R.from_quat(rotations[content + 1]) * R.from_quat(rotations[content].copy()).inv()).as_rotvec()

    avel_diff = avel[content + 1] - avel[content - 1] if content != 0 \
        else avel[content + 1]

    coeff = 0

    # 将旋转差均匀分布到每一帧
    for i in range(content + 1, motion_length):
        offset = decay_spring_implicit_damping_rot(
            -rot_diff, avel_diff, half_life, 1.*(i-content)/fps
        )
        offset_rot = R.from_rotvec(offset[0])
        rotations[i] = (offset_rot * R.from_quat(rotations[i])).as_quat()

    pos_diff = positions[content + 1] - positions[content]
    pos_diff[:,[0,2]] = 0

    vel_length = 3
    vel1_length = vel_length if content > vel_length else content
    vel1 = (positions[content] - positions[content - vel1_length])/vel1_length if vel1_length != 0 \
        else positions[content]
    vel2_length = vel_length if content + 1 + vel_length < motion_length else motion_length - content - 1
    vel2 = (positions[content + 1 + vel2_length] - positions[content + 1])/vel2_length
    vel_diff = (vel1 - vel2)/60

    for i in range(content + 1, motion_length):
        offset = decay_spring_implicit_damping_pos(
            -pos_diff, vel_diff, half_life, 1.*(i-content)/fps
        )
        offset_pos = offset[0]
        positions[i] += offset_pos

    positions = positions[content+1:]
    rotations = rotations[content+1:]

    res.joint_position = positions
    res.joint_rotation = rotations
    
    return res
