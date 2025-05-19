# 以下部分均为可更改部分
# Toturial: https://theorangeduck.com/page/code-vs-data-driven-displacement
# repo: https://github.com/orangeduck/Motion-Matching

from answer_task1 import BVHMotion, translation_motions
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import grep_motion, get_features, get_features_hat
from smooth_utils import decay_spring_implicit_damping_pos, decay_spring_implicit_damping_rot

class CharacterController():
    def __init__(self, controller, path) -> None:
        self.motions = []
        self.features = []
        self.entry2idx = []
        self.idx2entry = []
        self.max_length = 120
        total_length = 0
        motion_paths = grep_motion(path)
        joint_name = None
        for idx, motion_path in enumerate(motion_paths):
            # Motion-Matching/resources/generate_database.py#98L add sim bone
            motion = BVHMotion(motion_path) 
            if "idle" in motion_path:
                idle_idx = idx
            if joint_name is None:
                joint_name = motion.joint_name
            motion.adjust_joint_name(joint_name)
            # get features
            feature = get_features(motion, self.max_length) 
            feature_length = feature.shape[0]
            start = total_length
            end = total_length + feature_length
            self.motions.append(motion)
            self.features.append(feature) 
            self.entry2idx.append((start, end))
            self.idx2entry.append(np.ones(feature_length, dtype=np.int32) * idx)
            total_length += feature_length

        self.features = np.concatenate(self.features, axis=0)
        self.idx2entry = np.concatenate(self.idx2entry, axis=0)

        self.mean, self.std = self.cal_state(self.features)
        self.features = self.normalize_features(self.features)

        self.controller = controller
        # idle motion
        self.cur_motion = self.motions[idle_idx].translation_and_rotation(0, [0, 0], [0, 1])
        self.cur_entry = idle_idx
        self.cur_frame = 0
        self.bias = 0
        self.cur_root_pos = self.cur_motion.joint_position[self.cur_frame, 0]
        self.cur_root_rot = R.from_quat(self.cur_motion.joint_rotation[self.cur_frame, 0]).apply(np.array([0., 0., 1.]))
        self.N = 60
        self.joint_name = self.cur_motion.joint_name
        self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()
        pass

    def cal_state(self, features):
        """
            get mean and std of features
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return mean, std
    
    def normalize_features(self, features):
        """
            normalize features
        """
        return (features - self.mean) / self.std
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        if self.cur_frame % self.N == 0 or self.cur_frame + 1 >= self.cur_motion.motion_length:
            # print(f"Current frame: {self.cur_frame}")
            # print("Start motion matching")
            x_hat = get_features_hat(self.cur_motion, self.cur_frame, desired_pos_list, desired_rot_list)
            x_hat = self.normalize_features(x_hat)
            k = np.argmin(np.linalg.norm(self.features - x_hat, axis=1))
            entry = self.idx2entry[k]
            start, end = self.entry2idx[entry]
            # print(f"Matched entry: {entry}, idx: {k - start}")
            if self.cur_entry != entry or self.cur_frame + self.bias != k - start:
                # print(f"Switch from {self.cur_entry} to {entry}")
                self.cur_motion = translation_motions(self.cur_motion, self.motions[entry], self.cur_frame, k - start, self.max_length)
                self.cur_entry = entry
                self.cur_frame = 0
                self.bias = k - start
                self.joint_name = self.cur_motion.joint_name
                self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()

                # self.cur_motion = self.motions[entry].translation_and_rotation(k - start, self.cur_root_pos[::2], self.cur_root_rot[::2])
                # self.cur_entry = entry
                # self.cur_frame = k - start
                # self.joint_name = self.cur_motion.joint_name
                # self.joint_translation, self.joint_orientation = self.cur_motion.batch_forward_kinematics()

        joint_translation = self.joint_translation[self.cur_frame]
        joint_orientation = self.joint_orientation[self.cur_frame]
        
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        # self.cur_root_rot = R.from_quat(joint_orientation[0]).apply(np.array([0., 0., 1.]))
        self.cur_frame += 1

        velocity = self.joint_translation[self.cur_frame + 1][0] - self.joint_translation[self.cur_frame][0]
        velocity[1] = 0

        angular_velocity = (R.from_quat(self.joint_orientation[self.cur_frame + 1][0]) * R.from_quat(self.joint_orientation[self.cur_frame][0]).inv()).as_rotvec()

        return self.joint_name, joint_translation, joint_orientation, velocity, angular_velocity
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        _, joint_translation, joint_orientation, velocity, angular_velocity = character_state
        move_speed = controller.move_speed[0]
        
        # 一个简单的例子，将手柄的位置与角色对齐
        vel_half_life = 0.6
        dt = 1./60
        pos = controller.position
        delta_pos = (self.cur_root_pos - pos)
        delta_pos[1] = 0
        offset_pos = decay_spring_implicit_damping_pos(-delta_pos, 0, vel_half_life, dt)[0]
        # print(f"Offset pos: {offset_pos}, cur_root_pos: {self.cur_root_pos}, pos: {pos}")

        rot_half_life = 0.2
        rot = controller.rotation
        delta_rot = (R.from_quat(self.cur_root_rot) * R.from_quat(rot).inv()).as_rotvec()
        offset_rot = decay_spring_implicit_damping_rot(-delta_rot, 0, rot_half_life, dt)[0]

        max_length = 0.5 * np.linalg.norm(velocity)
        offset_pos_norm = np.linalg.norm(offset_pos)
        # if offset_pos_norm > max_length:
        #     offset_pos = offset_pos / offset_pos_norm * max_length

        joint_translation[0] += offset_pos
        joint_orientation[0] = (R.from_rotvec(offset_rot) * R.from_quat(joint_orientation[0])).as_quat()

        # controller.set_pos(self.cur_root_pos + offset_pos)
        # controller.set_rot((R.from_rotvec(offset_rot) * R.from_quat(self.cur_root_rot)).as_quat())

        # controller.set_pos(self.cur_root_pos)
        # controller.set_rot(self.cur_root_rot)
        
        return self.joint_name, joint_translation, joint_orientation
    # 你的其他代码,state matchine, motion matching, learning, etc.