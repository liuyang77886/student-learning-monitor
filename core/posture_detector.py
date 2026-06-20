"""
姿态检测模块 - 用于检测学生的坐姿问题（低头、弯腰等）

主要功能：
1. 实时检测学生头部、肩膀、脊椎的相对位置
2. 识别低头、弯腰等不良姿态
3. 计算姿态评分和改正建议
"""

import numpy as np
import mediapipe as mp
import cv2
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math


@dataclass
class PostureResult:
    """姿态检测结果数据类"""
    is_bad_posture: bool              # 是否为不良姿态
    posture_score: float              # 姿态评分 (0-1)
    posture_type: str                 # 姿态类型: normal, head_down, bending, tilted
    head_position: str                # 头部位置: normal, too_low, too_high
    spine_angle: float                # 脊椎弯曲角度（度数）
    confidence: float                 # 检测置信度
    suggestion: str                   # 改正建议
    landmarks: Optional[np.ndarray]   # 关键点坐标


class PostureDetector:
    """姿态检测器 - 基于 MediaPipe Pose"""
    
    def __init__(self, 
                 bad_posture_threshold: float = 0.7,
                 head_down_threshold: float = 50,
                 spine_bend_threshold: float = 30,
                 frames_to_trigger: int = 15):
        """
        初始化姿态检测器
        
        Args:
            bad_posture_threshold: 不良姿态阈值 (0-1)
            head_down_threshold: 头部下移阈值（像素）
            spine_bend_threshold: 脊椎弯曲角度阈值（度数）
            frames_to_trigger: 触发提醒所需的连续帧数
        """
        self.bad_posture_threshold = bad_posture_threshold
        self.head_down_threshold = head_down_threshold
        self.spine_bend_threshold = spine_bend_threshold
        self.frames_to_trigger = frames_to_trigger
        self.bad_posture_frame_count = 0
        
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 关键点索引
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        
    def detect(self, frame: np.ndarray) -> PostureResult:
        """
        检测单帧中的姿态
        
        Args:
            frame: 输入视频帧 (RGB 格式)
            
        Returns:
            PostureResult: 姿态检测结果
        """
        # 运行 MediaPipe Pose 检测
        results = self.pose.process(frame)
        
        if not results.pose_landmarks:
            return PostureResult(
                is_bad_posture=False,
                posture_score=0.0,
                posture_type="unknown",
                head_position="unknown",
                spine_angle=0.0,
                confidence=0.0,
                suggestion="无法检测到人物",
                landmarks=None
            )
        
        # 提取关键点
        landmarks = results.pose_landmarks.landmark
        h, w, c = frame.shape
        
        # 转换为像素坐标
        keypoints = self._landmarks_to_keypoints(landmarks, h, w)
        
        # 分析姿态
        posture_analysis = self._analyze_posture(keypoints, landmarks)
        
        # 更新不良姿态帧计数
        if posture_analysis['is_bad_posture']:
            self.bad_posture_frame_count += 1
        else:
            self.bad_posture_frame_count = 0
        
        # 判断是否触发提醒
        should_alert = self.bad_posture_frame_count >= self.frames_to_trigger
        
        return PostureResult(
            is_bad_posture=should_alert,
            posture_score=posture_analysis['posture_score'],
            posture_type=posture_analysis['posture_type'],
            head_position=posture_analysis['head_position'],
            spine_angle=posture_analysis['spine_angle'],
            confidence=posture_analysis['confidence'],
            suggestion=posture_analysis['suggestion'],
            landmarks=keypoints
        )
    
    def _landmarks_to_keypoints(self, landmarks, h: int, w: int) -> Dict:
        """将 MediaPipe 关键点转换为像素坐标"""
        keypoints = {}
        for i, landmark in enumerate(landmarks):
            keypoints[i] = {
                'x': int(landmark.x * w),
                'y': int(landmark.y * h),
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        return keypoints
    
    def _analyze_posture(self, keypoints: Dict, landmarks) -> Dict:
        """
        分析姿态
        
        Returns:
            包含姿态分析结果的字典
        """
        # 1. 检测头部位置（相对于肩膀）
        head_position, head_score = self._analyze_head_position(keypoints)
        
        # 2. 检测脊椎弯曲度
        spine_angle, spine_score = self._analyze_spine_angle(keypoints)
        
        # 3. 检测头部倾斜角度
        head_tilt, tilt_score = self._analyze_head_tilt(keypoints)
        
        # 4. 综合评分
        avg_confidence = np.mean([
            landmarks[self.NOSE].visibility,
            landmarks[self.LEFT_SHOULDER].visibility,
            landmarks[self.RIGHT_SHOULDER].visibility
        ])
        
        # 计算综合姿态评分
        posture_score = 1.0 - (head_score + spine_score + tilt_score) / 3
        
        # 判断姿态类型和是否为不良姿态
        is_bad_posture, posture_type, suggestion = self._classify_posture(
            head_position, spine_angle, head_tilt, posture_score
        )
        
        return {
            'is_bad_posture': is_bad_posture,
            'posture_score': posture_score,
            'posture_type': posture_type,
            'head_position': head_position,
            'spine_angle': spine_angle,
            'head_tilt': head_tilt,
            'confidence': avg_confidence,
            'suggestion': suggestion
        }
    
    def _analyze_head_position(self, keypoints: Dict) -> Tuple[str, float]:
        """
        分析头部位置（低头检测）
        
        Returns:
            (头部位置类型, 评分0-1)
        """
        # 获取鼻子和肩膀的 Y 坐标
        nose_y = keypoints[self.NOSE]['y']
        shoulder_y = (keypoints[self.LEFT_SHOULDER]['y'] + 
                     keypoints[self.RIGHT_SHOULDER]['y']) / 2
        
        # 头部与肩膀的垂直距离
        vertical_distance = shoulder_y - nose_y
        
        # 判断头部位置
        if vertical_distance > self.head_down_threshold:
            head_position = "too_low"  # 低头
            score = min(1.0, vertical_distance / (self.head_down_threshold * 2))
        elif vertical_distance < -50:
            head_position = "too_high"  # 抬头过高
            score = 0.3
        else:
            head_position = "normal"  # 正常
            score = 0.0
        
        return head_position, score
    
    def _analyze_spine_angle(self, keypoints: Dict) -> Tuple[float, float]:
        """
        分析脊椎弯曲度
        
        Returns:
            (脊椎角度, 评分0-1)
        """
        # 获取关键点坐标
        left_shoulder = np.array([keypoints[self.LEFT_SHOULDER]['x'], 
                                  keypoints[self.LEFT_SHOULDER]['y']])
        right_shoulder = np.array([keypoints[self.RIGHT_SHOULDER]['x'], 
                                   keypoints[self.RIGHT_SHOULDER]['y']])
        left_hip = np.array([keypoints[self.LEFT_HIP]['x'], 
                            keypoints[self.LEFT_HIP]['y']])
        right_hip = np.array([keypoints[self.RIGHT_HIP]['x'], 
                             keypoints[self.RIGHT_HIP]['y']])
        
        # 计算肩膀中点和臀部中点
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # 计算脊椎向量
        spine_vector = hip_center - shoulder_center
        
        # 计算与垂直线的夹角
        vertical_vector = np.array([0, 1])
        cos_angle = np.dot(spine_vector, vertical_vector) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        spine_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # 计算评分
        if spine_angle > self.spine_bend_threshold:
            score = min(1.0, spine_angle / (self.spine_bend_threshold * 2))
        else:
            score = 0.0
        
        return spine_angle, score
    
    def _analyze_head_tilt(self, keypoints: Dict) -> Tuple[float, float]:
        """
        分析头部倾斜角度
        
        Returns:
            (倾斜角度, 评分0-1)
        """
        # 获取两只耳朵的坐标
        left_ear = np.array([keypoints[self.LEFT_EAR]['x'], 
                            keypoints[self.LEFT_EAR]['y']])
        right_ear = np.array([keypoints[self.RIGHT_EAR]['x'], 
                             keypoints[self.RIGHT_EAR]['y']])
        
        # 计算耳朵连线与水平线的夹角
        ear_vector = right_ear - left_ear
        horizontal_vector = np.array([1, 0])
        
        cos_angle = np.dot(ear_vector, horizontal_vector) / (
            np.linalg.norm(ear_vector) * np.linalg.norm(horizontal_vector) + 1e-6
        )
        tilt_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # 计算评分
        if tilt_angle > 20:  # 倾斜超过 20 度
            score = min(1.0, tilt_angle / 40)
        else:
            score = 0.0
        
        return tilt_angle, score
    
    def _classify_posture(self, 
                          head_position: str,
                          spine_angle: float,
                          head_tilt: float,
                          posture_score: float) -> Tuple[bool, str, str]:
        """
        分类姿态类型并提供建议
        
        Returns:
            (是否为不良姿态, 姿态类型, 改正建议)
        """
        if posture_score < self.bad_posture_threshold:
            is_bad_posture = True
            
            if head_position == "too_low":
                posture_type = "head_down"
                suggestion = "请抬起头，保持正常坐姿"
            elif spine_angle > self.spine_bend_threshold:
                posture_type = "bending"
                suggestion = "请挺直背部，不要弯腰"
            elif head_tilt > 20:
                posture_type = "tilted"
                suggestion = "请保持头部竖直，不要倾斜"
            else:
                posture_type = "bad_posture"
                suggestion = "请调整坐姿，保持正确的写字姿态"
        else:
            is_bad_posture = False
            posture_type = "normal"
            suggestion = "坐姿很好，继续保持！"
        
        return is_bad_posture, posture_type, suggestion
    
    def reset(self):
        """重置检测状态"""
        self.bad_posture_frame_count = 0
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()
