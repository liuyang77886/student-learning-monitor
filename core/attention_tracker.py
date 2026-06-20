"""
注意力追踪模块 - 用于检测学生是否分心

主要功能：
1. 检测眼睛闭合时间
2. 追踪头部和眼睛移动
3. 分析面部表情（是否在思考）
4. 综合判断学生是否分心
"""

import numpy as np
import mediapipe as mp
import cv2
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from collections import deque
import time


@dataclass
class AttentionResult:
    """注意力检测结果数据类"""
    is_distracted: bool              # 是否分心
    attention_score: float           # 注意力评分 (0-1, 1 为最专注)
    eye_status: str                  # 眼睛状态: open, closed, blinking
    head_movement: float             # 头部移动幅度
    gaze_direction: str              # 目光方向: forward, left, right, down
    distraction_duration: float      # 分心持续时长（秒）
    suggestion: str                  # 建议信息
    confidence: float                # 检测置信度


class AttentionTracker:
    """注意力追踪器 - 基于 MediaPipe Face Detection"""
    
    def __init__(self,
                 distraction_timeout: float = 3.0,
                 head_movement_threshold: float = 10.0,
                 eye_closure_timeout: float = 2.0,
                 history_size: int = 30):
        """
        初始化注意力追踪器
        
        Args:
            distraction_timeout: 分心判定超时时间（秒）
            head_movement_threshold: 头部移动阈值（度数）
            eye_closure_timeout: 眼睛闭合超时（秒）
            history_size: 历史数据保存大小
        """
        self.distraction_timeout = distraction_timeout
        self.head_movement_threshold = head_movement_threshold
        self.eye_closure_timeout = eye_closure_timeout
        self.history_size = history_size
        
        # 初始化 MediaPipe Face Detection 和 Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 关键点索引
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        self.IRIS_LEFT = [468, 469, 470, 471, 472]
        self.IRIS_RIGHT = [473, 474, 475, 476, 477]
        
        # 历史数据
        self.head_position_history = deque(maxlen=history_size)
        self.eye_openness_history = deque(maxlen=history_size)
        self.blink_count = 0
        self.last_eye_open_time = time.time()
        self.distraction_start_time = None
        self.last_valid_head_position = None
    
    def detect(self, frame: np.ndarray) -> AttentionResult:
        """
        检测单帧中的注意力状态
        
        Args:
            frame: 输入视频帧 (RGB 格式)
            
        Returns:
            AttentionResult: 注意力检测结果
        """
        # 运行 Face Mesh 检测
        face_mesh_results = self.face_mesh.process(frame)
        h, w, c = frame.shape
        
        if not face_mesh_results.multi_face_landmarks:
            return AttentionResult(
                is_distracted=False,
                attention_score=0.5,
                eye_status="unknown",
                head_movement=0.0,
                gaze_direction="unknown",
                distraction_duration=0.0,
                suggestion="无法检测到人脸",
                confidence=0.0
            )
        
        landmarks = face_mesh_results.multi_face_landmarks[0].landmark
        
        # 1. 检测眼睛状态
        eye_status, eye_openness = self._detect_eye_status(landmarks)
        self.eye_openness_history.append(eye_openness)
        
        # 2. 检测头部位置和运动
        head_position = self._extract_head_position(landmarks)
        if head_position is not None:
            self.head_position_history.append(head_position)
            self.last_valid_head_position = head_position
        
        head_movement = self._calculate_head_movement()
        gaze_direction = self._detect_gaze_direction(landmarks)
        
        # 3. 判断注意力状态
        is_distracted, distraction_duration = self._determine_distraction_state(
            eye_status, eye_openness, head_movement
        )
        
        # 4. 计算注意力评分
        attention_score = self._calculate_attention_score(
            eye_openness, head_movement, gaze_direction
        )
        
        # 5. 生成建议
        suggestion = self._generate_suggestion(
            is_distracted, eye_status, gaze_direction
        )
        
        # 获取检测置信度
        confidence = max([lm.visibility for lm in landmarks[:10]])
        
        return AttentionResult(
            is_distracted=is_distracted,
            attention_score=attention_score,
            eye_status=eye_status,
            head_movement=head_movement,
            gaze_direction=gaze_direction,
            distraction_duration=distraction_duration,
            suggestion=suggestion,
            confidence=confidence
        )
    
    def _detect_eye_status(self, landmarks) -> Tuple[str, float]:
        """
        检测眼睛状态（张开/闭合/眨眼）
        
        Returns:
            (眼睛状态, 眼睛张开程度0-1)
        """
        # 计算眼睛高宽比（Eye Aspect Ratio, EAR）
        left_ear = self._calculate_eye_aspect_ratio(
            landmarks, self.LEFT_EYE_LANDMARKS
        )
        right_ear = self._calculate_eye_aspect_ratio(
            landmarks, self.RIGHT_EYE_LANDMARKS
        )
        
        ear = (left_ear + right_ear) / 2
        
        # EAR 阈值
        eye_open_threshold = 0.15
        eye_close_threshold = 0.10
        
        if ear > eye_open_threshold:
            status = "open"
            self.last_eye_open_time = time.time()
        elif ear < eye_close_threshold:
            status = "closed"
        else:
            status = "blinking"
        
        return status, ear
    
    def _calculate_eye_aspect_ratio(self, landmarks, eye_landmarks: list) -> float:
        """
        计算眼睛宽高比
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        # 获取眼睛关键点
        points = np.array([landmarks[i] for i in eye_landmarks])
        
        # 计算距离
        dist1 = np.linalg.norm(
            np.array([points[1].x, points[1].y]) - 
            np.array([points[5].x, points[5].y])
        )
        dist2 = np.linalg.norm(
            np.array([points[2].x, points[2].y]) - 
            np.array([points[4].x, points[4].y])
        )
        dist3 = np.linalg.norm(
            np.array([points[0].x, points[0].y]) - 
            np.array([points[3].x, points[3].y])
        )
        
        ear = (dist1 + dist2) / (2.0 * dist3 + 1e-6)
        return ear
    
    def _extract_head_position(self, landmarks) -> Optional[np.ndarray]:
        """提取头部位置（使用面部中心）"""
        try:
            # 使用多个面部点计算头部中心
            face_points = np.array([
                [landmarks[i].x, landmarks[i].y] 
                for i in [10, 152, 33, 263, 1, 199]  # 鼻子上下、两侧、前额
            ])
            head_center = face_points.mean(axis=0)
            return head_center
        except:
            return None
    
    def _calculate_head_movement(self) -> float:
        """计算头部移动幅度"""
        if len(self.head_position_history) < 2:
            return 0.0
        
        # 计算最后 5 帧之间的位移
        positions = list(self.head_position_history)
        if len(positions) >= 5:
            movement = np.linalg.norm(positions[-1] - positions[-5])
        else:
            movement = np.linalg.norm(positions[-1] - positions[0])
        
        return movement
    
    def _detect_gaze_direction(self, landmarks) -> str:
        """
        检测视线方向
        
        通过检测虹膜相对于眼睛的位置来判断
        """
        # 计算左眼虹膜位置
        left_iris_points = np.array([
            [landmarks[i].x, landmarks[i].y] 
            for i in self.IRIS_LEFT
        ])
        left_eye_points = np.array([
            [landmarks[i].x, landmarks[i].y] 
            for i in self.LEFT_EYE_LANDMARKS
        ])
        
        # 虹膜在眼睛中的位置
        iris_center_x = left_iris_points.mean(axis=0)[0]
        eye_center_x = left_eye_points.mean(axis=0)[0]
        eye_left_x = left_eye_points[:, 0].min()
        eye_right_x = left_eye_points[:, 0].max()
        
        # 判断方向
        eye_width = eye_right_x - eye_left_x
        iris_offset = iris_center_x - eye_center_x
        
        if iris_offset < -eye_width * 0.2:
            direction = "left"
        elif iris_offset > eye_width * 0.2:
            direction = "right"
        else:
            direction = "forward"
        
        return direction
    
    def _determine_distraction_state(self,
                                    eye_status: str,
                                    eye_openness: float,
                                    head_movement: float) -> Tuple[bool, float]:
        """
        综合判断是否分心
        
        分心条件：
        1. 眼睛闭合超过 2 秒
        2. 视线离开超过 3 秒
        3. 头部快速转动
        """
        is_distracted = False
        distraction_duration = 0.0
        
        # 检查眼睛闭合时间
        if eye_status == "closed":
            eye_closure_duration = time.time() - self.last_eye_open_time
            if eye_closure_duration > self.eye_closure_timeout:
                is_distracted = True
                distraction_duration = eye_closure_duration
        
        # 检查头部快速运动
        if head_movement > self.head_movement_threshold:
            is_distracted = True
        
        # 更新分心开始时间
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = time.time()
            distraction_duration = time.time() - self.distraction_start_time
        else:
            self.distraction_start_time = None
        
        return is_distracted, distraction_duration
    
    def _calculate_attention_score(self,
                                  eye_openness: float,
                                  head_movement: float,
                                  gaze_direction: str) -> float:
        """计算综合注意力评分 (0-1, 1 为最专注)"""
        score = 1.0
        
        # 眼睛睁开程度评分
        if eye_openness > 0.15:
            eye_score = 1.0
        elif eye_openness > 0.10:
            eye_score = 0.5
        else:
            eye_score = 0.0
        
        # 头部运动评分
        if head_movement < 1.0:
            movement_score = 1.0
        elif head_movement < 5.0:
            movement_score = 0.7
        else:
            movement_score = 0.3
        
        # 视线方向评分
        gaze_score = 1.0 if gaze_direction == "forward" else 0.6
        
        # 综合评分
        score = (eye_score * 0.5 + movement_score * 0.3 + gaze_score * 0.2)
        
        return score
    
    def _generate_suggestion(self,
                            is_distracted: bool,
                            eye_status: str,
                            gaze_direction: str) -> str:
        """生成建议信息"""
        if not is_distracted:
            return "注意力集中，继续加油！"
        
        if eye_status == "closed":
            return "眼睛闭合，请保持清醒专注！"
        
        if gaze_direction == "left":
            return "视线向左，请回到题目！"
        elif gaze_direction == "right":
            return "视线向右，请回到题目！"
        
        return "检测到分心，请回到作业上！"
    
    def reset(self):
        """重置追踪状态"""
        self.head_position_history.clear()
        self.eye_openness_history.clear()
        self.blink_count = 0
        self.last_eye_open_time = time.time()
        self.distraction_start_time = None
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
