# cross_ped_async.py - 异步行人检测脚本
# 视频正常播放，VLM 推理异步进行

import cv2
import base64
import sys
import time
import json
import asyncio
import threading
from pathlib import Path
from io import BytesIO
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, Set  # [修改] 新增 Set
from collections import deque
from enum import Enum
from dataclasses import dataclass, field  # [修改] 新增 field
from queue import Queue, Empty
import concurrent.futures

# 导入 CANoe 接口
from connect import CANoeInterface

# ======================
# 状态枚举
# ======================
class DetectionMode(Enum):
    NONE = "None"
    GREEN_LIGHT_PED = "mode_green_light_ped_on_crosswalk"
    RED_LIGHT_PED = "mode_red_light_ped_on_crosswalk"
    INTERSECTION = "mode_intersection"

# ======================
# Pydantic 结构化输出
# ======================
class SceneDetection(BaseModel):
    """道路场景检测结果"""
    frame_index: int = Field(description="帧编号")
    
    # [修改] 增加置信度字段，提高识别准确性
    has_crosswalk: bool = Field(description="画面中是否出现斑马线（人行横道白色条纹）")
    crosswalk_confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description="斑马线检测置信度：high=清晰可见完整斑马线，medium=部分可见或较远，low=不确定或无"
    )
    
    has_pedestrian: bool = Field(description="斑马线上是否有行人正在过马路")
    
    traffic_light: Literal["None", "Red", "Green", "Yellow"] = Field(
        description="交通信号灯状态"
    )
    
    # [修改] 增加置信度字段，提高识别准确性
    has_intersection: bool = Field(
        default=False,
        description="画面中是否处于十字路口（必须能看到多个方向的道路交汇）"
    )
    intersection_confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description="十字路口检测置信度：high=明确看到多条道路交汇，medium=可能是路口，low=不确定或普通道路"
    )

# ======================
# 帧数据容器
# ======================
@dataclass
class FrameData:
    """帧数据包装"""
    frame: any  # numpy array
    frame_index: int
    timestamp: float  # 视频时间戳（秒）
    capture_time: float  # 捕获时的系统时间

# ======================
# [修改] 多状态平滑器 - 支持同时存在多个状态
# ======================
class MultiStateSmoother:
    """多状态平滑器 - 每个状态独立进行帧容忍（线程安全）"""
    
    def __init__(self, tolerance_frames: int = 5):
        self.tolerance_frames = tolerance_frames
        self._lock = threading.Lock()
        
        # 每个状态独立管理
        self.current_states: Set[DetectionMode] = set()  # 当前激活的状态集合
        self.pending_states: dict = {}  # {DetectionMode: pending_count}
        self.exit_counts: dict = {}  # {DetectionMode: exit_count} 用于退出计数
    
    def update(self, raw_states: Set[DetectionMode]) -> Set[DetectionMode]:
        """
        更新状态并返回平滑后的状态集合
        raw_states: 本次检测到的原始状态集合（不含 NONE）
        """
        with self._lock:
            # 处理每个可能的状态（除了 NONE）
            all_modes = {DetectionMode.GREEN_LIGHT_PED, DetectionMode.RED_LIGHT_PED, DetectionMode.INTERSECTION}
            
            for mode in all_modes:
                if mode in raw_states:
                    # 检测到该状态
                    self.exit_counts[mode] = 0  # 重置退出计数
                    
                    if mode in self.current_states:
                        # 已经是激活状态，保持
                        pass
                    else:
                        # 尝试进入该状态
                        self.pending_states[mode] = self.pending_states.get(mode, 0) + 1
                        if self.pending_states[mode] >= self.tolerance_frames:
                            self.current_states.add(mode)
                            self.pending_states[mode] = 0
                else:
                    # 未检测到该状态
                    self.pending_states[mode] = 0  # 重置进入计数
                    
                    if mode in self.current_states:
                        # 当前是激活状态，尝试退出
                        self.exit_counts[mode] = self.exit_counts.get(mode, 0) + 1
                        if self.exit_counts[mode] >= self.tolerance_frames:
                            self.current_states.discard(mode)
                            self.exit_counts[mode] = 0
            
            return self.current_states.copy()
    
    def get_current_states(self) -> Set[DetectionMode]:
        with self._lock:
            return self.current_states.copy()

# ======================
# 辅助函数
# ======================
def frame_to_base64(frame) -> str:
    """将帧转换为 base64 编码"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

# [修改] 返回状态集合而非单一状态，并增加置信度过滤
def get_raw_states(result: SceneDetection) -> Set[DetectionMode]:
    """从检测结果获取原始状态集合（可能包含多个状态）"""
    if result is None:
        return set()
    
    states = set()
    
    # 检测行人状态（需要斑马线置信度至少为 medium）
    if result.has_pedestrian and result.has_crosswalk and result.crosswalk_confidence in ("high", "medium"):
        if result.traffic_light == "Green":
            states.add(DetectionMode.GREEN_LIGHT_PED)
        elif result.traffic_light == "Red":
            states.add(DetectionMode.RED_LIGHT_PED)
    
    # 检测十字路口（只有 high 置信度才算检测到）
    if result.has_intersection and result.intersection_confidence == "high":
        states.add(DetectionMode.INTERSECTION)
    
    return states

# ======================
# 异步行人检测器
# ======================
class AsyncPedestrianDetector:
    """
    异步行人检测器
    - 视频正常倍速播放
    - VLM 推理在后台线程异步进行
    - 推理完成后处理 CANoe 逻辑，然后获取当前帧继续推理
    """
    
    def __init__(
        self,
        video_path: str,
        tolerance_frames: int = 5,
        show_video: bool = True,
        display_scale: float = 0.5,
        canoe: CANoeInterface = None,
        green_light_folder: str = None,
        red_light_folder: str = None,
        intersection_folder: str = None
    ):
        self.video_path = video_path
        self.tolerance_frames = tolerance_frames
        self.show_video = show_video
        self.display_scale = display_scale
        
        # CANoe 相关
        self.canoe = canoe
        self.green_light_folder = Path(green_light_folder) if green_light_folder else None
        self.red_light_folder = Path(red_light_folder) if red_light_folder else None
        self.intersection_folder = Path(intersection_folder) if intersection_folder else None
        
        # [修改] 使用多状态平滑器
        self.smoother = MultiStateSmoother(tolerance_frames)
        
        # vLLM 客户端
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://10.214.153.118:22002/v1",
            timeout=3600
        )
        
        # 视频捕获
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 0.033
        
        print(f"视频信息: {self.total_frames} 帧, {self.fps:.2f} FPS, 时长 {self.duration:.2f} 秒")
        
        # 状态变量
        self.current_frame_index = 0
        self.current_frame = None
        # [修改] 使用状态集合
        self.last_output_states: Set[DetectionMode] = set()
        self.state_start_frame = 0
        
        # 线程控制
        self.running = False
        self.frame_lock = threading.Lock()
        
        # 推理结果
        self.latest_result: Optional[SceneDetection] = None
        # [修改] 使用状态集合
        self.latest_smoothed_states: Set[DetectionMode] = set()
        
        # 推理线程池（单线程，确保顺序执行）
        self.inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.inference_future: Optional[concurrent.futures.Future] = None
        
        # 统计信息
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # CANoe 操作队列
        self.canoe_action_queue: Queue = Queue()
    
    def get_current_frame_copy(self) -> Optional[FrameData]:
        """获取当前帧的副本（线程安全）"""
        with self.frame_lock:
            if self.current_frame is not None:
                return FrameData(
                    frame=self.current_frame.copy(),
                    frame_index=self.current_frame_index,
                    timestamp=self.current_frame_index / self.fps if self.fps > 0 else 0,
                    capture_time=time.time()
                )
        return None
    
    def analyze_frame(self, frame_data: FrameData) -> Optional[SceneDetection]:
        """分析单帧（在推理线程中调用）"""
        frame_b64 = frame_to_base64(frame_data.frame)
        
        # [修改] 更新系统提示，强调严格判断标准
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的道路场景分析助手。请**严格、保守**地分析图片中的以下内容：

1. **斑马线检测**（请严格判断）：
   - 只有看到清晰的白色条纹状人行横道才算斑马线
   - 普通的道路标线、车道线、停止线都**不是**斑马线
   - 如果不确定，请返回 has_crosswalk=false
   - 置信度：high=完整清晰可见，medium=部分可见或较远，low=不确定

2. **行人检测**：
   - 只有行人正在斑马线上行走才算 has_pedestrian=true
   - 行人在人行道上、路边站立都不算

3. **交通信号灯**：
   - None=没有或看不清，Red=红灯，Green=绿灯，Yellow=黄灯

4. **十字路口检测**（请严格判断）：
   - 必须能明确看到**多个方向的道路交汇**才算十字路口
   - 普通的直道、弯道都**不是**十字路口
   - 仅仅看到红绿灯不代表是十字路口
   - 如果不确定，请返回 has_intersection=false
   - 置信度：high=明确看到多条道路交汇点，medium=可能是路口，low=不确定或普通道路

**重要**：宁可漏报也不要误报，不确定时请返回 false 和 low 置信度。"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": frame_b64}},
                    {"type": "text", "text": "请严格分析这一帧道路画面，给出结构化结果。注意：不确定时请返回false和low置信度。"}
                ]
            }
        ]
        
        try:
            start_time = time.time()
            response = self.client.beta.chat.completions.parse(
                model="Qwen3-VL-2B-Instruct-4bit",
                messages=messages,
                response_format=SceneDetection,
                max_tokens=800,
                temperature=0.0
            )
            inference_time = time.time() - start_time
            
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            result = response.choices[0].message.parsed
            result.frame_index = frame_data.frame_index
            
            # [修改] 更新日志输出，显示置信度
            print(f"[推理完成] 帧 {frame_data.frame_index} | 耗时 {inference_time:.2f}s | "
                  f"斑马线:{result.has_crosswalk} | "
                  f"行人:{result.has_pedestrian} | 灯:{result.traffic_light} | "
                  f"路口:{result.has_intersection}")
            
            return result
        except Exception as e:
            print(f"[推理失败] 帧 {frame_data.frame_index}: {e}", file=sys.stderr)
            return None
    
    # [修改] 支持多状态变化回调
    def on_states_change(self, new_states: Set[DetectionMode], old_states: Set[DetectionMode], frame_index: int):
        """
        状态变化回调 - 将 CANoe 操作放入队列
        """
        video_seconds = frame_index / self.fps if self.fps > 0 else 0
        video_time_str = time.strftime("%M:%S", time.gmtime(video_seconds))
        video_time_ms = f"{video_time_str}.{int((video_seconds % 1) * 1000):03d}"
        
        # 计算新增和移除的状态
        added_states = new_states - old_states
        removed_states = old_states - new_states
        
        print(f"\n{'='*70}")
        print(f"[状态变化] 视频时间: {video_time_ms} (帧 {frame_index}/{self.total_frames})")
        
        if added_states:
            print(f"  新增状态: {[s.value for s in added_states]}")
        if removed_states:
            print(f"  移除状态: {[s.value for s in removed_states]}")
        
        # 显示当前所有激活状态
        if new_states:
            print(f"  当前激活: {[s.value for s in new_states]}")
        else:
            print(f"  当前激活: [None]")
        
        print(f"{'='*70}")
        
        # 将状态变化放入队列
        self.canoe_action_queue.put((added_states, removed_states, frame_index))
        
        self.last_output_states = new_states.copy()
    
    def process_canoe_actions(self):
        """处理 CANoe 操作队列（在主线程中调用）"""
        while not self.canoe_action_queue.empty():
            try:
                added_states, removed_states, frame_index = self.canoe_action_queue.get_nowait()
                self._execute_canoe_action(added_states, removed_states, frame_index)
            except Empty:
                break
    
    # [修改] 支持多状态的 CANoe 操作
    def _execute_canoe_action(self, added_states: Set[DetectionMode], removed_states: Set[DetectionMode], frame_index: int):
        """执行 CANoe 操作（必须在主线程中调用）"""
        if self.canoe is None:
            print(f"  (CANoe 未连接，跳过控制)")
            return
        
        print(f"  [主线程执行 CANoe 操作]")
        
        # 处理移除的状态
        for state in removed_states:
            print(f"  状态退出: {state.value}")
            # 如果所有状态都退出了，执行 init
            current_states = self.smoother.get_current_states()
            if not current_states:
                print("  执行: init_HD_light()")
                self.canoe.init_HD_light()
        
        # 处理新增的状态
        for state in added_states:
            if state == DetectionMode.GREEN_LIGHT_PED:
                if self.green_light_folder:
                    print(f"  执行: play_ai({self.green_light_folder})")
                    self.canoe.play_ai(str(self.green_light_folder))
                else:
                    print("  警告: 未设置 green_light_folder")
                    
            elif state == DetectionMode.RED_LIGHT_PED:
                if self.red_light_folder:
                    print(f"  执行: play_ai({self.red_light_folder})")
                    self.canoe.play_ai(str(self.red_light_folder))
                else:
                    print("  警告: 未设置 red_light_folder")
            
            elif state == DetectionMode.INTERSECTION:
                if self.intersection_folder:
                    print(f"  执行: play_ai({self.intersection_folder})")
                    self.canoe.play_ai(str(self.intersection_folder))
                else:
                    print("  警告: 未设置 intersection_folder")
    
    def inference_loop(self):
        """推理循环（在后台线程中运行）"""
        print("[推理线程] 启动")
        
        while self.running:
            frame_data = self.get_current_frame_copy()
            
            if frame_data is None:
                time.sleep(0.1)
                continue
            
            result = self.analyze_frame(frame_data)
            
            if not self.running:
                break
            
            if result is not None:
                # [修改] 使用多状态处理
                raw_states = get_raw_states(result)
                smoothed_states = self.smoother.update(raw_states)
                
                # 更新最新结果
                self.latest_result = result
                self.latest_smoothed_states = smoothed_states.copy()
                
                # 检查状态变化
                if smoothed_states != self.last_output_states:
                    self.on_states_change(smoothed_states, self.last_output_states, frame_data.frame_index)
            
            time.sleep(0.01)
        
        print("[推理线程] 结束")
    
    # [修改] 支持多状态显示
    def draw_overlay(self, frame):
        """在帧上绘制叠加信息"""
        result = self.latest_result
        smoothed_states = self.latest_smoothed_states
        
        video_time = self.current_frame_index / self.fps if self.fps > 0 else 0
        time_str = time.strftime("%M:%S", time.gmtime(video_time))
        
        # 第一行：时间和帧号
        line1 = f"Time: {time_str} | Frame: {self.current_frame_index}/{self.total_frames}"
        cv2.putText(frame, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        if result:
            # [修改] 显示置信度信息
            line2 = f"Crosswalk: {'Yes' if result.has_crosswalk else 'No'} | Ped: {'Yes' if result.has_pedestrian else 'No'} | Light: {result.traffic_light}"
            line2b = f"Intersection: {'Yes' if result.has_intersection else 'No'}"
            
            # 根据状态选择颜色
            if result.has_pedestrian:
                text_color = (0, 0, 255)
            elif result.has_intersection and result.intersection_confidence == "high":
                text_color = (255, 165, 0)
            else:
                text_color = (0, 255, 0)
            
            cv2.putText(frame, line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, line2b, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, text_color, 2, cv2.LINE_AA)
            
            # [修改] 显示多状态
            if smoothed_states:
                modes_str = " + ".join([s.value for s in smoothed_states])
            else:
                modes_str = "None"
            line3 = f"Active Modes: {modes_str}"
            
            # 状态颜色（取最高优先级）
            if DetectionMode.GREEN_LIGHT_PED in smoothed_states:
                state_color = (0, 0, 255)
            elif DetectionMode.RED_LIGHT_PED in smoothed_states:
                state_color = (0, 255, 255)
            elif DetectionMode.INTERSECTION in smoothed_states:
                state_color = (255, 165, 0)
            else:
                state_color = (128, 128, 128)
            
            cv2.putText(frame, line3, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, state_color, 2, cv2.LINE_AA)
            
            # 推理统计
            if self.inference_count > 0:
                avg_time = self.total_inference_time / self.inference_count
                line4 = f"Inferences: {self.inference_count} | Avg: {avg_time:.2f}s"
                cv2.putText(frame, line4, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # 红绿灯状态圆圈
            light_colors = {
                "Red": (0, 0, 255),
                "Green": (0, 255, 0),
                "Yellow": (0, 255, 255),
                "None": (128, 128, 128)
            }
            light_color = light_colors.get(result.traffic_light, (128, 128, 128))
            cv2.circle(frame, (frame.shape[1] - 50, 50), 30, light_color, -1)
            cv2.circle(frame, (frame.shape[1] - 50, 50), 30, (255, 255, 255), 2)
            
            # [修改] 多状态警告边框和文字
            warnings = []
            border_color = None
            border_thickness = 2
            
            if DetectionMode.GREEN_LIGHT_PED in smoothed_states:
                warnings.append("WARNING: Pedestrian on Green!")
                border_color = (0, 0, 255)
                border_thickness = 4
            if DetectionMode.RED_LIGHT_PED in smoothed_states:
                warnings.append("Pedestrian on Red")
                if border_color is None:
                    border_color = (0, 255, 255)
            if DetectionMode.INTERSECTION in smoothed_states:
                warnings.append("INTERSECTION")
                if border_color is None:
                    border_color = (255, 165, 0)
                    border_thickness = 3
            
            if border_color:
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), border_color, border_thickness)
            
            # 显示所有警告
            for i, warning in enumerate(warnings):
                y_pos = frame.shape[0] - 30 - i * 30
                cv2.putText(frame, warning, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, border_color or (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Waiting for inference...", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """运行主循环"""
        self.running = True
        
        inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        inference_thread.start()
        
        playback_start_time = time.time()
        video_start_frame = 0
        
        try:
            while self.running:
                elapsed_time = time.time() - playback_start_time
                target_frame = video_start_frame + int(elapsed_time * self.fps)
                
                while self.current_frame_index < target_frame and self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.running = False
                        break
                    
                    with self.frame_lock:
                        self.current_frame = frame
                        self.current_frame_index += 1
                
                if not self.running:
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    print("\n视频播放完毕")
                    break
                
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_frame_index += 1
                
                self.process_canoe_actions()
                
                if self.show_video:
                    display_frame = frame.copy()
                    display_frame = self.draw_overlay(display_frame)
                    
                    if self.display_scale != 1.0:
                        new_width = int(display_frame.shape[1] * self.display_scale)
                        new_height = int(display_frame.shape[0] * self.display_scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    cv2.imshow("Async Video Analysis", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n用户退出")
                        break
                    elif key == ord(' '):
                        print("暂停中... 按空格继续")
                        while True:
                            key = cv2.waitKey(100) & 0xFF
                            if key == ord(' '):
                                playback_start_time = time.time()
                                video_start_frame = self.current_frame_index
                                break
                            elif key == ord('q'):
                                self.running = False
                                break
                
                next_frame_time = playback_start_time + (self.current_frame_index - video_start_frame + 1) / self.fps
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            self.running = False
            inference_thread.join(timeout=2.0)
            self.cap.release()
            if self.show_video:
                cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print("检测结束统计")
            print(f"  总帧数: {self.current_frame_index}")
            print(f"  推理次数: {self.inference_count}")
            if self.inference_count > 0:
                print(f"  平均推理时间: {self.total_inference_time / self.inference_count:.2f}s")
            print(f"{'='*60}")


# ======================
# 命令行入口
# ======================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Async Pedestrian Detection')
    parser.add_argument('video_path', type=str, help='视频文件路径')
    parser.add_argument('--tolerance', type=int, default=2, help='帧容忍数量（默认2）')
    parser.add_argument('--no-display', action='store_true', help='不显示视频窗口')
    parser.add_argument('--no-canoe', action='store_true', help='不连接 CANoe')
    parser.add_argument('--green-folder', type=str, default=None, help='绿灯行人图片文件夹')
    parser.add_argument('--red-folder', type=str, default=None, help='红灯行人图片文件夹')
    parser.add_argument('--intersection-folder', type=str, default=None, help='十字路口图片文件夹')
    parser.add_argument('--scale', type=float, default=0.5, help='视频显示缩放比例（默认0.5）')
    
    args = parser.parse_args()
    
    BASE_PATH = Path(r"C:\Users\uik02458\Desktop\files\PythonProject\YOLO\pics")
    green_folder = args.green_folder or str(BASE_PATH / "green_light_ped_on_crosswalk")
    red_folder = args.red_folder or str(BASE_PATH / "red_light_ped_on_crosswalk")
    intersection_folder = args.intersection_folder or str(BASE_PATH / "intersection")
    
    canoe = None
    if not args.no_canoe:
        canoe = CANoeInterface()
        
        if canoe.connect():
            print("\n--- CANoe 连接成功 ---")
            
            canoe.stop_measurement()
            time.sleep(0.5)
            canoe.start_measurement()
            
            print(f"测量状态: {'运行中' if canoe.is_running() else '已停止'}")
            
            canoe.set_system_variable("SPA_CAR_Panel::NormalMode", 1)
            time.sleep(0.5)
            canoe.set_system_variable("UDP::Connect_L", 1)
            time.sleep(0.5)
            canoe.set_signal_by_path("ZCL_CANFD1::ZCL_CANFD1::ZCL::ZCLZCLCanFD1Frame02::ActnOfProjectAreaHD", 1)
            time.sleep(0.5)
            canoe.set_signal_by_path("ZCL_CANFD1::ZCL_CANFD1::ZCL::ZCLZCLCanFD1Frame10::HDDistortionModeReq", 1)
            time.sleep(0.5)
            canoe.set_signal_by_path("ZCL_CANFD1::ZCL_CANFD1::ZCL::ZCLZCLCanFD1Frame02::HDLightStreamReqLeft", 1)
            time.sleep(0.5)
            canoe.set_system_variable("UDP::AIStartStream", 0)
            time.sleep(0.5)
            
            print("CANoe 初始化完成\n")
        else:
            print("CANoe 连接失败，将以离线模式运行")
            canoe = None
    
    detector = AsyncPedestrianDetector(
        video_path=args.video_path,
        tolerance_frames=args.tolerance,
        show_video=not args.no_display,
        display_scale=args.scale,
        canoe=canoe,
        green_light_folder=green_folder,
        red_light_folder=red_folder,
        intersection_folder=intersection_folder
    )
    
    print(f"\n{'='*60}")
    print("异步行人检测系统启动")
    print(f"视频: {args.video_path}")
    print(f"绿灯文件夹: {green_folder}")
    print(f"红灯文件夹: {red_folder}")
    print(f"十字路口文件夹: {intersection_folder}")
    print(f"CANoe: {'已连接' if canoe else '未连接'}")
    print(f"帧容忍: {args.tolerance}")
    print(f"显示缩放: {args.scale}")
    print(f"{'='*60}")
    print("\n按 'q' 退出 | 按空格暂停/继续\n")
    
    detector.run()


if __name__ == "__main__":
    main()
