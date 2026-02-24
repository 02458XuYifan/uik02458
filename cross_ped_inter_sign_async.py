# sign_async_simple.py - 简化版多场景异步检测脚本（无风险评估）

import cv2
import base64
import sys
import time
import threading
from pathlib import Path
from io import BytesIO
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from dataclasses import dataclass
from queue import Queue, Empty
from enum import Enum

# 导入 CANoe 接口
from connect import CANoeInterface


# ======================
# 状态枚举
# ======================
class DetectionMode(Enum):
    NONE = "None"
    GREEN_LIGHT_PED = "green_light_ped"
    RED_LIGHT_PED = "red_light_ped"
    INTERSECTION_AHEAD = "intersection_ahead"
    SIGN_SPEED_LIMIT = "speed_limit"
    SIGN_STOP = "stop"
    SIGN_SCHOOL_ZONE = "school_zone"


# ======================
# Pydantic 结构化输出
# ======================
class SceneDetection(BaseModel):
    """道路场景检测结果"""
    
    # 行人检测
    has_crosswalk: bool = Field(default=False, description="画面中是否有斑马线")
    has_pedestrian: bool = Field(default=False, description="斑马线上是否有行人正在过马路")
    
    # 信号灯
    light_state: Literal["None", "Red", "Green", "Yellow"] = Field(
        default="None", description="交通信号灯状态"
    )
    
    # 路口
    has_intersection: bool = Field(default=False, description="前方是否有十字路口")
    intersection_type: Literal["None", "T_Junction", "Cross", "Y_Junction", "Roundabout"] = Field(
        default="None", description="路口类型"
    )
    intersection_distance: Literal["Approaching", "At", "Passed", "Far", "Unknown"] = Field(
        default="Unknown", description="与路口的距离"
    )
    
    # 交通标志
    has_traffic_sign: bool = Field(default=False, description="画面中是否有交通标志")
    sign_types: List[str] = Field(default_factory=list, description="检测到的交通标志类型列表")
    speed_limit: Optional[int] = Field(default=None, description="限速值")
    
    # 场景描述
    scene_description: str = Field(default="", description="场景简要描述")


# ======================
# 帧数据容器
# ======================
@dataclass
class FrameData:
    frame: any
    frame_index: int
    timestamp: float = 0.0


# ======================
# 状态跟踪器
# ======================
@dataclass
class StateTracker:
    current: DetectionMode = DetectionMode.NONE
    pending: Optional[DetectionMode] = None
    count: int = 0


class StateManager:
    """状态管理器"""
    
    def __init__(self, tolerance: int = 3):
        self.tolerance = tolerance
        self._lock = threading.Lock()
        self.trackers = {
            'pedestrian': StateTracker(),
            'intersection': StateTracker(),
            'sign': StateTracker()
        }
        self.speed_limit: Optional[int] = None
        self.sign_types: List[str] = []
    
    def _update_tracker(self, tracker: StateTracker, raw: DetectionMode) -> bool:
        """更新单个跟踪器，返回是否变化"""
        old = tracker.current
        
        if tracker.current == DetectionMode.NONE:
            if raw != DetectionMode.NONE:
                if tracker.pending == raw:
                    tracker.count += 1
                    if tracker.count >= self.tolerance:
                        tracker.current = raw
                        tracker.pending = None
                        tracker.count = 0
                else:
                    tracker.pending = raw
                    tracker.count = 1
        else:
            if raw == tracker.current:
                tracker.pending = None
                tracker.count = 0
            else:
                tracker.count += 1
                if tracker.count >= self.tolerance:
                    tracker.current = raw if raw != DetectionMode.NONE else DetectionMode.NONE
                    tracker.count = 0
        
        return old != tracker.current
    
    def update(self, det: SceneDetection) -> dict:
        """更新所有状态"""
        with self._lock:
            raw_states = {
                'pedestrian': self._get_ped_mode(det),
                'intersection': self._get_int_mode(det),
                'sign': self._get_sign_mode(det)
            }
            
            changes = {}
            for key, raw in raw_states.items():
                tracker = self.trackers[key]
                old = tracker.current
                changed = self._update_tracker(tracker, raw)
                changes[key] = {'changed': changed, 'old': old, 'new': tracker.current}
            
            self.speed_limit = det.speed_limit
            self.sign_types = det.sign_types
            
            return changes
    
    def _get_ped_mode(self, det: SceneDetection) -> DetectionMode:
        if det.has_pedestrian:
            if det.light_state == "Green":
                return DetectionMode.GREEN_LIGHT_PED
            elif det.light_state == "Red":
                return DetectionMode.RED_LIGHT_PED
        return DetectionMode.NONE
    
    def _get_int_mode(self, det: SceneDetection) -> DetectionMode:
        if det.has_intersection and det.intersection_distance in ["Approaching", "At"]:
            return DetectionMode.INTERSECTION_AHEAD
        return DetectionMode.NONE
    
    def _get_sign_mode(self, det: SceneDetection) -> DetectionMode:
        if not det.has_traffic_sign:
            return DetectionMode.NONE
        signs = det.sign_types
        for keyword, mode in [("Stop", DetectionMode.SIGN_STOP), 
                               ("School", DetectionMode.SIGN_SCHOOL_ZONE),
                               ("SpeedLimit", DetectionMode.SIGN_SPEED_LIMIT)]:
            if any(keyword in s for s in signs):
                return mode
        return DetectionMode.NONE
    
    def get_priority_mode(self) -> DetectionMode:
        """获取最高优先级状态"""
        with self._lock:
            for key in ['pedestrian', 'sign', 'intersection']:
                state = self.trackers[key].current
                if state != DetectionMode.NONE:
                    return state
            return DetectionMode.NONE


# ======================
# 辅助函数
# ======================
def frame_to_base64(frame) -> str:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ======================
# 检测器主类
# ======================
class SceneDetector:
    
    MODE_FOLDERS = {
        DetectionMode.GREEN_LIGHT_PED: 'green_light_ped',
        DetectionMode.RED_LIGHT_PED: 'red_light_ped',
        DetectionMode.INTERSECTION_AHEAD: 'intersection',
        DetectionMode.SIGN_SPEED_LIMIT: 'speed_limit',
        DetectionMode.SIGN_STOP: 'stop_sign',
        DetectionMode.SIGN_SCHOOL_ZONE: 'school_zone',
    }
    
    SYSTEM_PROMPT = """你是道路场景分析助手。请分析图片中的：

1. 行人检测
   - 是否有斑马线
   - 斑马线上是否有行人正在过马路

2. 交通信号灯
   - 状态：None/Red/Green/Yellow

3. 十字路口
   - 是否有路口
   - 类型：None/T_Junction/Cross/Y_Junction/Roundabout
   - 距离：Approaching/At/Passed/Far/Unknown

4. 交通标志
   - 是否有标志
   - 类型列表：SpeedLimit_30, SpeedLimit_60, Stop, SchoolZone 等
   - 限速值（如有）

5. 简要描述场景"""

    def __init__(self, video_path: str, tolerance: int = 3, show_video: bool = True,
                 scale: float = 0.5, canoe: CANoeInterface = None, asset_base: Path = None):
        
        self.show_video = show_video
        self.scale = scale
        self.canoe = canoe
        self.asset_base = asset_base or Path(".")
        
        self.state_mgr = StateManager(tolerance)
        self.latest: Optional[SceneDetection] = None
        
        self.client = OpenAI(api_key="EMPTY", base_url="http://10.214.153.118:22002/v1", timeout=3600)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.current_idx = 0
        self.action_queue: Queue = Queue()
        
        self.infer_count = 0
        self.infer_time = 0.0
        
        print(f"视频: {self.total_frames}帧, {self.fps:.1f}FPS")
    
    def analyze(self, fd: FrameData) -> Optional[SceneDetection]:
        """分析单帧"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": frame_to_base64(fd.frame)}},
                {"type": "text", "text": "分析此道路场景"}
            ]}
        ]
        
        try:
            t0 = time.time()
            resp = self.client.beta.chat.completions.parse(
                model="Qwen3-VL-8B-Instruct-4bit",
                messages=messages,
                response_format=SceneDetection,
                max_tokens=600,
                temperature=0.0
            )
            dt = time.time() - t0
            self.infer_count += 1
            self.infer_time += dt
            
            result = resp.choices[0].message.parsed
            
            print(f"[帧{fd.frame_index}] {dt:.1f}s | 斑马线:{result.has_crosswalk} 行人:{result.has_pedestrian} "
                  f"灯:{result.light_state} 路口:{result.has_intersection} 标志:{result.sign_types}")
            return result
        except Exception as e:
            print(f"[推理失败] {e}", file=sys.stderr)
            return None
    
    def get_frame(self) -> Optional[FrameData]:
        with self.frame_lock:
            if self.current_frame is not None:
                return FrameData(self.current_frame.copy(), self.current_idx, self.current_idx / self.fps)
        return None
    
    def inference_loop(self):
        """推理线程"""
        while self.running:
            fd = self.get_frame()
            if fd is None:
                time.sleep(0.1)
                continue
            
            result = self.analyze(fd)
            if not self.running:
                break
            
            if result:
                self.latest = result
                changes = self.state_mgr.update(result)
                
                for cat, chg in changes.items():
                    if chg['changed']:
                        print(f"  [{cat}] {chg['old'].value} -> {chg['new'].value}")
                        self.action_queue.put((chg['new'], result))
            
            time.sleep(0.01)
    
    def process_actions(self):
        """处理 CANoe 操作"""
        while not self.action_queue.empty():
            try:
                mode, det = self.action_queue.get_nowait()
                self._do_canoe(mode, det)
            except Empty:
                break
    
    def _do_canoe(self, mode: DetectionMode, det: SceneDetection):
        """执行 CANoe 操作"""
        if not self.canoe:
            return
        
        if mode == DetectionMode.NONE:
            self.canoe.init_HD_light()
        else:
            folder_key = self.MODE_FOLDERS.get(mode)
            if folder_key:
                if mode == DetectionMode.SIGN_SPEED_LIMIT and det.speed_limit:
                    folder_key = f"speed_limit_{det.speed_limit}"
                folder = self.asset_base / folder_key
                if folder.exists():
                    self.canoe.play_ai(str(folder))
    
    def draw_overlay(self, frame):
        """绘制叠加信息"""
        det = self.latest
        h, w = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 时间和帧号
        time_str = time.strftime("%M:%S", time.gmtime(self.current_idx / self.fps))
        cv2.putText(frame, f"{time_str} | Frame {self.current_idx}/{self.total_frames}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if det:
            # 检测信息
            info = f"Ped:{det.has_pedestrian} | Light:{det.light_state} | Int:{det.has_intersection}"
            cv2.putText(frame, info, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if det.sign_types:
                cv2.putText(frame, f"Signs: {', '.join(det.sign_types[:2])}", 
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if det.speed_limit:
                cv2.putText(frame, f"Limit: {det.speed_limit} km/h",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # 信号灯圆圈
            light_colors = {"Red": (0,0,255), "Green": (0,255,0), "Yellow": (0,255,255), "None": (128,128,128)}
            cv2.circle(frame, (w-40, 40), 25, light_colors.get(det.light_state, (128,128,128)), -1)
            
            # 警告边框
            priority = self.state_mgr.get_priority_mode()
            if priority == DetectionMode.GREEN_LIGHT_PED:
                cv2.rectangle(frame, (3, 3), (w-3, h-3), (0, 0, 255), 3)
                cv2.putText(frame, "WARNING: Pedestrian!", (20, h-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """主循环"""
        self.running = True
        threading.Thread(target=self.inference_loop, daemon=True).start()
        
        t0 = time.time()
        start_frame = 0
        
        try:
            while self.running:
                target = start_frame + int((time.time() - t0) * self.fps)
                while self.current_idx < target:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.running = False
                        break
                    with self.frame_lock:
                        self.current_frame = frame
                        self.current_idx += 1
                
                if not self.running:
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_idx += 1
                
                self.process_actions()
                
                if self.show_video:
                    disp = self.draw_overlay(frame.copy())
                    if self.scale != 1.0:
                        disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale)
                    cv2.imshow("Detection", disp)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        while cv2.waitKey(100) & 0xFF != ord(' '):
                            pass
                        t0, start_frame = time.time(), self.current_idx
                
                next_time = t0 + (self.current_idx - start_frame + 1) / self.fps
                if (sleep_t := next_time - time.time()) > 0:
                    time.sleep(sleep_t)
        
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
            
            if self.infer_count:
                print(f"\n统计: {self.infer_count}次推理, 平均{self.infer_time/self.infer_count:.2f}s")


# ======================
# 入口
# ======================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--tolerance', type=int, default=2)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-canoe', action='store_true')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--asset-base', type=str, default=None)
    args = parser.parse_args()
    
    asset_base = Path(args.asset_base) if args.asset_base else \
                 Path(r"C:\Users\uik02458\Desktop\files\PythonProject\YOLO\pics")
    
    canoe = None
    if not args.no_canoe:
        canoe = CANoeInterface()
        if canoe.connect():
            canoe.stop_measurement()
            time.sleep(0.5)
            canoe.start_measurement()
            for var, val in [("SPA_CAR_Panel::NormalMode", 1), ("UDP::Connect_L", 1), ("UDP::AIStartStream", 0)]:
                canoe.set_system_variable(var, val)
                time.sleep(0.3)
            for sig in ["ZCLZCLCanFD1Frame02::ActnOfProjectAreaHD", "ZCLZCLCanFD1Frame10::HDDistortionModeReq", 
                        "ZCLZCLCanFD1Frame02::HDLightStreamReqLeft"]:
                canoe.set_signal_by_path(f"ZCL_CANFD1::ZCL_CANFD1::ZCL::{sig}", 1)
                time.sleep(0.3)
        else:
            canoe = None
    
    detector = SceneDetector(
        args.video_path,
        tolerance=args.tolerance,
        show_video=not args.no_display,
        scale=args.scale,
        canoe=canoe,
        asset_base=asset_base
    )
    
    print(f"\n{'='*50}\n场景检测启动\n按 'q' 退出 | 空格暂停\n{'='*50}\n")
    detector.run()


if __name__ == "__main__":
    main()
