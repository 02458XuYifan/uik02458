# sign_async_v2.py - 优化交通标志检测

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

from connect import CANoeInterface


# ======================
# 状态枚举
# ======================
class Mode(Enum):
    NONE = "None"
    GREEN_LIGHT_PED = "green_light_ped"
    RED_LIGHT_PED = "red_light_ped"
    INTERSECTION = "intersection"
    SPEED_LIMIT = "speed_limit"
    STOP_SIGN = "stop"
    SCHOOL_ZONE = "school_zone"


# ======================
# 标志类型枚举（关键优化）
# ======================
SignType = Literal[
    "None",
    # 限速
    "SpeedLimit_30", "SpeedLimit_40", "SpeedLimit_50", 
    "SpeedLimit_60", "SpeedLimit_80", "SpeedLimit_100", "SpeedLimit_120",
    # 禁令
    "Stop", "Yield", "NoEntry", "NoParking", "NoOvertaking"
]


# ======================
# Pydantic 结构化输出
# ======================
class SignInfo(BaseModel):
    """单个交通标志"""
    type: SignType = Field(description="标志类型")
    confidence: Literal["High", "Medium", "Low"] = Field(
        default="Medium", 
        description="置信度：High=清晰可见非常确定, Medium=比较确定, Low=不太清楚可能误判"
    )


class SceneResult(BaseModel):
    """道路场景检测结果"""
    # 行人
    has_crosswalk: bool = Field(default=False, description="是否有斑马线")
    has_pedestrian: bool = Field(default=False, description="斑马线上是否有行人")
    
    # 信号灯
    light: Literal["None", "Red", "Green", "Yellow"] = Field(default="None")
    
    # 路口
    has_intersection: bool = Field(default=False, description="是否有路口")
    intersection_dist: Literal["Approaching", "At", "Passed", "Far", "Unknown"] = Field(default="Unknown")
    
    # 交通标志（优化后）
    has_sign: bool = Field(default=False, description="是否有清晰可见的交通标志")
    signs: List[SignInfo] = Field(default_factory=list, description="检测到的标志列表")
    speed_limit: Optional[int] = Field(default=None, description="限速值（仅当检测到限速标志时）")


# ======================
# 帧数据
# ======================
@dataclass
class Frame:
    data: any
    idx: int
    time: float = 0.0


# ======================
# 状态管理
# ======================
@dataclass
class Tracker:
    current: Mode = Mode.NONE
    pending: Optional[Mode] = None
    count: int = 0


class StateMgr:
    """状态管理器"""
    
    # 标志关键词到模式的映射
    SIGN_MAP = {
        "Stop": Mode.STOP_SIGN,
        "School": Mode.SCHOOL_ZONE,
        "SpeedLimit": Mode.SPEED_LIMIT,
    }
    
    def __init__(self, tolerance: int = 2, sign_confidence_threshold: str = "Medium"):
        self.tolerance = tolerance
        self.sign_threshold = sign_confidence_threshold  # 新增：置信度阈值
        self._lock = threading.Lock()
        self.trackers = {k: Tracker() for k in ['ped', 'inter', 'sign']}
        self.speed_limit: Optional[int] = None
        self.current_signs: List[SignInfo] = []
    
    def update(self, det: SceneResult) -> dict:
        """更新状态，返回变化"""
        with self._lock:
            raws = {
                'ped': self._ped_mode(det),
                'inter': self._int_mode(det),
                'sign': self._sign_mode(det)
            }
            
            changes = {}
            for k, raw in raws.items():
                t = self.trackers[k]
                old = t.current
                self._update_one(t, raw)
                changes[k] = {'changed': old != t.current, 'old': old, 'new': t.current}
            
            self.speed_limit = det.speed_limit
            self.current_signs = det.signs
            return changes
    
    def _update_one(self, t: Tracker, raw: Mode):
        """更新单个跟踪器"""
        if t.current == Mode.NONE:
            if raw != Mode.NONE:
                if t.pending == raw:
                    t.count += 1
                    if t.count >= self.tolerance:
                        t.current, t.pending, t.count = raw, None, 0
                else:
                    t.pending, t.count = raw, 1
        else:
            if raw == t.current:
                t.pending, t.count = None, 0
            else:
                t.count += 1
                if t.count >= self.tolerance:
                    t.current, t.count = (raw if raw != Mode.NONE else Mode.NONE), 0
    
    def _ped_mode(self, d: SceneResult) -> Mode:
        if d.has_pedestrian:
            return Mode.GREEN_LIGHT_PED if d.light == "Green" else \
                   Mode.RED_LIGHT_PED if d.light == "Red" else Mode.NONE
        return Mode.NONE
    
    def _int_mode(self, d: SceneResult) -> Mode:
        return Mode.INTERSECTION if d.has_intersection and d.intersection_dist in ["At"] else Mode.NONE
    
    def _sign_mode(self, d: SceneResult) -> Mode:
        """获取标志模式（带置信度过滤）"""
        if not d.has_sign or not d.signs:
            return Mode.NONE
        
        # 置信度优先级
        confidence_order = {"High": 3, "Medium": 2, "Low": 1}
        threshold_value = confidence_order.get(self.sign_threshold, 2)
        
        # 过滤低置信度标志
        valid_signs = [
            s for s in d.signs 
            if confidence_order.get(s.confidence, 0) >= threshold_value
        ]
        
        if not valid_signs:
            return Mode.NONE
        
        # 按优先级检查
        for keyword, mode in [("Stop", Mode.STOP_SIGN), 
                               ("School", Mode.SCHOOL_ZONE), 
                               ("SpeedLimit", Mode.SPEED_LIMIT)]:
            for sign in valid_signs:
                if keyword in sign.type:
                    return mode
        
        return Mode.NONE
    
    def priority(self) -> Mode:
        """获取最高优先级"""
        with self._lock:
            for k in ['ped', 'sign', 'inter']:
                if (s := self.trackers[k].current) != Mode.NONE:
                    return s
            return Mode.NONE


# ======================
# 辅助函数
# ======================
def to_base64(frame) -> str:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ======================
# 检测器
# ======================
class Detector:
    
    FOLDERS = {
        Mode.GREEN_LIGHT_PED: 'green_light_ped',
        Mode.RED_LIGHT_PED: 'red_light_ped',
        Mode.INTERSECTION: 'intersection',
        Mode.SPEED_LIMIT: 'speed_limit',
        Mode.STOP_SIGN: 'stop_sign',
        Mode.SCHOOL_ZONE: 'school_zone',
    }
    
    # 优化后的提示词
    PROMPT = """你是专业的道路场景分析助手。请仔细分析图片：

1. 行人检测
- 是否有斑马线（人行横道）
- 斑马线上是否有行人正在过马路

2. 交通信号灯
- 状态：None（无/看不清）、Red、Green、Yellow

3. 十字路口
- 是否有路口
- 距离：Approaching（接近中）、At（正在通过）、Passed（已通过）、Far（较远）、Unknown

4. 交通标志（请仔细辨认，不要猜测）

【限速标志】圆形，白底红边，中间有黑色数字
- SpeedLimit_30/40/50/60/80/100/120

【禁令标志】
- Stop: 八角形，红底白字，写有 STOP 或 停
- NoEntry: 圆形，红底，中间白色横杠
- Yield: 倒三角形，红边白底

⚠️ 重要提示：
1. 只报告清晰可见、能够确认的标志
2. 对每个标志给出置信度：High（非常清晰确定）、Medium（比较确定）、Low（不太清楚）
3. 如果看不清或不确定，不要猜测，返回空列表
4. 宁可漏报，不要误报"""

    def __init__(self, video: str, tolerance: int = 2, show: bool = True,
                 scale: float = 0.5, canoe=None, assets: Path = None,
                 sign_confidence: str = "Medium"):
        
        self.show = show
        self.scale = scale
        self.canoe = canoe
        self.assets = assets or Path(".")
        
        # 状态管理器，传入置信度阈值
        self.state = StateMgr(tolerance, sign_confidence)
        self.latest: Optional[SceneResult] = None
        
        self.client = OpenAI(api_key="EMPTY", base_url="http://10.214.153.118:22002/v1", timeout=3600)
        
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开: {video}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self.idx = 0
        self.queue: Queue = Queue()
        
        self.stats = {'count': 0, 'time': 0.0}
        
        print(f"视频: {self.total}帧, {self.fps:.0f}FPS")
        print(f"标志置信度阈值: {sign_confidence}")
    
    def analyze(self, f: Frame) -> Optional[SceneResult]:
        """VLM分析"""
        try:
            t0 = time.time()
            resp = self.client.beta.chat.completions.parse(
                model="Qwen3-VL-8B-Instruct-4bit",
                messages=[
                    {"role": "system", "content": self.PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": to_base64(f.data)}},
                        {"type": "text", "text": "请分析这张道路图片"}
                    ]}
                ],
                response_format=SceneResult,
                max_tokens=500,
                temperature=0.0
            )
            dt = time.time() - t0
            self.stats['count'] += 1
            self.stats['time'] += dt
            
            r = resp.choices[0].message.parsed
            
            # 打印检测结果（包含置信度）
            sign_str = ", ".join([f"{s.type}" for s in r.signs]) if r.signs else "None"
            print(f"[{f.idx}] {dt:.1f}s | 斑马线:{r.has_crosswalk} 行人:{r.has_pedestrian} 灯:{r.light} "
                  f"路口:{r.has_intersection} 标志:[{sign_str}]")
            
            return r
        except Exception as e:
            print(f"[错误] {e}", file=sys.stderr)
            return None
    
    def get_frame(self) -> Optional[Frame]:
        with self.lock:
            return Frame(self.frame.copy(), self.idx, self.idx/self.fps) if self.frame is not None else None
    
    def infer_loop(self):
        """推理线程"""
        while self.running:
            if (f := self.get_frame()) is None:
                time.sleep(0.1)
                continue
            
            if (r := self.analyze(f)) and self.running:
                self.latest = r
                for k, c in self.state.update(r).items():
                    if c['changed']:
                        print(f"  >>> [{k}] {c['old'].value} -> {c['new'].value}")
                        self.queue.put((c['new'], r))
            
            time.sleep(0.01)
    
    def do_canoe(self):
        """处理CANoe"""
        while not self.queue.empty():
            try:
                mode, det = self.queue.get_nowait()
                if not self.canoe:
                    continue
                if mode == Mode.NONE:
                    self.canoe.init_HD_light()
                elif (key := self.FOLDERS.get(mode)):
                    if mode == Mode.SPEED_LIMIT and det.speed_limit:
                        key = f"speed_limit_{det.speed_limit}"
                    if (p := self.assets / key).exists():
                        self.canoe.play_ai(str(p))
            except Empty:
                break
    
    def overlay(self, frame):
        """绘制叠加"""
        h, w = frame.shape[:2]
        d = self.latest
        
        # 背景（稍微加大以显示更多信息）
        cv2.rectangle(frame, (10,10), (400,130), (0,0,0), -1)
        cv2.addWeighted(frame, 0.6, frame, 0.4, 0, frame)
        
        # 时间信息
        t = time.strftime("%M:%S", time.gmtime(self.idx/self.fps))
        cv2.putText(frame, f"{t} | {self.idx}/{self.total}", (20,35), 0, 0.6, (255,255,255), 1)
        
        if d:
            # 基本信息
            cv2.putText(frame, f"Ped:{d.has_pedestrian} Light:{d.light} Int:{d.has_intersection}",
                        (20,60), 0, 0.5, (200,200,200), 1)
            
            # 标志信息（显示置信度）
            if d.signs:
                sign_strs = [f"{s.type[:15]}({s.confidence[0]})" for s in d.signs[:2]]
                cv2.putText(frame, f"Signs: {', '.join(sign_strs)}", (20,85), 0, 0.45, (200,200,200), 1)
            
            # 限速
            if d.speed_limit:
                cv2.putText(frame, f"Limit: {d.speed_limit}km/h", (20,110), 0, 0.5, (0,165,255), 1)
            
            # 信号灯
            colors = {"Red":(0,0,255), "Green":(0,255,0), "Yellow":(0,255,255), "None":(128,128,128)}
            cv2.circle(frame, (w-40,40), 22, colors.get(d.light,(128,128,128)), -1)
            
            # 警告边框
            p = self.state.priority()
            if p == Mode.GREEN_LIGHT_PED:
                cv2.rectangle(frame, (2,2), (w-2,h-2), (0,0,255), 3)
                cv2.putText(frame, "WARNING: Pedestrian!", (20,h-15), 0, 0.7, (0,0,255), 2)
            elif p == Mode.STOP_SIGN:
                cv2.rectangle(frame, (2,2), (w-2,h-2), (0,0,255), 2)
                cv2.putText(frame, "STOP SIGN", (20,h-15), 0, 0.7, (0,0,255), 2)
        
        return frame
    
    def run(self):
        """主循环"""
        self.running = True
        threading.Thread(target=self.infer_loop, daemon=True).start()
        
        t0, base = time.time(), 0
        
        try:
            while self.running:
                target = base + int((time.time()-t0) * self.fps)
                while self.idx < target and self.running:
                    ret, f = self.cap.read()
                    if not ret:
                        self.running = False
                        break
                    with self.lock:
                        self.frame, self.idx = f, self.idx + 1
                
                if not self.running:
                    break
                
                ret, f = self.cap.read()
                if not ret:
                    break
                with self.lock:
                    self.frame, self.idx = f, self.idx + 1
                
                self.do_canoe()
                
                if self.show:
                    disp = self.overlay(f.copy())
                    if self.scale != 1:
                        disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale)
                    cv2.imshow("Detect", disp)
                    
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord(' '):
                        while cv2.waitKey(100) & 0xFF != ord(' '):
                            pass
                        t0, base = time.time(), self.idx
                
                if (s := t0 + (self.idx-base+1)/self.fps - time.time()) > 0:
                    time.sleep(s)
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
            if self.stats['count']:
                print(f"\n统计: {self.stats['count']}次, 平均{self.stats['time']/self.stats['count']:.2f}s")


# ======================
# 入口
# ======================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--tolerance', type=int, default=2, help='状态切换容忍次数')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--no-canoe', action='store_true')
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--assets', type=str, default=None)
    p.add_argument('--sign-confidence', type=str, default="High", 
                   choices=["High", "Medium", "Low"],
                   help='标志检测置信度阈值：High=只接受高置信度, Medium=接受中高, Low=全部接受')
    a = p.parse_args()
    
    assets = Path(a.assets) if a.assets else Path(r"C:\Users\uik02458\Desktop\files\PythonProject\YOLO\pics")
    
    canoe = None
    if not a.no_canoe:
        canoe = CANoeInterface()
        if canoe.connect():
            canoe.stop_measurement()
            time.sleep(0.5)
            canoe.start_measurement()
            for v, n in [("SPA_CAR_Panel::NormalMode",1), ("UDP::Connect_L",1), ("UDP::AIStartStream",0)]:
                canoe.set_system_variable(v, n)
                time.sleep(0.2)
            for s in ["ZCLZCLCanFD1Frame02::ActnOfProjectAreaHD", "ZCLZCLCanFD1Frame10::HDDistortionModeReq",
                      "ZCLZCLCanFD1Frame02::HDLightStreamReqLeft"]:
                canoe.set_signal_by_path(f"ZCL_CANFD1::ZCL_CANFD1::ZCL::{s}", 1)
                time.sleep(0.2)
        else:
            canoe = None
    
    print(f"\n{'='*50}")
    print(f"场景检测 | q退出 | 空格暂停")
    print(f"标志置信度阈值: {a.sign_confidence}")
    print(f"状态切换容忍: {a.tolerance}次")
    print(f"{'='*50}\n")
    
    Detector(
        a.video, 
        a.tolerance, 
        not a.no_display, 
        a.scale, 
        canoe, 
        assets,
        a.sign_confidence
    ).run()


if __name__ == "__main__":
    main()
