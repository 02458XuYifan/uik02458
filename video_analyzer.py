#!/usr/bin/env python3
"""
视频道路场景分析脚本
使用 vLLM 部署的 Qwen3-VL 分析视频中的斑马线、行人和交通灯
"""

import cv2
import base64
import sys
from typing import Optional
from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel


# ============ 数据模型 ============

class SceneDetection(BaseModel):
    """场景检测结果"""
    has_crosswalk: bool          # 是否有斑马线
    has_pedestrian: bool         # 斑马线上是否有行人
    traffic_light: str           # 交通灯状态: None/Red/Green/Yellow
    frame_index: int = 0         # 帧索引
    description: Optional[str] = None  # 可选的描述


# ============ 工具函数 ============

def frame_to_base64(frame) -> str:
    """将 OpenCV 帧转换为 base64 编码的图片 URL"""
    # 编码为 JPEG（比 PNG 更快）
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode frame")
    
    # 转换为 base64
    b64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"


# ============ 分析器类 ============

class VideoAnalyzer:
    """视频分析器"""
    
    def __init__(
        self, 
        base_url: str = "http://10.214.153.118:22002/v1",
        model: str = "Qwen3-VL-8B-Instruct-4bit"
    ):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=3600
        )
        self.model = model
        self.frame_index = 0
    
    def analyze_frame(self, frame) -> Optional[SceneDetection]:
        """分析单帧"""
        frame_b64 = frame_to_base64(frame)
        
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的道路场景分析助手。请仔细分析图片中的以下内容：
1. 是否有斑马线（人行横道）
2. 斑马线上是否有行人正在过马路
3. 交通信号灯的状态（红灯/绿灯/黄灯/无）

对于交通信号灯：
- 如果画面中没有交通信号灯，或者看不清楚，请返回 "None"
- 如果是红灯，返回 "Red"
- 如果是绿灯，返回 "Green"
- 如果是黄灯，返回 "Yellow"

请给出准确的结构化判断结果。"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": frame_b64}},
                    {"type": "text", "text": "请分析这一帧道路画面：1) 是否有斑马线？2) 斑马线上是否有行人？3) 交通信号灯是什么状态（None/Red/Green/Yellow）？"}
                ]
            }
        ]
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=SceneDetection,
                max_tokens=800,
                temperature=0.0
            )
            result = response.choices[0].message.parsed
            result.frame_index = self.frame_index
            return result
        except Exception as e:
            print(f"Frame {self.frame_index} analysis failed: {e}", file=sys.stderr)
            return None
    
    def analyze_video(
        self, 
        video_path: str, 
        fps: float = 1.0,
        max_frames: int = None
    ) -> list[SceneDetection]:
        """
        分析整个视频
        
        Args:
            video_path: 视频文件路径
            fps: 每秒分析多少帧（例如 1.0 表示每秒1帧）
            max_frames: 最大分析帧数（用于测试）
        
        Returns:
            分析结果列表
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"视频信息:")
        print(f"  - 路径: {video_path}")
        print(f"  - 帧率: {video_fps:.2f} FPS")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 时长: {duration:.2f} 秒")
        print(f"  - 分析帧率: {fps} FPS")
        
        # 计算帧间隔
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        print(f"  - 帧间隔: 每 {frame_interval} 帧取1帧")
        print("-" * 50)
        
        results = []
        frame_count = 0
        analyzed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔取帧
            if frame_count % frame_interval == 0:
                self.frame_index = frame_count
                
                print(f"分析帧 {frame_count}/{total_frames}...", end=" ")
                
                result = self.analyze_frame(frame)
                
                if result:
                    results.append(result)
                    print(f"✓ 斑马线:{result.has_crosswalk}, "
                          f"行人:{result.has_pedestrian}, "
                          f"信号灯:{result.traffic_light}")
                else:
                    print("✗ 分析失败")
                
                analyzed_count += 1
                
                # 检查是否达到最大帧数
                if max_frames and analyzed_count >= max_frames:
                    print(f"已达到最大帧数限制 ({max_frames})")
                    break
            
            frame_count += 1
        
        cap.release()
        
        print("-" * 50)
        print(f"分析完成: {len(results)}/{analyzed_count} 帧成功")
        
        return results


# ============ 单张图片测试 ============

def test_single_image(image_path: str):
    """测试单张图片分析"""
    analyzer = VideoAnalyzer()
    
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return
    
    print(f"分析图片: {image_path}")
    result = analyzer.analyze_frame(frame)
    
    if result:
        print(f"\n分析结果:")
        print(f"  - 斑马线: {'是' if result.has_crosswalk else '否'}")
        print(f"  - 行人: {'是' if result.has_pedestrian else '否'}")
        print(f"  - 信号灯: {result.traffic_light}")
        if result.description:
            print(f"  - 描述: {result.description}")
    else:
        print("分析失败")


# ============ 主函数 ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="视频道路场景分析")
    parser.add_argument("input", help="视频文件或图片路径")
    parser.add_argument("--fps", type=float, default=1.0, help="分析帧率 (默认: 1.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="最大分析帧数")
    parser.add_argument("--api-url", default="http://10.214.153.118:22002/v1", help="API 地址")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct-4bit", help="模型名称")
    
    args = parser.parse_args()
    
    # 判断是图片还是视频
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        test_single_image(args.input)
    else:
        analyzer = VideoAnalyzer(base_url=args.api_url, model=args.model)
        results = analyzer.analyze_video(
            args.input, 
            fps=args.fps, 
            max_frames=args.max_frames
        )
        
        # 统计结果
        if results:
            print("\n" + "=" * 50)
            print("统计汇总:")
            crosswalk_count = sum(1 for r in results if r.has_crosswalk)
            pedestrian_count = sum(1 for r in results if r.has_pedestrian)
            
            traffic_lights = {}
            for r in results:
                traffic_lights[r.traffic_light] = traffic_lights.get(r.traffic_light, 0) + 1
            
            print(f"  - 检测到斑马线的帧数: {crosswalk_count}/{len(results)}")
            print(f"  - 检测到行人的帧数: {pedestrian_count}/{len(results)}")
            print(f"  - 信号灯统计: {traffic_lights}")


if __name__ == "__main__":
    main()
