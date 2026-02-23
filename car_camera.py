import cv2
import base64
import tempfile
import os
import time
from openai import OpenAI


class VideoClipAnalyzer:
    """视频片段分析 - 录制短视频片段后一次性分析"""
    
    def __init__(
        self,
        api_url: str = "http://10.214.153.118:22002/v1",
        model: str = "Qwen3-VL-8B-Instruct-4bit",
        clip_duration: float = 3.0,  # 每个片段3秒
    ):
        self.client = OpenAI(api_key="EMPTY", base_url=api_url, timeout=120)
        self.model = model
        self.clip_duration = clip_duration
    
    def analyze_video_clip(self, video_path: str) -> str:
        """分析视频片段"""
        print(f"正在读取视频: {video_path}")
        file_size = os.path.getsize(video_path) / 1024  # KB
        print(f"视频大小: {file_size:.1f} KB")
        
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        print("正在发送到 VLM 分析...")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                    },
                    {
                        "type": "text",
                        "text": """你是一个专业的道路场景分析助手。请仔细分析图片中的以下内容：
1. 是否有斑马线（人行横道）
2. 斑马线上是否有行人正在过马路
3. 交通信号灯的状态（红灯/绿灯/黄灯/无）
4. 是否来到十字路口

对于交通信号灯：
- 如果画面中没有交通信号灯，或者看不清楚，请返回 "None"
- 如果是红灯，返回 "Red"
- 如果是绿灯，返回 "Green"
- 如果是黄灯，返回 "Yellow"

请给出准确的结构化判断结果。"""
                    }
                ]
            }],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def run(self, camera_id: int = 0):
        """循环录制并分析"""
        print("=" * 60)
        print("视频片段分析器 (自动连续模式)")
        print("=" * 60)
        print(f"摄像头 ID: {camera_id}")
        print(f"每个片段时长: {self.clip_duration} 秒")
        print("按 'q' 退出程序，按 's' 跳过当前录制")
        print("每个片段分析完成后会自动继续下一个片段")
        print("=" * 60)
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头！")
            print("请检查：")
            print("  - 摄像头是否连接")
            print("  - 是否被其他程序占用")
            print("  - 尝试不同的 camera_id (0, 1, 2...)")
            return
        
        # 获取摄像头参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 默认30fps
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"摄像头分辨率: {width}x{height}")
        print(f"摄像头帧率: {fps}")
        print("-" * 60)
        
        clip_count = 0
        should_exit = False
        
        try:
            while not should_exit:
                clip_count += 1
                print(f"\n[片段 {clip_count}] 准备录制...")
                
                # 创建临时视频文件
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                temp_path = temp_file.name
                temp_file.close()
                
                # 设置视频写入器（使用较小的分辨率）
                output_width, output_height = 640, 360
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, fps, (output_width, output_height))
                
                frames_to_record = int(fps * self.clip_duration)
                frames_recorded = 0
                skip_clip = False
                
                print(f"正在录制 {self.clip_duration} 秒 ({frames_to_record} 帧)...")
                
                # 录制视频片段
                start_time = time.time()
                while frames_recorded < frames_to_record:
                    ret, frame = cap.read()
                    if not ret:
                        print("读取帧失败")
                        break
                    
                    # 缩小帧
                    frame_resized = cv2.resize(frame, (output_width, output_height))
                    out.write(frame_resized)
                    frames_recorded += 1
                    
                    # 在画面上显示录制状态
                    progress = frames_recorded / frames_to_record
                    cv2.rectangle(frame, (10, 10), (210, 50), (0, 0, 0), -1)
                    cv2.putText(frame, f"Recording: {progress*100:.0f}%", 
                                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 显示画面
                    cv2.imshow('Camera - Recording', frame)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n用户退出")
                        should_exit = True
                        break
                    elif key == ord('s'):
                        print("跳过此片段")
                        skip_clip = True
                        break
                
                out.release()
                
                if should_exit:
                    os.remove(temp_path)
                    break
                
                record_time = time.time() - start_time
                print(f"录制完成: {frames_recorded} 帧, 耗时 {record_time:.2f} 秒")
                
                if skip_clip:
                    os.remove(temp_path)
                    continue
                
                # 分析视频片段（同时继续显示摄像头画面）
                print("-" * 40)
                try:
                    analysis_start = time.time()
                    
                    # 在分析过程中继续显示摄像头画面（显示分析中状态）
                    # 注意：由于 analyze_video_clip 是阻塞的，这里只是在分析前后显示状态
                    ret, frame = cap.read()
                    if ret:
                        cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
                        cv2.putText(frame, "Analyzing...", 
                                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.imshow('Camera - Recording', frame)
                        cv2.waitKey(1)
                    
                    result = self.analyze_video_clip(temp_path)
                    analysis_time = time.time() - analysis_start
                    
                    print(f"\n分析结果 (耗时 {analysis_time:.2f} 秒):")
                    print("-" * 40)
                    print(result)
                    print("-" * 40)
                    
                except Exception as e:
                    print(f"分析失败: {e}")
                
                # 删除临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                # 自动继续下一个片段，不需要等待用户确认
                print("\n自动继续下一个片段...")
                
                # 短暂检查是否有退出请求
                for _ in range(5):  # 检查约50ms
                    ret, frame = cap.read()
                    if ret:
                        cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
                        cv2.putText(frame, "Starting next clip...", 
                                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Camera - Recording', frame)
                    
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        should_exit = True
                        break
        
        except KeyboardInterrupt:
            print("\n程序被中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n程序结束")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="车载摄像头视频片段分析（自动连续模式）")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID (默认: 0)")
    parser.add_argument("--duration", type=float, default=3.0, help="每个片段时长秒数 (默认: 3.0)")
    parser.add_argument("--api-url", default="http://10.214.153.118:22002/v1", help="API地址")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct-4bit", help="模型名称")
    
    args = parser.parse_args()
    
    analyzer = VideoClipAnalyzer(
        api_url=args.api_url,
        model=args.model,
        clip_duration=args.duration,
    )
    
    analyzer.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
