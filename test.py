from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.214.153.118:22002/v1",
    timeout=3600
)

# ========== 请修改为你的视频文件路径 ==========
VIDEO_PATH = r"C:\Users\uik02458\Desktop\files\PythonProject\YOLO\test_vedio.mp4"
# =============================================

def test_video_base64():
    """测试 base64 视频输入"""
    
    # 检查文件是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        print("请修改 VIDEO_PATH 变量为你实际的视频文件路径")
        return False
    
    # 读取视频并转为 base64
    print(f"正在读取视频: {VIDEO_PATH}")
    file_size = os.path.getsize(VIDEO_PATH)
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    with open(VIDEO_PATH, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    
    print("正在发送请求...")
    
    try:
        response = client.chat.completions.create(
            model="Qwen3-VL-8B-Instruct-4bit",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请简单描述这个视频的内容"
                    }
                ]
            }],
            max_tokens=500
        )
        print("✅ 支持原生视频输入！")
        print(f"响应: {response.choices[0].message.content}")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 测试失败")
        print(f"错误信息: {error_msg}")
        
        # 分析错误类型
        if "video" in error_msg.lower() and "not supported" in error_msg.lower():
            print("\n结论: vLLM 当前配置不支持视频输入")
        elif "video_url" in error_msg.lower():
            print("\n结论: vLLM 不识别 video_url 类型")
        elif "400" in error_msg:
            print("\n结论: 请求格式可能不正确或不支持视频")
        
        return False


if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    
    print("\n[测试] Base64 编码方式:")
    print("-" * 40)
    test_video_base64()
    
    print("\n" + "=" * 60)
    print("测试完成")
