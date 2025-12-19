#!/usr/bin/env python3
"""
Example: Using custom model cache path with EnhancedRetrievalMCPServer
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_server import EnhancedRetrievalMCPServer

async def test_custom_model_path():
    """Test the server with custom model cache path"""
    
    # 方法1: 使用构造函数指定模型缓存路径
    print("🚀 测试自定义模型缓存路径...")
    
    # 指定自定义模型缓存路径
    custom_model_path = "/data/agentic-rec/rec-mcp-bench/models/my_sentence_models"
    
    # 创建服务器实例，指定模型缓存路径
    server = EnhancedRetrievalMCPServer(
        data_path="/data/zhendong_data/cx/ReCall/data/amazon-electronics/processed/item_meta_5core.jsonl",
        model_cache_path=custom_model_path
    )
    
    print(f"✅ 服务器初始化完成，模型缓存路径: {custom_model_path}")
    
    # 测试语义检索功能
    print("\n🧠 测试语义检索功能...")
    try:
        result = await server.server.tools['semantic_retrieval'].function(
            "我需要一台适合编程的笔记本电脑"
        )
        print(f"语义检索结果: {result}")
    except Exception as e:
        print(f"❌ 语义检索错误: {e}")
    
    print("\n✅ 自定义模型路径测试完成！")

async def test_env_variable():
    """Test using environment variable for model path"""
    
    print("🌍 使用环境变量指定模型路径...")
    
    # 设置环境变量
    import os
    os.environ['SENTENCE_TRANSFORMERS_CACHE'] = "/data/agentic-rec/rec-mcp-bench/models/env_models"
    
    # 创建服务器实例（不指定model_cache_path，会使用环境变量）
    server = EnhancedRetrievalMCPServer()
    
    print(f"✅ 使用环境变量模型缓存路径: {os.environ['SENTENCE_TRANSFORMERS_CACHE']}")
    
    # 测试关键词检索功能
    print("\n🔍 测试关键词检索功能...")
    try:
        result = await server.server.tools['keyword_retrieval'].function(
            "16GB RAM i7 processor"
        )
        print(f"关键词检索结果: {result}")
    except Exception as e:
        print(f"❌ 关键词检索错误: {e}")

async def main():
    """Main function"""
    print("=" * 60)
    print("🎯 自定义SentenceTransformer模型路径测试")
    print("=" * 60)
    
    # 测试构造函数指定路径
    await test_custom_model_path()
    
    print("\n" + "=" * 60)
    
    # 测试环境变量指定路径
    await test_env_variable()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())