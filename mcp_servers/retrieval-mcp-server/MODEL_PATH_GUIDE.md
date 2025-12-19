# SentenceTransformer 模型路径配置指南

## 概述
EnhancedRetrievalMCPServer 支持多种方式指定 SentenceTransformer 模型的下载和缓存路径。

## 使用方法

### 方法1: 构造函数参数
```python
from enhanced_server import EnhancedRetrievalMCPServer

# 指定自定义模型缓存路径
server = EnhancedRetrievalMCPServer(
    data_path="/data/zhendong_data/cx/ReCall/data/amazon-electronics/processed/item_meta_5core.jsonl",
    model_cache_path="/path/to/your/models"
)
```

### 方法2: 环境变量
```bash
export SENTENCE_TRANSFORMERS_CACHE="/path/to/your/models"
python enhanced_server.py
```

### 方法3: 默认路径
如果不指定路径，模型将默认缓存到：
```
/data/agentic-rec/rec-mcp-bench/models/sentence_transformers
```

## 模型管理

### 1. 首次下载
首次运行时会自动下载 `all-MiniLM-L6-v2` 模型到指定路径。

### 2. 离线使用
下载完成后，模型文件会保存在本地，后续使用无需联网。

### 3. 模型文件结构
```
/path/to/models/
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── config_sentence_transformers.json
    ├── modules.json
    ├── pytorch_model.bin
    ├── sentence_bert_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

## 示例代码

### 基本使用
```python
import asyncio
from enhanced_server import EnhancedRetrievalMCPServer

async def main():
    # 使用自定义模型路径
    server = EnhancedRetrievalMCPServer(
        model_cache_path="/data/my_models"
    )
    
    # 测试语义检索
    result = await server.server.tools['semantic_retrieval'].function(
        "我需要一台适合游戏的笔记本电脑"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 环境变量方式
```python
import os
import asyncio
from enhanced_server import EnhancedRetrievalMCPServer

# 设置环境变量
os.environ['SENTENCE_TRANSFORMERS_CACHE'] = '/data/models/sentence_transformers'

async def main():
    # 不需要额外参数，会自动使用环境变量
    server = EnhancedRetrievalMCPServer()
    
    # 测试关键词检索
    result = await server.server.tools['keyword_retrieval'].function(
        "16GB RAM gaming laptop"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 故障排除

### 1. 模型下载失败
- 检查网络连接
- 确保有足够磁盘空间（模型约 80MB）
- 验证缓存目录权限

### 2. 模型加载失败
- 检查模型文件是否完整
- 尝试删除缓存目录重新下载
- 使用 `logger` 查看详细错误信息

### 3. 性能优化
- 将模型缓存到 SSD 磁盘提高加载速度
- 使用本地模型避免重复下载
- 考虑使用更小的模型如 `all-MiniLM-L12-v2`

## 相关文件
- `enhanced_server.py` - 主要服务器实现
- `example_model_path.py` - 自定义路径示例
- `test_enhanced_server.py` - 完整功能测试