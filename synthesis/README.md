# Task Synthesis

CUDA_VISIBLE_DEVICES=7 cd /data/agentic-rec/rec-mcp-bench && python synthesis/run_llm_retrieval_synthesis.py --processed_dir /data/zhendong_data/cx/ReCall/data/amazon-electronics/processed --max_tasks 50 --mode keyword


CUDA_VISIBLE_DEVICES=7 python3 synthesis/run_llm_retrieval_synthesis.py --processed_dir /data/zhendong_data/cx/ReCall/data/amazon-electronics/processed --max_tasks 50 --mode semantic


CUDA_VISIBLE_DEVICES=7 python3 synthesis/run_llm_retrieval_synthesis.py --processed_dir /data/zhendong_data/cx/ReCall/data/amazon-electronics/processed --max_tasks 1 --mode sql

CUDA_VISIBLE_DEVICES=7 python3 synthesis/run_llm_retrieval_synthesis.py --processed_dir /data/zhendong_data/cx/ReCall/data/amazon-electronics/processed --max_tasks 50 --mode missing_price

Generate benchmark tasks for MCP servers.
recommendation-mcp-suite/
├── retrieval-mcp-server/     # 检索工具
├── ranking-mcp-server/       # 排序工具  
├── knowledge-mcp-server/     # 知识获取工具
└── nlp-mcp-server/          # NLP 工具



retrieval-mcp-server (8个工具)
├── 🔍 检索工具 (3个)
│   ├── search_products_by_keywords    # 关键词搜索
│   ├── vector_similarity_search       # 语义相似搜索  
│   └── item_to_item_similarity       # 协同过滤推荐
├── 🗄️ 查询工具 (1个)
│   └── sql_query_products            # SQL自定义查询
├── 📊 元数据工具 (2个)
│   ├── get_product_categories        # 获取类别列表
│   └── get_product_brands           # 获取品牌列表

## Quick Start

### Generate Single-Server Tasks
```bash
nohup python synthesis/generate_benchmark_tasks.py \
  --mode single \
  --filter-problematic --tasks-per-combination 2 \
  --output benchmark_tasks_single_$(date +%Y%m%d)test.json \
  > task_generation_single_$(date +%Y%m%d)test.log 2>&1 &
```

### Generate Multi-Server Tasks (2 servers)
```bash
nohup python synthesis/generate_benchmark_tasks.py \
  --mode multi \
  --combinations-file synthesis/split_combinations/mcp_2server_combinations.json \
  --filter-problematic --tasks-per-combination 2 \
  --output benchmark_tasks_multi_2server_$(date +%Y%m%d)test.json \
  > task_generation_multi_2server_$(date +%Y%m%d)test.log 2>&1 &
```

### Generate Multi-Server Tasks (3 servers)
```bash
nohup python synthesis/generate_benchmark_tasks.py \
  --mode multi \
  --combinations-file synthesis/split_combinations/mcp_3server_combinations.json \
  --filter-problematic --tasks-per-combination 2 \
  --output benchmark_tasks_multi_3server_$(date +%Y%m%d)test.json \
  > task_generation_multi_3server_$(date +%Y%m%d)test.log 2>&1 &
```

## Files

- `task_synthesis.py` - Core task generation and fuzzy conversion
- `benchmark_generator.py` - Unified task generator for single/multi-server
- `generate_benchmark_tasks.py` - CLI script for batch generation
- `split_combinations/` - Pre-defined server combinations for multi-server tasks

## Output

Tasks are saved to `tasks/` directory in JSON format.