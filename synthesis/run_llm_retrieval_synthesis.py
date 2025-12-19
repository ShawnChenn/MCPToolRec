#!/usr/bin/env python3
"""
Single MCP Tool Task Synthesis
Generates tasks for individual MCP tools based on real product data
"""

import asyncio
import json
import sys
import random
import argparse
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from .llm_retrieval_synthesizer import LLMRetrievalSynthesizer
except ImportError:
    from llm_retrieval_synthesizer import LLMRetrievalSynthesizer


def _load_meta(processed_dir: str):
    meta_path = Path(processed_dir) / "item_meta_5core.jsonl"
    asin_to_meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            a = obj.get("asin")
            if a:
                asin_to_meta[a] = obj
    return asin_to_meta

def _load_targets(processed_dir: str):
    seqs_path = Path(processed_dir) / "user_sequences_5core.jsonl"
    targets = []
    with open(seqs_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            items = o.get("items", [])
            if not items:
                continue
            last = items[-1]
            asin = last.get("asin")
            uid = o.get("user_id")
            if uid and asin:
                targets.append((uid, asin))
    return targets

def _sample_negatives(asin_to_meta, exclude_asin: str, k: int = 10):
    pool = [a for a in asin_to_meta.keys() if a != exclude_asin]
    if len(pool) <= k:
        sample = pool
    else:
        sample = random.sample(pool, k)
    items = []
    for a in sample:
        p = asin_to_meta.get(a, {})
        items.append({
            "asin": a,
            "title": (p.get("title") or "").strip(),
            "brand": (p.get("brand") or "").strip(),
            "category": (p.get("category") or ["Electronics"])[0] if isinstance(p.get("category"), list) else (p.get("category") or "Electronics")
        })
    return items

def _build_ground_truth(task):
    calls = []
    seq = task.get('expected_tool_sequence') or []
    for step in seq:
        tool = step.get('tool')
        params = step.get('parameters') or {}
        cmd = f"call_tool('{tool}', {json.dumps(params, ensure_ascii=False)})"
        calls.append({"tool": tool, "parameters": params, "command": cmd})
    return calls

def _normalize_text_value(x):
    try:
        import pandas as pd
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
    except Exception:
        if x is None:
            return ""
    if isinstance(x, list):
        return " ".join(str(i) for i in x if i is not None)
    if isinstance(x, dict):
        return " ".join(str(v) for k, v in x.items() if v is not None)
    return str(x)

def _extract_price_value(x):
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x)
        # Extract first numeric token
        import re
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
        if m:
            return float(m.group(0))
    except Exception:
        return None
    return None

async def generate_single_tool_tasks(processed_dir: str, max_tasks: int = 50):
    synthesizer = LLMRetrievalSynthesizer(data_path=str(Path(processed_dir)))
    asin_to_meta = _load_meta(processed_dir)
    targets = _load_targets(processed_dir)
    all_generated_tasks = []
    for i, (uid, asin) in enumerate(targets, 1):
        pd = asin_to_meta.get(asin, {})
        title = (pd.get("title") or "").strip()
        brand = (pd.get("brand") or "").strip()
        category = pd.get("category", ["Electronics"])
        category = category[0] if isinstance(category, list) and category else (category or "Electronics")
        price = (pd.get("price") or "").strip()
        product_context = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,
            "target_item": {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "price": price
            }
        }
        try:
            task = synthesizer.generate_synthetic_task(
                scenario="keyword_search",
                force_tools=["bm25_keyword_search"],
                product_context=product_context
            )
            if task:
                task["ground_truth_calls"] = _build_ground_truth(task)
                all_generated_tasks.append(task)
                print(f"✅ {i}: {uid} → {asin}")
                if len(all_generated_tasks) >= max_tasks:
                    break
        except Exception as e:
            print(f"❌ {i}: {uid} → {asin}: {e}")
            continue
    return all_generated_tasks

async def generate_semantic_retrieval_tasks(processed_dir: str, max_tasks: int = 50):
    synthesizer = LLMRetrievalSynthesizer(data_path=str(Path(processed_dir)))
    asin_to_meta = _load_meta(processed_dir)
    targets = _load_targets(processed_dir)
    all_generated_tasks = []
    for i, (uid, asin) in enumerate(targets, 1):
        pd = asin_to_meta.get(asin, {})
        title = (pd.get("title") or "").strip()
        brand = (pd.get("brand") or "").strip()
        category = pd.get("category", ["Electronics"])
        category = category[0] if isinstance(category, list) and category else (category or "Electronics")
        description = _normalize_text_value(pd.get("description")).strip()
        prompt_description = description if description else (title or "").strip()
        price = (pd.get("price") or "").strip()
        product_context = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "description": description,
            "prompt_description": prompt_description,
            "price": price,
            "target_item": {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "description": description,
                "price": price
            }
        }
        try:
            task = synthesizer.generate_synthetic_task(
                scenario="semantic_retrieval",
                force_tools=["sbert_vector_search"],
                product_context=product_context
            )
            if task:
                task["ground_truth_calls"] = _build_ground_truth(task)
                all_generated_tasks.append(task)
                print(f"✅ {i}: {uid} → {asin}")
                if len(all_generated_tasks) >= max_tasks:
                    break
        except Exception as e:
            print(f"❌ {i}: {uid} → {asin}: {e}")
            continue
    return all_generated_tasks

async def generate_sql_hard_filter_tasks(processed_dir: str, max_tasks: int = 50):
    synthesizer = LLMRetrievalSynthesizer(data_path=str(Path(processed_dir)))
    asin_to_meta = _load_meta(processed_dir)
    targets = _load_targets(processed_dir)
    all_generated_tasks = []
    for i, (uid, asin) in enumerate(targets, 1):
        pd = asin_to_meta.get(asin, {})
        title = (pd.get("title") or "").strip()
        brand = (pd.get("brand") or "").strip()
        category = pd.get("category", ["Electronics"])
        category = category[0] if isinstance(category, list) and category else (category or "Electronics")
        price = (pd.get("price") or "").strip()
        product_context = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,
            "target_item": {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category
            }
        }
        try:
            task = synthesizer.generate_synthetic_task(
                scenario="sql_hard_filter",
                force_tools=["sql_query_products"],
                product_context=product_context
            )
            if task:
                task["ground_truth_calls"] = _build_ground_truth(task)
                all_generated_tasks.append(task)
                print(f"✅ {i}: {uid} → {asin}")
                if len(all_generated_tasks) >= max_tasks:
                    break
        except Exception as e:
            print(f"❌ {i}: {uid} → {asin}: {e}")
            continue
    return all_generated_tasks

async def generate_missing_price_query_tasks(processed_dir: str, max_tasks: int = 50):
    """Generate tasks for items with missing price values to query their prices"""
    synthesizer = LLMRetrievalSynthesizer(data_path=str(Path(processed_dir)))
    asin_to_meta = _load_meta(processed_dir)
    targets = _load_targets(processed_dir)
    all_generated_tasks = []
    missing_price_count = 0
    
    for i, (uid, asin) in enumerate(targets, 1):
        pd = asin_to_meta.get(asin, {})
        
        # Check if price is missing or empty
        price = pd.get("price")
        if price and str(price).strip():
            continue  # Skip items that already have price
            
        missing_price_count += 1
        
        title = (pd.get("title") or "").strip()
        brand = (pd.get("brand") or "").strip()
        category = pd.get("category", ["Electronics"])
        category = category[0] if isinstance(category, list) and category else (category or "Electronics")
        description = _normalize_text_value(pd.get("description")).strip()
        
        # Create product context for missing price scenario
        product_context = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "description": description,
            "price_missing": True,  # Flag to indicate missing price
            "target_item": {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "description": description
            }
        }
        
        try:
            # Generate task specifically for querying missing price
            # Use keyword search to find price information
            task = synthesizer.generate_synthetic_task(
                scenario="missing_price_query",
                force_tools=["bm25_keyword_search"],
                product_context=product_context
            )
            if task:
                task["ground_truth_calls"] = _build_ground_truth(task)
                all_generated_tasks.append(task)
                print(f"✅ {missing_price_count}: {uid} → {asin} (missing price)")
                if len(all_generated_tasks) >= max_tasks:
                    break
        except Exception as e:
            print(f"❌ {missing_price_count}: {uid} → {asin}: {e}")
            continue
    
    print(f"Found {missing_price_count} items with missing prices, generated {len(all_generated_tasks)} tasks")
    return all_generated_tasks

async def generate_review_analysis_tasks(processed_dir: str, max_tasks: int = 50, variant: str = "sentiment"):
    """Generate tasks focused on review sentiment analysis or review summary"""
    synthesizer = LLMRetrievalSynthesizer(data_path=str(Path(processed_dir)))
    asin_to_meta = _load_meta(processed_dir)
    targets = _load_targets(processed_dir)
    all_generated_tasks = []

    scenario = "review_sentiment" if variant == "sentiment" else "review_summary"
    force_tools = ["sbert_vector_search"] if variant == "sentiment" else ["bm25_keyword_search"]

    for i, (uid, asin) in enumerate(targets, 1):
        pd = asin_to_meta.get(asin, {})
        title = (pd.get("title") or "").strip()
        brand = (pd.get("brand") or "").strip()
        category = pd.get("category", ["Electronics"])
        category = category[0] if isinstance(category, list) and category else (category or "Electronics")
        price = _extract_price_value(pd.get("price"))

        raw_reviews = pd.get("review") or []
        if isinstance(raw_reviews, list):
            reviews = [str(r).strip() for r in raw_reviews if str(r).strip()][:3]
        else:
            reviews = [str(raw_reviews).strip()][:1] if str(raw_reviews).strip() else []

        product_context = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price or "",
            "reviews": reviews,
            "target_item": {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "price": price or "",
            }
        }

        try:
            task = synthesizer.generate_synthetic_task(
                scenario=scenario,
                force_tools=force_tools,
                product_context=product_context
            )
            if task:
                task["ground_truth_calls"] = _build_ground_truth(task)
                all_generated_tasks.append(task)
                print(f"✅ {i}: {uid} → {asin} ({scenario})")
                if len(all_generated_tasks) >= max_tasks:
                    break
        except Exception as e:
            print(f"❌ {i}: {uid} → {asin}: {e}")
            continue

    return all_generated_tasks

def save_single_tool_tasks(tasks, filename="single_tool_tasks.json"):
    """Save simplified single-tool tasks to file (query + target + call chain)"""
    output_file = Path(__file__).parent / filename
    simple_tasks = []
    for task in tasks:
        pc = task.get('product_context', {})
        target = pc.get('target_item') or {
            'asin': pc.get('asin'),
            'title': pc.get('title'),
            'brand': pc.get('brand'),
            'category': pc.get('category'),
            'description': _normalize_text_value(pc.get('description'))
        }
        simple_tasks.append({
            'user_query': task.get('user_query', ''),
            'target_item': {
                'asin': target.get('asin'),
                'title': target.get('title'),
                'brand': target.get('brand'),
                'category': target.get('category'),
                'description': _normalize_text_value(target.get('description')),
                'price': _extract_price_value(target.get('price') or pc.get('price'))
            },
            'ground_truth_calls': task.get('ground_truth_calls', [])
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simple_tasks, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Tasks saved to: {output_file} (simplified)")
    return str(output_file)

def analyze_single_tool_tasks(tasks):
    """Analyze generated single-tool tasks"""
    
    print(f"\n📊 Single-Tool Task Analysis:")
    print("-" * 30)
    
    # Tool distribution
    tool_counts = {}
    for task in tasks:
        tool = task['required_tools'][0]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    print(f"Tool Distribution:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count} tasks")
    
    # Product categories covered
    categories = set()
    brands = set()
    for task in tasks:
        if task.get('product_context'):
            pc = task['product_context']
            if pc.get('category'):
                categories.add(pc['category'])
            if pc.get('brand'):
                brands.add(pc['brand'])
    
    print(f"\nData Coverage:")
    print(f"  Categories: {len(categories)} unique")
    print(f"  Brands: {len(brands)} unique")
    
    # Sample queries by tool
    print(f"\nSample Queries by Tool:")
    for tool in sorted(tool_counts.keys()):
        tool_tasks = [t for t in tasks if t['required_tools'][0] == tool]
        if tool_tasks:
            sample_query = tool_tasks[0]['user_query'][:80] + "..."
            print(f"  {tool}:")
            print(f"    \"{sample_query}\"")

async def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--processed_dir", type=str, required=True)
        parser.add_argument("--max_tasks", type=int, default=50)
        parser.add_argument("--mode", type=str, choices=["keyword", "semantic", "sql", "missing_price", "review_sentiment", "review_summary"], default="keyword")
        parser.add_argument("--gpu", type=str, help="CUDA device id(s), e.g., '0' or '0,1'")
        args = parser.parse_args()
        if args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("🚀 Starting Single MCP Tool Task Generation...")
        if args.mode == "keyword":
            tasks = await generate_single_tool_tasks(args.processed_dir, max_tasks=args.max_tasks)
        elif args.mode == "semantic":
            tasks = await generate_semantic_retrieval_tasks(args.processed_dir, max_tasks=args.max_tasks)
        elif args.mode == "sql":
            tasks = await generate_sql_hard_filter_tasks(args.processed_dir, max_tasks=args.max_tasks)
        elif args.mode == "missing_price":
            tasks = await generate_missing_price_query_tasks(args.processed_dir, max_tasks=args.max_tasks)
        elif args.mode == "review_sentiment":
            tasks = await generate_review_analysis_tasks(args.processed_dir, max_tasks=args.max_tasks, variant="sentiment")
        else:
            tasks = await generate_review_analysis_tasks(args.processed_dir, max_tasks=args.max_tasks, variant="summary")
        if not tasks:
            print("❌ No tasks generated!")
            return False
        analyze_single_tool_tasks(tasks)
        output_file = save_single_tool_tasks(tasks)
        print(f"\n" + "=" * 50)
        print("✨ Single-Tool Task Generation Complete!")
        print(f"📈 Total Tasks Generated: {len(tasks)}")
        print(f"📁 Output File: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error during single-tool task generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
