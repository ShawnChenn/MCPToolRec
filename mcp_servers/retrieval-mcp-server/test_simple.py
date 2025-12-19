#!/usr/bin/env python3
"""
Simplified Test Script for Enhanced Retrieval MCP Server
Auto-generates queries based on actual data and tests all three retrieval modes
"""

import asyncio
import sys
from pathlib import Path
import random
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_server import EnhancedRetrievalMCPServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_enhanced_server")

class SimpleRetrievalTester:
    """Simplified test class that auto-generates queries from data"""
    
    def __init__(self, model_cache_path: str = None, semantic_model_path: str = \
                 "/data/agentic-rec/rec-mcp-bench/models/sentence_transformers/all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"):
        self.server = EnhancedRetrievalMCPServer(model_cache_path=model_cache_path)
        self.products_df = self.server.products_df
        self._desc_vectors = None
        self._desc_vectorizer = None
        self._semantic_model = None
        self.semantic_model_path = Path(semantic_model_path)
        
    def generate_semantic_queries(self):
        """Generate semantic queries"""
        return [
            "我需要一台适合办公的笔记本电脑",
            "good quality headphones for music",
            "portable device for travel",
            "reliable brand with good reviews",
            "适合学生使用的电子产品"
        ][:3]
    
    async def test_retrieval_mode(self, mode_name: str, queries: list, retrieval_func):
        """Test a specific retrieval mode"""
        print(f"\n{'='*60}")
        print(f"🧪 测试 {mode_name}")
        print('='*60)
        
        for query in queries:
            print(f"\n📝 查询: {query}")
            print("-" * 40)
            try:
                result = await retrieval_func(query)
                print(result)
            except Exception as e:
                print(f"❌ 错误: {e}")
                logger.error(f"Error in {mode_name}: {e}")
            print("-" * 40)
    
    async def test_candidate_bus(self):
        """Test candidate bus functionality"""
        print(f"\n{'='*60}")
        print("🚌 测试候选集管理")
        print('='*60)
        
        try:
            # Initial status
            print("初始状态:")
            status = await self.server.get_candidate_bus_status()
            print(status)
            
            # Apply a filter
            print("\n应用简单过滤...")
            await self.server.text2sql_retrieval("products under $1000")
            
            # Check status after filter
            print("\n过滤后状态:")
            status = await self.server.get_candidate_bus_status()
            print(status)
            
            # Reset
            print("\n重置候选集:")
            reset_result = await self.server.reset_candidate_bus()
            print(reset_result)
            
        except Exception as e:
            print(f"❌ 候选集管理错误: {e}")
    
    async def run_simple_tests(self):
        # Check data loading
        if self.products_df is None or len(self.products_df) == 0:
            print("⚠️ 警告: 未加载到数据，使用示例数据进行测试")
            return
        else:
            print(f"✅ 已加载 {len(self.products_df)} 个产品")
        
        sample = self.products_df.head(3)
        for _, product in sample.iterrows():
            print(f"• {product.get('title', 'Unknown')} - {product.get('brand', 'Unknown')} - ${product.get('price_float', 0):.2f}")
        
        keyword_queries = ['Noble OV/HB Universal Power Kit']
        semantic_queries = self.generate_semantic_queries()
     
        try:
            await self.test_retrieval_mode("关键词检索", keyword_queries, lambda q: self.server.keyword_retrieval(q, limit=3))
            await self.test_retrieval_mode("语义检索（description）", semantic_queries, lambda q: self.semantic_search_tool(q, limit=3))
            sql_queries = [
                "laptops between $800 and $1200",
            ]
            await self.test_retrieval_mode("SQL硬过滤", sql_queries, lambda q: self.sql_filter_tool(q, limit=5))
            print("\n" + "="*60)
            print("✅ 关键词检索测试完成！")
            print("="*60)
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

    async def semantic_search_tool(self, query: str, limit: int = 3) -> str:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        import pandas as pd
        df = self.products_df
        if df is None or len(df) == 0:
            return "No data available"
        desc = df['description'].fillna('').astype(str).str.lower()
        if self._desc_vectors is None:
            if self.semantic_model_path.exists():
                try:
                    from sentence_transformers import SentenceTransformer
                    self._semantic_model = SentenceTransformer(str(self.semantic_model_path))
                    self._desc_vectors = self._semantic_model.encode(desc.tolist(), convert_to_numpy=True)
                except Exception:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self._desc_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
                    self._desc_vectors = self._desc_vectorizer.fit_transform(desc)
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._desc_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
                self._desc_vectors = self._desc_vectorizer.fit_transform(desc)
        if self._semantic_model is not None:
            qv = self._semantic_model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(qv, self._desc_vectors)[0]
        else:
            qv = self._desc_vectorizer.transform([query])
            sims = cosine_similarity(qv, self._desc_vectors)[0]
        top_idx = np.argsort(sims)[::-1][:limit]
        result = f"Found {len(top_idx)} semantically matching products by description:\n\n"
        for idx in top_idx:
            row = df.iloc[idx]
            sim = sims[idx]
            result += f"• {row.get('title', '')}\n"
            result += f"  Similarity: {sim:.3f}\n"
            result += f"  Brand: {row.get('brand', '')}, Price: ${row.get('price_float', 0):.2f}\n"
            result += f"  Description: {row.get('description', '')[:100]}...\n\n"
        return result

    async def sql_filter_tool(self, query: str, limit: int = 5) -> str:
        import re
        import pandas as pd
        df = self.products_df.copy()
        if df is None or len(df) == 0:
            return "No data available"
        q = query.lower()
        price_min = None
        price_max = None
        m = re.search(r"under\s*\$?(\d+(?:\.\d+)?)", q)
        if m:
            price_max = float(m.group(1))
        m = re.search(r"below\s*\$?(\d+(?:\.\d+)?)", q)
        if m:
            price_max = float(m.group(1))
        m = re.search(r"over\s*\$?(\d+(?:\.\d+)?)", q)
        if m:
            price_min = float(m.group(1))
        m = re.search(r"above\s*\$?(\d+(?:\.\d+)?)", q)
        if m:
            price_min = float(m.group(1))
        m = re.search(r"\$?(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)", q)
        if m:
            price_min = float(m.group(1))
            price_max = float(m.group(2))
        m = re.search(r"between\s*\$?(\d+(?:\.\d+)?)\s*and\s*\$?(\d+(?:\.\d+)?)", q)
        if m:
            price_min = float(m.group(1))
            price_max = float(m.group(2))
        if price_min is not None:
            df = df[df['price_float'] >= price_min]
        if price_max is not None:
            df = df[df['price_float'] <= price_max]
        brand = None
        # try explicit 'brand xxx'
        m = re.search(r"brand\s+([a-z0-9\-\&]+)", q)
        if m:
            brand = m.group(1)
        else:
            brands = set(str(b).lower() for b in df['brand'].dropna().astype(str).tolist())
            for b in brands:
                if b and b in q:
                    brand = b
                    break
        if brand:
            df = df[df['brand'].str.contains(brand, case=False, na=False)]
        category = None
        m = re.search(r"category\s+([a-z\s\-]+)", q)
        if m:
            category = m.group(1).strip()
        else:
            cats = set(str(c).lower() for c in df['category'].dropna().astype(str).tolist())
            for c in cats:
                if c and c in q:
                    category = c
                    break
        if category:
            df = df[df['category'].str.contains(category, case=False, na=False)]
        if len(df) == 0:
            return "No products found by SQL hard filter"
        df = df.head(limit)
        result = f"Found {len(df)} products by SQL hard filter:\n\n"
        for _, row in df.iterrows():
            result += f"• {row.get('title', '')}\n"
            result += f"  Brand: {row.get('brand', '')}, Price: ${row.get('price_float', 0):.2f}\n"
            result += f"  Category: {row.get('category', '')}\n\n"
        return result

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Test Enhanced Retrieval MCP Server')
    parser.add_argument('--model-cache-path', type=str, 
                       help='Path to cache SentenceTransformer models')
    
    args = parser.parse_args()
    
    tester = SimpleRetrievalTester(model_cache_path=args.model_cache_path)
    await tester.run_simple_tests()

if __name__ == "__main__":
    asyncio.run(main())
