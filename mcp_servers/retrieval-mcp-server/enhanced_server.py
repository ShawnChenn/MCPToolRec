#!/usr/bin/env python3
"""
Enhanced Retrieval MCP Server with Three Retrieval Modes
Supports Text-to-SQL (Hard Filter), Semantic Retrieval, and Keyword Retrieval
"""

import asyncio
import json
import logging
import re
import os
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Semantic search will use TF-IDF fallback.")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Semantic search will use TF-IDF fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# MCP imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-retrieval-mcp-server")

# Limit for initial candidate bus population
CANDIDATE_LIMIT = 1000

class CandidateBus:
    """Candidate item storage and management"""
    
    def __init__(self):
        self.item_ids: Set[str] = set()
        self.metadata: Dict[str, Dict] = {}
        self.last_updated = datetime.now()
    
    def update_candidates(self, item_ids: Set[str], metadata: Optional[Dict] = None):
        """Update the candidate bus with new item IDs"""
        self.item_ids = item_ids
        if metadata:
            self.metadata.update(metadata)
        self.last_updated = datetime.now()
        logger.info(f"Candidate bus updated with {len(item_ids)} items")
    
    def get_candidates(self) -> Set[str]:
        """Get current candidate item IDs"""
        return self.item_ids.copy()
    
    def clear(self):
        """Clear all candidates"""
        self.item_ids.clear()
        self.metadata.clear()
        self.last_updated = datetime.now()

class Text2SQLEngine:
    """Text-to-SQL engine for hard filtering"""
    
    def __init__(self):
        self.price_patterns = [
            r'(?:price|cost|价钱|价格)\s*(?:under|below|less than|<)\s*\$?(\d+(?:\.\d+)?)',
            r'(?:price|cost|价钱|价格)\s*(?:above|over|more than|>)\s*\$?(\d+(?:\.\d+)?)',
            r'(?:price|cost|价钱|价格)\s*(?:between|from)\s*\$?(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*\$?(\d+(?:\.\d+)?)',
            r'\$?(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*\$?(\d+(?:\.\d+)?)',
        ]
        
        self.brand_patterns = [
            r'(?:brand|manufacturer|品牌)\s+(?:is\s+)?(\w+)',
            r'(索尼|Sony|苹果|Apple|三星|Samsung|戴尔|Dell|惠普|HP|联想|Lenovo)',
            r'(\w+)\s*(?:品牌|brand)',  # Match "Co-link品牌" or "Co-link brand"
            r'(?:by|from)\s+(\w+)',  # Match "products by Co-link"
            r'(?:给我找一款|推荐|show me)\s*(\w+)\s*(?:品牌|brand)',  # Chinese: "给我找一款Accuon品牌的电子产品"
            r'(\w+)\s*(?:的|品牌)',  # Match "Accuon的" or "Accuon品牌"
        ]
        
        self.category_patterns = [
            r'(?:category|type|类别|分类)\s+(?:is\s+)?(\w+)',
            r'(耳机|headphones|earphones|手机|smartphone|电脑|laptop|笔记本)',
            r'(\w+)\s*(?:类别|category)',  # Match "Computer Screws类别"
            r'(laptops?|headphones?|computers?|adapters?|chargers?|cameras?)',  # Common product types
        ]
        
        self.rating_patterns = [
            r'(?:rating|rated|评分)\s*(?:above|over|more than|>)\s*(\d+(?:\.\d+)?)',
            r'(?:rating|rated|评分)\s*(?:below|under|less than|<)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:stars?|星)',
        ]
    
    def parse_constraints(self, query: str) -> Dict[str, Any]:
        """Parse natural language query and extract structured constraints"""
        constraints = {}
        query_lower = query.lower()
        
        # Price constraints
        for pattern in self.price_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                if len(matches[0]) == 2:  # Range
                    constraints['price_min'] = float(matches[0][0])
                    constraints['price_max'] = float(matches[0][1])
                else:  # Single value
                    if 'under' in query_lower or 'below' in query_lower or 'less than' in query_lower:
                        constraints['price_max'] = float(matches[0])
                    elif 'above' in query_lower or 'over' in query_lower or 'more than' in query_lower:
                        constraints['price_min'] = float(matches[0])
                break
        
        # Brand constraints
        for pattern in self.brand_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                constraints['brand'] = matches[0].title()
                break
        
        # Category constraints
        for pattern in self.category_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                constraints['category'] = matches[0].title()
                break
        
        # Rating constraints
        for pattern in self.rating_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                rating = float(matches[0])
                if 'above' in query_lower or 'over' in query_lower or rating >= 4:
                    constraints['rating_min'] = rating
                elif 'below' in query_lower or 'under' in query_lower:
                    constraints['rating_max'] = rating
                else:
                    constraints['rating_min'] = rating
                break
        
        logger.info(f"Extracted constraints: {constraints}")
        return constraints

class EnhancedRetrievalMCPServer:
    """Enhanced Retrieval MCP Server with three retrieval modes"""
    
    def __init__(self, data_path: str = "/data/zhendong_data/cx/ReCall/data/amazon-electronics/processed/item_meta_5core.jsonl", model_cache_path: Optional[str] = None):
        self.server = Server("enhanced-retrieval-mcp-server")
        self.data_path = Path(data_path)
        self.model_cache_path = Path(model_cache_path) if model_cache_path else None
        self.candidate_bus = CandidateBus()
        self.text2sql_engine = Text2SQLEngine()
        
        # Data storage
        self.products_df = None
        self.products_subset_df = None
        self.product_vectors = None
        self.vectorizer = None
        self.semantic_model = None
        self.title_vectorizer = None
        self.full_title_vectors = None
        
        # Initialize components
        self._load_data()
        self._setup_vector_models()
        self._setup_tools()

    def _load_data(self):
        """Load product data from JSONL file"""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            # Create sample data for testing
            self._create_sample_data()
            return
        
        products = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        product = json.loads(line.strip())
                        # Ensure required fields
                        product.setdefault('asin', f'UNKNOWN_{line_num}')
                        product.setdefault('title', 'Unknown Product')
                        product.setdefault('price', '0.0')
                        product.setdefault('brand', 'Unknown Brand')
                        product.setdefault('category', 'Unknown Category')
                        product.setdefault('description', '')
                        product.setdefault('rating', 0.0)
                        
                        # Parse price to float
                        try:
                            price_str = str(product['price']).replace('$', '').replace(',', '')
                            product['price_float'] = float(price_str) if price_str else 0.0
                        except (ValueError, TypeError):
                            product['price_float'] = 0.0
                        
                        products.append(product)
                        
                        if line_num % 1000 == 0:
                            logger.info(f"Loaded {line_num} products")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._create_sample_data()
            return
        
        self.products_df = pd.DataFrame(products)
        logger.info(f"Successfully loaded {len(self.products_df)} products")
        
        # Initialize candidate bus with limited items (first N)
        limited_ids = set(self.products_df['asin'].head(CANDIDATE_LIMIT).tolist())
        self.candidate_bus.update_candidates(limited_ids)
        
        # Prepare subset dataframe aligned with candidate bus
        self.products_subset_df = (
            self.products_df[self.products_df['asin'].isin(limited_ids)].reset_index(drop=True)
        )
    
    def _create_sample_data(self):
        """Create sample data for testing when real data is not available"""
        logger.info("Creating sample data for testing")
        
        sample_products = [
            {
                'asin': 'B08PZHYWJS',
                'title': 'Sony WH-1000XM4 Wireless Headphones',
                'price': '$299.99',
                'price_float': 299.99,
                'brand': 'Sony',
                'category': 'Headphones',
                'description': 'Industry-leading noise canceling with Dual Noise Sensor technology',
                'rating': 4.5
            },
            {
                'asin': 'B08N5WRWNW',
                'title': 'Apple AirPods Pro',
                'price': '$249.00',
                'price_float': 249.00,
                'brand': 'Apple',
                'category': 'Headphones',
                'description': 'Active noise cancellation for immersive sound',
                'rating': 4.4
            },
            {
                'asin': 'B08N5M7S6K',
                'title': 'Dell XPS 13 Laptop',
                'price': '$999.99',
                'price_float': 999.99,
                'brand': 'Dell',
                'category': 'Laptops',
                'description': '13.4-inch FHD laptop with Intel Core i7 processor',
                'rating': 4.3
            }
        ]
        
        self.products_df = pd.DataFrame(sample_products)
        all_item_ids = set(self.products_df['asin'].tolist())
        self.candidate_bus.update_candidates(all_item_ids)

    def _normalize_text_value(self, x):
        try:
            import pandas as pd  # local import to avoid global state issues
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ''
        except Exception:
            if x is None:
                return ''
        if isinstance(x, list):
            return ' '.join(str(i) for i in x if i is not None)
        if isinstance(x, dict):
            return ' '.join(f"{k}:{v}" for k, v in x.items() if v is not None)
        return str(x)

    def _setup_vector_models(self):
        logger.info("Setting up vector models")
        title_series = self.products_df['title'].apply(self._normalize_text_value).str.lower()
        self.title_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.full_title_vectors = self.title_vectorizer.fit_transform(title_series)
        logger.info("TF-IDF vectorizer fitted successfully")
 
    
    def _setup_tools(self):
        """Setup MCP tools with list/call pattern"""

        async def _text2sql_retrieval_impl(query: str) -> str:
            try:
                constraints = self.text2sql_engine.parse_constraints(query)
                if not constraints:
                    return "No structured constraints found in query. Please specify price, brand, rating, or category."
                filtered_df = self.products_df.copy()
                if 'price_min' in constraints:
                    filtered_df = filtered_df[filtered_df['price_float'] >= constraints['price_min']]
                if 'price_max' in constraints:
                    filtered_df = filtered_df[filtered_df['price_float'] <= constraints['price_max']]
                if 'brand' in constraints:
                    filtered_df = filtered_df[filtered_df['brand'].str.contains(constraints['brand'], case=False, na=False)]
                if 'category' in constraints:
                    filtered_df = filtered_df[filtered_df['category'].str.contains(constraints['category'], case=False, na=False)]
                if 'rating_min' in constraints:
                    filtered_df = filtered_df[filtered_df['rating'] >= constraints['rating_min']]
                if 'rating_max' in constraints:
                    filtered_df = filtered_df[filtered_df['rating'] <= constraints['rating_max']]
                if len(filtered_df) == 0:
                    return f"No products found matching constraints: {constraints}"
                limited_ids = set(filtered_df['asin'].head(CANDIDATE_LIMIT).tolist())
                # Only update candidate bus if we have matches; keep base subset intact
                self.candidate_bus.update_candidates(limited_ids, {'mode': 'text2sql', 'constraints': constraints})
                result = f"Found {len(filtered_df)} products matching constraints:\n"
                result += f"Constraints: {constraints}\n\n"
                for _, product in filtered_df.head(10).iterrows():
                    result += f"• {product['title']}\n"
                    result += f"  Brand: {product['brand']}, Price: ${product['price_float']:.2f}, Rating: {product['rating']}\n"
                    result += f"  Category: {product['category']}\n\n"
                if len(filtered_df) > 10:
                    result += f"... and {len(filtered_df) - 10} more products\n"
                return result
            except Exception as e:
                logger.error(f"Error in text2sql_retrieval: {e}")
                return f"Error processing query: {str(e)}"

        async def _semantic_retrieval_impl(query: str, limit: int = 10) -> str:
            try:
                if self.product_vectors is None:
                    return "Vector model not initialized. Please check data loading."
                if self.semantic_model:
                    query_vector = self.semantic_model.encode([query], convert_to_numpy=True)
                else:
                    query_vector = self.vectorizer.transform([query]).toarray()
                similarities = cosine_similarity(query_vector, self.product_vectors)[0]
                top_indices = np.argsort(similarities)[::-1][:limit]
                candidate_ids = self.candidate_bus.get_candidates()
                if candidate_ids:
                    valid_indices = [
                        i for i in top_indices 
                        if i < len(self.products_subset_df) and self.products_subset_df.iloc[i]['asin'] in candidate_ids
                    ]
                else:
                    valid_indices = top_indices.tolist()
                # Ensure indices are within bounds
                valid_indices = [i for i in valid_indices if i < len(self.products_subset_df)]
                if not valid_indices:
                    return "No products found matching semantic criteria in current candidate set."
                result = f"Found {len(valid_indices)} semantically similar products:\n\n"
                for idx in valid_indices:
                    product = self.products_subset_df.iloc[idx]
                    similarity = similarities[idx]
                    result += f"• {product['title']}\n"
                    result += f"  Similarity: {similarity:.3f}\n"
                    result += f"  Brand: {product['brand']}, Price: ${product['price_float']:.2f}\n"
                    result += f"  Description: {product['description'][:100]}...\n\n"
                semantic_ids = {self.products_subset_df.iloc[i]['asin'] for i in valid_indices}
                # Update candidate bus only; keep base subset intact to avoid vector misalignment
                self.candidate_bus.update_candidates(semantic_ids, {'mode': 'semantic', 'query': query})
                return result
            except Exception as e:
                logger.error(f"Error in semantic_retrieval: {e}")
                return f"Error in semantic retrieval: {str(e)}"

        async def _keyword_retrieval_impl(query: str, limit: int = 10) -> str:
            try:
                if self.title_vectorizer is None or self.full_title_vectors is None:
                    return "Keyword model not initialized."
                candidate_ids = self.candidate_bus.get_candidates()
                if not candidate_ids:
                    return "No candidates available for search."
                candidate_mask = self.products_df['asin'].isin(candidate_ids)
                candidate_indices = self.products_df.index[candidate_mask].tolist()
                if not candidate_indices:
                    return "No candidates available for search."
                query_vector = self.title_vectorizer.transform([query])
                candidate_matrix = self.full_title_vectors[candidate_indices]
                similarities = cosine_similarity(query_vector, candidate_matrix)[0]
                local_top = np.argsort(similarities)[::-1][:limit]
                top_indices = [candidate_indices[i] for i in local_top]
                result = f"Found {len(top_indices)} products matching keywords:\n\n"
                for idx in top_indices:
                    product = self.products_df.iloc[idx]
                    sim = cosine_similarity(query_vector, self.full_title_vectors[idx]).flatten()[0]
                    result += f"• {product['title']}\n"
                    result += f"  Relevance: {sim:.3f}\n"
                    result += f"  Brand: {product['brand']}, Price: ${product['price_float']:.2f}\n"
                    result += f"  Category: {product['category']}\n\n"
                return result
            except Exception as e:
                logger.error(f"Error in keyword_retrieval: {e}")
                return f"Error in keyword retrieval: {str(e)}"

        async def _get_candidate_bus_status_impl() -> str:
            candidates = self.candidate_bus.get_candidates()
            metadata = self.candidate_bus.metadata
            result = f"Candidate Bus Status:\n"
            result += f"• Total candidates: {len(candidates)}\n"
            result += f"• Last updated: {self.candidate_bus.last_updated}\n"
            if metadata:
                result += f"• Last operation: {metadata.get('mode', 'unknown')}\n"
                if 'constraints' in metadata:
                    result += f"• Last constraints: {metadata['constraints']}\n"
                if 'query' in metadata:
                    result += f"• Last query: {metadata['query']}\n"
            if candidates:
                result += f"\nSample candidates:\n"
                sample_ids = list(candidates)[:5]
                for asin in sample_ids:
                    # Try to find product in subset first, then fall back to full dataset
                    if asin in self.products_subset_df['asin'].values:
                        product = self.products_subset_df[self.products_subset_df['asin'] == asin].iloc[0]
                        result += f"• {product['title']} ({asin})\n"
                    elif asin in self.products_df['asin'].values:
                        product = self.products_df[self.products_df['asin'] == asin].iloc[0]
                        result += f"• {product['title']} ({asin})\n"
                    else:
                        result += f"• Product {asin} (details not available)\n"
            return result

        async def _reset_candidate_bus_impl() -> str:
            limited_ids = set(self.products_df['asin'].head(CANDIDATE_LIMIT).tolist())
            self.candidate_bus.update_candidates(limited_ids)
            self.products_subset_df = (
                self.products_df[self.products_df['asin'].isin(limited_ids)].reset_index(drop=True)
            )
            return f"Candidate bus reset to include {len(limited_ids)} items"

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="keyword_retrieval",
                    description="Keyword TF-IDF retrieval",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            try:
                if name == "keyword_retrieval":
                    text = await _keyword_retrieval_impl(**arguments)
                else:
                    text = f"Unknown tool: {name}"
                return [types.TextContent(type="text", text=text)]
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        self._keyword_retrieval_impl = _keyword_retrieval_impl

    async def keyword_retrieval(self, query: str, limit: int = 10) -> str:
        return await self._keyword_retrieval_impl(query, limit=limit)

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Enhanced Retrieval MCP Server...")
        logger.info(f"Loaded {len(self.products_df)} products from {self.data_path}")
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="enhanced-retrieval-mcp-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Retrieval MCP Server')
    parser.add_argument('--data-path', type=str, 
                       default="/data/zhendong_data/cx/ReCall/data/amazon-electronics/processed/item_meta_5core.jsonl",
                       help='Path to the data file')
    parser.add_argument('--model-cache-path', type=str, 
                       help='Path to cache SentenceTransformer models')
    
    args = parser.parse_args()
    
    server = EnhancedRetrievalMCPServer(
        data_path=args.data_path,
        model_cache_path=args.model_cache_path
    )
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
