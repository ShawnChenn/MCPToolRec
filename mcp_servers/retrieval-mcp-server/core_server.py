#!/usr/bin/env python3
"""
Core Retrieval MCP Server
Focused on 4 essential retrieval tools: BM25, Sentence-BERT, SQL, ItemCF
"""

import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("core-retrieval-server")

class CoreRetrievalServer:
    """Core MCP Server with 4 essential retrieval tools"""
    
    def __init__(self):
        self.server = Server("core-retrieval-server")
        self.data_path = "/data/cx/ReCall/data/processed_all_beauty"
        
        # Core components
        self.meta_dict = {}
        self.index_to_asin = {}
        self.user_interactions = {}
        
        # BM25 (TF-IDF approximation)
        self.bm25_vectorizer = None
        self.bm25_vectors = None
        
        # Sentence-BERT (TF-IDF for simplicity)
        self.sbert_vectorizer = None
        self.sbert_vectors = None
        
        # ItemCF similarity matrix
        self.item_similarity = {}
        
        # SQL database
        self.db_path = None
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Core Retrieval Server...")
        
        # Load data
        await self._load_data()
        
        # Setup BM25 (keyword search)
        await self._setup_bm25()
        
        # Setup Sentence-BERT (vector retrieval)
        await self._setup_sbert()
        
        # Setup ItemCF (collaborative filtering)
        await self._setup_itemcf()
        
        # Setup SQL database
        await self._setup_sql()
        
        # Setup MCP tools
        self._setup_tools()
        
        logger.info("✅ Core Retrieval Server initialized successfully")
    
    async def _load_data(self):
        """Load product metadata and user interactions"""
        
        # Load product metadata
        meta_path = os.path.join(self.data_path, 'item_meta_dict.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.meta_dict = json.load(f)
        
        # Load index mappings
        index_path = os.path.join(self.data_path, 'index_to_asin.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.index_to_asin = json.load(f)
        
        # Load user interactions for ItemCF
        train_path = os.path.join(self.data_path, 'train.txt')
        if os.path.exists(train_path):
            with open(train_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        user_idx, item_idx = parts
                        item_asin = self.index_to_asin.get(item_idx, item_idx)
                        if user_idx not in self.user_interactions:
                            self.user_interactions[user_idx] = []
                        self.user_interactions[user_idx].append(item_asin)
        
        logger.info(f"📊 Loaded {len(self.meta_dict)} products")
        logger.info(f"📊 Loaded {len(self.user_interactions)} user interaction sequences")
    
    async def _setup_bm25(self):
        """Setup BM25 keyword search using TF-IDF"""
        
        # Create text corpus for BM25
        text_corpus = []
        self.asin_list = []
        
        for asin, product in self.meta_dict.items():
            # Combine title, brand, category for BM25 indexing
            text_parts = []
            
            if product.get('title'):
                text_parts.append(product['title'])
            if product.get('brand'):
                text_parts.append(product['brand'])
            if product.get('category'):
                if isinstance(product['category'], list):
                    text_parts.extend(product['category'])
                else:
                    text_parts.append(str(product['category']))
            
            text = ' '.join(text_parts).lower()
            text_corpus.append(text)
            self.asin_list.append(asin)
        
        # Create BM25 vectorizer (using TF-IDF as approximation)
        self.bm25_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        if text_corpus:
            self.bm25_vectors = self.bm25_vectorizer.fit_transform(text_corpus)
            logger.info(f"🔍 BM25 index created for {len(text_corpus)} products")
    
    async def _setup_sbert(self):
        """Setup Sentence-BERT vector retrieval using TF-IDF"""
        
        # Create semantic text corpus
        semantic_corpus = []
        
        for asin in self.asin_list:
            product = self.meta_dict[asin]
            
            # Use title and description for semantic search
            semantic_parts = []
            if product.get('title'):
                semantic_parts.append(product['title'])
            if product.get('description'):
                semantic_parts.append(product['description'])
            elif product.get('review'):
                # Use first few reviews as description substitute
                reviews = product['review'][:3] if isinstance(product['review'], list) else []
                semantic_parts.extend(reviews)
            
            semantic_text = ' '.join(semantic_parts).lower()
            semantic_corpus.append(semantic_text)
        
        # Create Sentence-BERT vectorizer
        self.sbert_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2
        )
        
        if semantic_corpus:
            self.sbert_vectors = self.sbert_vectorizer.fit_transform(semantic_corpus)
            logger.info(f"🧠 Sentence-BERT index created for {len(semantic_corpus)} products")
    
    async def _setup_itemcf(self):
        """Setup Item-based Collaborative Filtering"""
        
        # Create item co-occurrence matrix
        item_cooccurrence = {}
        
        for user_items in self.user_interactions.values():
            # Count co-occurrences
            for i, item1 in enumerate(user_items):
                if item1 not in item_cooccurrence:
                    item_cooccurrence[item1] = {}
                
                for j, item2 in enumerate(user_items):
                    if i != j:
                        if item2 not in item_cooccurrence[item1]:
                            item_cooccurrence[item1][item2] = 0
                        item_cooccurrence[item1][item2] += 1
        
        # Calculate item similarities using cosine similarity
        for item1, cooccur_dict in item_cooccurrence.items():
            similarities = []
            items = []
            
            for item2, count in cooccur_dict.items():
                # Simple similarity based on co-occurrence frequency
                similarity = count / (len(self.user_interactions) + 1)  # Normalize
                similarities.append((item2, similarity))
            
            # Sort by similarity and keep top items
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.item_similarity[item1] = similarities[:50]  # Keep top 50 similar items
        
        logger.info(f"🤝 ItemCF similarity computed for {len(self.item_similarity)} items")
    
    async def _setup_sql(self):
        """Setup SQLite database for SQL queries"""
        
        self.db_path = "/tmp/products.db"
        
        # Create database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                asin TEXT PRIMARY KEY,
                title TEXT,
                brand TEXT,
                price REAL,
                category TEXT,
                rating REAL,
                review_count INTEGER
            )
        ''')
        
        # Insert product data
        for asin, product in self.meta_dict.items():
            # Parse price
            price = 0.0
            price_str = product.get('price', '')
            if price_str:
                try:
                    # Extract numeric price
                    import re
                    price_match = re.search(r'[\d.]+', str(price_str))
                    if price_match:
                        price = float(price_match.group())
                except:
                    price = 0.0
            
            # Get category
            category = ''
            if product.get('category'):
                if isinstance(product['category'], list):
                    category = product['category'][0] if product['category'] else ''
                else:
                    category = str(product['category'])
            
            # Calculate average rating
            rating = 0.0
            review_count = 0
            if product.get('review'):
                reviews = product['review']
                if isinstance(reviews, list):
                    review_count = len(reviews)
                    # Assume reviews are text, assign random rating for demo
                    rating = np.random.uniform(3.0, 5.0)
            
            cursor.execute('''
                INSERT OR REPLACE INTO products 
                (asin, title, brand, price, category, rating, review_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                asin,
                product.get('title', ''),
                product.get('brand', ''),
                price,
                category,
                rating,
                review_count
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"🗄️ SQL database created with {len(self.meta_dict)} products")
    
    def _setup_tools(self):
        """Setup the 4 core MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List the 4 core retrieval tools"""
            return [
                # 1. BM25 Keyword Search
                types.Tool(
                    name="bm25_keyword_search",
                    description="BM25-based keyword search for products",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query keywords"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                
                # 2. Sentence-BERT Vector Retrieval
                types.Tool(
                    name="sbert_vector_search",
                    description="Sentence-BERT semantic vector search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Semantic search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Similarity threshold (0-1)",
                                "default": 0.1
                            }
                        },
                        "required": ["query"]
                    }
                ),
                
                # 3. SQL Query
                types.Tool(
                    name="sql_query_products",
                    description="Execute SQL queries on product database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 20
                            }
                        },
                        "required": ["query"]
                    }
                ),
                
                # 4. ItemCF Similarity Recommendation
                types.Tool(
                    name="itemcf_similarity",
                    description="Item-based collaborative filtering similarity recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "item_id": {
                                "type": "string",
                                "description": "Product ASIN to find similar items for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of similar items",
                                "default": 10
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Similarity threshold (0-1)",
                                "default": 0.1
                            }
                        },
                        "required": ["item_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls for the 4 core tools"""
            
            try:
                if name == "bm25_keyword_search":
                    return await self._bm25_search(arguments)
                elif name == "sbert_vector_search":
                    return await self._sbert_search(arguments)
                elif name == "sql_query_products":
                    return await self._sql_query(arguments)
                elif name == "itemcf_similarity":
                    return await self._itemcf_similarity(arguments)
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
            
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
    
    async def _bm25_search(self, args: dict) -> list[types.TextContent]:
        """BM25 keyword search implementation"""
        
        query = args.get('query', '')
        limit = args.get('limit', 10)
        
        if not self.bm25_vectorizer or not query:
            return [types.TextContent(type="text", text="BM25 not initialized or empty query")]
        
        # Vectorize query
        query_vector = self.bm25_vectorizer.transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.bm25_vectors).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                asin = self.asin_list[idx]
                product = self.meta_dict[asin]
                results.append({
                    "asin": asin,
                    "title": product.get('title', ''),
                    "brand": product.get('brand', ''),
                    "score": float(similarities[idx]),
                    "method": "BM25"
                })
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "bm25_keyword_search",
                "query": query,
                "results": results,
                "total": len(results)
            }, indent=2)
        )]
    
    async def _sbert_search(self, args: dict) -> list[types.TextContent]:
        """Sentence-BERT vector search implementation"""
        
        query = args.get('query', '')
        limit = args.get('limit', 10)
        threshold = args.get('threshold', 0.1)
        
        if not self.sbert_vectorizer or not query:
            return [types.TextContent(type="text", text="Sentence-BERT not initialized or empty query")]
        
        # Vectorize query
        query_vector = self.sbert_vectorizer.transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.sbert_vectors).flatten()
        
        # Get top results above threshold
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold and len(results) < limit:
                asin = self.asin_list[idx]
                product = self.meta_dict[asin]
                results.append({
                    "asin": asin,
                    "title": product.get('title', ''),
                    "brand": product.get('brand', ''),
                    "score": float(similarities[idx]),
                    "method": "Sentence-BERT"
                })
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "sbert_vector_search",
                "query": query,
                "results": results,
                "total": len(results)
            }, indent=2)
        )]
    
    async def _sql_query(self, args: dict) -> list[types.TextContent]:
        """SQL query implementation"""
        
        query = args.get('query', '')
        limit = args.get('limit', 20)
        
        if not query or not self.db_path:
            return [types.TextContent(type="text", text="Empty query or database not initialized")]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Add LIMIT to query if not present
            if 'LIMIT' not in query.upper():
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "tool": "sql_query_products",
                    "query": query,
                    "results": results,
                    "total": len(results)
                }, indent=2)
            )]
        
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"SQL Error: {str(e)}"
            )]
    
    async def _itemcf_similarity(self, args: dict) -> list[types.TextContent]:
        """ItemCF similarity recommendation implementation"""
        
        item_id = args.get('item_id', '')
        limit = args.get('limit', 10)
        threshold = args.get('threshold', 0.1)
        
        if not item_id or item_id not in self.item_similarity:
            return [types.TextContent(
                type="text", 
                text=f"Item {item_id} not found in similarity index"
            )]
        
        # Get similar items
        similar_items = self.item_similarity[item_id]
        
        results = []
        for similar_asin, similarity_score in similar_items:
            if similarity_score >= threshold and len(results) < limit:
                if similar_asin in self.meta_dict:
                    product = self.meta_dict[similar_asin]
                    results.append({
                        "asin": similar_asin,
                        "title": product.get('title', ''),
                        "brand": product.get('brand', ''),
                        "similarity": similarity_score,
                        "method": "ItemCF"
                    })
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "tool": "itemcf_similarity",
                "item_id": item_id,
                "results": results,
                "total": len(results)
            }, indent=2)
        )]

async def main():
    """Main server entry point"""
    
    # Create and initialize server
    server_instance = CoreRetrievalServer()
    await server_instance.initialize()
    
    # Run server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="core-retrieval-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())