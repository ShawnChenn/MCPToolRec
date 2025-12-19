#!/usr/bin/env python3
"""
Retrieval MCP Server for Recommendation Systems
Provides various retrieval tools for product search and recommendation
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval-mcp-server")

class RetrievalMCPServer:
    """Retrieval MCP Server for recommendation systems"""
    
    def __init__(self):
        self.server = Server("retrieval-mcp-server")
        self.db_path = Path(__file__).parent / "data" / "products.db"
        self.vectorizer = None
        self.product_vectors = None
        self.products_df = None
        
        # Initialize database and data
        self._init_database()
        self._load_sample_data()
        self._setup_tools()
    
    def _init_database(self):
        """Initialize SQLite database for product storage"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    price REAL,
                    rating REAL,
                    brand TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product_id INTEGER,
                    interaction_type TEXT, -- 'view', 'purchase', 'like', 'cart'
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_user ON user_interactions(user_id)")
    
    def _load_sample_data(self):
        """Load sample product data for demonstration"""
        sample_products = [
            {
                "name": "iPhone 15 Pro",
                "category": "Electronics",
                "description": "Latest iPhone with advanced camera system and A17 Pro chip",
                "price": 999.99,
                "rating": 4.8,
                "brand": "Apple",
                "tags": "smartphone,camera,premium,5G"
            },
            {
                "name": "Samsung Galaxy S24",
                "category": "Electronics", 
                "description": "Flagship Android phone with AI features and excellent display",
                "price": 899.99,
                "rating": 4.7,
                "brand": "Samsung",
                "tags": "smartphone,android,AI,display"
            },
            {
                "name": "MacBook Air M3",
                "category": "Computers",
                "description": "Lightweight laptop with M3 chip for productivity and creativity",
                "price": 1299.99,
                "rating": 4.9,
                "brand": "Apple",
                "tags": "laptop,productivity,lightweight,M3"
            },
            {
                "name": "Dell XPS 13",
                "category": "Computers",
                "description": "Premium ultrabook with Intel processors and stunning display",
                "price": 1199.99,
                "rating": 4.6,
                "brand": "Dell",
                "tags": "laptop,ultrabook,premium,intel"
            },
            {
                "name": "Sony WH-1000XM5",
                "category": "Audio",
                "description": "Industry-leading noise canceling wireless headphones",
                "price": 399.99,
                "rating": 4.8,
                "brand": "Sony",
                "tags": "headphones,noise-canceling,wireless,premium"
            }
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if data already exists
            cursor = conn.execute("SELECT COUNT(*) FROM products")
            if cursor.fetchone()[0] == 0:
                # Insert sample data
                for product in sample_products:
                    conn.execute("""
                        INSERT INTO products (name, category, description, price, rating, brand, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        product["name"], product["category"], product["description"],
                        product["price"], product["rating"], product["brand"], product["tags"]
                    ))
                logger.info(f"Loaded {len(sample_products)} sample products")
        
        # Load products into DataFrame for vector operations
        self._load_products_dataframe()
    
    def _load_products_dataframe(self):
        """Load products into pandas DataFrame and create vectors"""
        with sqlite3.connect(self.db_path) as conn:
            self.products_df = pd.read_sql_query("SELECT * FROM products", conn)
        
        if not self.products_df.empty:
            # Create text corpus for vectorization
            text_corpus = []
            for _, row in self.products_df.iterrows():
                text = f"{row['name']} {row['description']} {row['category']} {row['brand']} {row['tags']}"
                text_corpus.append(text)
            
            # Create TF-IDF vectors
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.product_vectors = self.vectorizer.fit_transform(text_corpus)
            logger.info(f"Created vectors for {len(text_corpus)} products")
    
    def _setup_tools(self):
        """Setup MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available retrieval tools"""
            return [
                types.Tool(
                    name="search_products_by_keywords",
                    description="Search products using keywords with filtering options",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "string",
                                "description": "Search keywords"
                            },
                            "category": {
                                "type": "string",
                                "description": "Filter by product category (optional)"
                            },
                            "brand": {
                                "type": "string", 
                                "description": "Filter by brand (optional)"
                            },
                            "min_price": {
                                "type": "number",
                                "description": "Minimum price filter (optional)"
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Maximum price filter (optional)"
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating filter (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            }
                        },
                        "required": ["keywords"]
                    }
                ),
                types.Tool(
                    name="vector_similarity_search",
                    description="Find products similar to a query using vector similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for similarity matching"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Minimum similarity threshold (0-1)",
                                "default": 0.1
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="item_to_item_similarity",
                    description="Find products similar to a given product ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "integer",
                                "description": "ID of the reference product"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of similar products to return",
                                "default": 10
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Minimum similarity threshold (0-1)",
                                "default": 0.1
                            }
                        },
                        "required": ["product_id"]
                    }
                ),
                types.Tool(
                    name="sql_query_products",
                    description="Execute SQL queries on the products database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute (SELECT statements only)"
                            },
                            "params": {
                                "type": "array",
                                "description": "Parameters for parameterized queries (optional)",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="get_product_categories",
                    description="Get all available product categories",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_product_brands",
                    description="Get all available product brands",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_user_history",
                    description="Get user interaction history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User identifier"
                            },
                            "interaction_type": {
                                "type": "string",
                                "description": "Filter by interaction type (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of interactions to return",
                                "default": 50
                            }
                        },
                        "required": ["user_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls"""
            try:
                if name == "search_products_by_keywords":
                    result = await self._search_products_by_keywords(**arguments)
                elif name == "vector_similarity_search":
                    result = await self._vector_similarity_search(**arguments)
                elif name == "item_to_item_similarity":
                    result = await self._item_to_item_similarity(**arguments)
                elif name == "sql_query_products":
                    result = await self._sql_query_products(**arguments)
                elif name == "get_product_categories":
                    result = await self._get_product_categories()
                elif name == "get_product_brands":
                    result = await self._get_product_brands()
                elif name == "get_user_history":
                    result = await self._get_user_history(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]
    
    async def _search_products_by_keywords(self, keywords: str, category: str = None, 
                                         brand: str = None, min_price: float = None,
                                         max_price: float = None, min_rating: float = None,
                                         limit: int = 10) -> Dict[str, Any]:
        """Search products by keywords with filters"""
        
        # Build SQL query with filters
        query = """
            SELECT * FROM products 
            WHERE (name LIKE ? OR description LIKE ? OR tags LIKE ?)
        """
        params = [f"%{keywords}%", f"%{keywords}%", f"%{keywords}%"]
        
        if category:
            query += " AND category LIKE ?"
            params.append(f"%{category}%")
        
        if brand:
            query += " AND brand LIKE ?"
            params.append(f"%{brand}%")
        
        if min_price is not None:
            query += " AND price >= ?"
            params.append(min_price)
        
        if max_price is not None:
            query += " AND price <= ?"
            params.append(max_price)
        
        if min_rating is not None:
            query += " AND rating >= ?"
            params.append(min_rating)
        
        query += " ORDER BY rating DESC, price ASC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            products = [dict(row) for row in cursor.fetchall()]
        
        return {
            "query": keywords,
            "filters": {
                "category": category,
                "brand": brand,
                "min_price": min_price,
                "max_price": max_price,
                "min_rating": min_rating
            },
            "total_results": len(products),
            "products": products
        }
    
    async def _vector_similarity_search(self, query: str, limit: int = 10, 
                                      threshold: float = 0.1) -> Dict[str, Any]:
        """Find products using vector similarity"""
        
        if self.vectorizer is None or self.product_vectors is None:
            return {"error": "Vector search not available - no products loaded"}
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.product_vectors).flatten()
        
        # Get top similar products above threshold
        similar_indices = np.where(similarities >= threshold)[0]
        similar_scores = similarities[similar_indices]
        
        # Sort by similarity score
        sorted_indices = similar_indices[np.argsort(similar_scores)[::-1]][:limit]
        
        results = []
        for idx in sorted_indices:
            product = self.products_df.iloc[idx].to_dict()
            product['similarity_score'] = float(similarities[idx])
            results.append(product)
        
        return {
            "query": query,
            "total_results": len(results),
            "threshold": threshold,
            "products": results
        }
    
    async def _item_to_item_similarity(self, product_id: int, limit: int = 10,
                                     threshold: float = 0.1) -> Dict[str, Any]:
        """Find products similar to a given product"""
        
        if self.product_vectors is None:
            return {"error": "Similarity search not available - no products loaded"}
        
        # Find the product index
        product_idx = None
        for idx, row in self.products_df.iterrows():
            if row['id'] == product_id:
                product_idx = idx
                break
        
        if product_idx is None:
            return {"error": f"Product with ID {product_id} not found"}
        
        # Get the product vector
        product_vector = self.product_vectors[product_idx]
        
        # Calculate similarities with all other products
        similarities = cosine_similarity(product_vector, self.product_vectors).flatten()
        
        # Exclude the product itself and filter by threshold
        similar_indices = np.where((similarities >= threshold) & (np.arange(len(similarities)) != product_idx))[0]
        similar_scores = similarities[similar_indices]
        
        # Sort by similarity score
        sorted_indices = similar_indices[np.argsort(similar_scores)[::-1]][:limit]
        
        results = []
        for idx in sorted_indices:
            product = self.products_df.iloc[idx].to_dict()
            product['similarity_score'] = float(similarities[idx])
            results.append(product)
        
        reference_product = self.products_df.iloc[product_idx].to_dict()
        
        return {
            "reference_product": reference_product,
            "total_results": len(results),
            "threshold": threshold,
            "similar_products": results
        }
    
    async def _sql_query_products(self, query: str, params: List[str] = None) -> Dict[str, Any]:
        """Execute SQL query on products database"""
        
        # Security check - only allow SELECT statements
        if not query.strip().upper().startswith('SELECT'):
            return {"error": "Only SELECT queries are allowed"}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)
                results = [dict(row) for row in cursor.fetchall()]
            
            return {
                "query": query,
                "params": params,
                "total_results": len(results),
                "results": results
            }
        
        except Exception as e:
            return {"error": f"SQL execution error: {str(e)}"}
    
    async def _get_product_categories(self) -> Dict[str, Any]:
        """Get all available product categories"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT category FROM products ORDER BY category")
            categories = [row[0] for row in cursor.fetchall()]
        
        return {"categories": categories}
    
    async def _get_product_brands(self) -> Dict[str, Any]:
        """Get all available product brands"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT brand FROM products ORDER BY brand")
            brands = [row[0] for row in cursor.fetchall()]
        
        return {"brands": brands}
    

    async def _get_user_history(self, user_id: str, interaction_type: str = None,
                               limit: int = 50) -> Dict[str, Any]:
        """Get user interaction history"""
        
        query = """
            SELECT ui.*, p.name as product_name, p.category, p.brand, p.price
            FROM user_interactions ui
            JOIN products p ON ui.product_id = p.id
            WHERE ui.user_id = ?
        """
        params = [user_id]
        
        if interaction_type:
            query += " AND ui.interaction_type = ?"
            params.append(interaction_type)
        
        query += " ORDER BY ui.timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            interactions = [dict(row) for row in cursor.fetchall()]
        
        return {
            "user_id": user_id,
            "interaction_type_filter": interaction_type,
            "total_interactions": len(interactions),
            "interactions": interactions
        }

async def main():
    """Main server entry point"""
    server_instance = RetrievalMCPServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="retrieval-mcp-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())