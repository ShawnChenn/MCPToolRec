Enhanced Retrieval MCP Server

Three modes:
1. Text-to-SQL: Precise filtering (price, brand, rating)
2. Semantic: Vector similarity for descriptions
3. Keyword: Exact technical term matching

Usage: await text2sql_retrieval("Sony under $300")
Setup: pip install -r enhanced_requirements.txt; python enhanced_server.py