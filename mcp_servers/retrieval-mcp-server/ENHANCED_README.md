Enhanced Retrieval MCP Server Guide

Overview:
This enhanced retrieval MCP server provides three different retrieval modes specifically optimized for Amazon electronics data retrieval.

Three Retrieval Modes:

1. Text-to-SQL Hard Filter
Goal: Convert natural language requests into precise SQL queries for high-precision, low-recall filtering
Use Cases: Queries with clear structured constraints
- Price range: "price under $300", "under $500"
- Brand specification: "Sony brand", "Apple products"
- Rating requirements: "rating above 4.0", "rating above 4.5"
- Category constraints: "headphones", "laptops"

2. Semantic Retrieval
Goal: Handle unstructured descriptive needs using vector similarity search
Use Cases: Descriptive and experiential queries
- Usage scenarios: "suitable for travel lightweight products"
- Performance descriptions: "excellent cooling for gaming laptops"
- Experience descriptions: "comfortable for long wearing", "good for long flights"
- Quality descriptions: "high-quality sound", "excellent build quality"

3. Keyword Retrieval
Goal: Exact matching of specific models and technical terms
Use Cases: Technical specifications and model names
- Processors: "i7 processor", "M1 chip"
- Technical features: "noise cancelling", "4K OLED display"
- Connection interfaces: "USB-C Thunderbolt", "Bluetooth 5.0"
- Storage specs: "16GB RAM", "512GB SSD"

Installation and Running:

1. Install Dependencies:
cd /data/agentic-rec/rec-mcp-bench/mcp_servers/retrieval-mcp-server
pip install -r enhanced_requirements.txt

2. Run Server:
python enhanced_server.py

3. Run Tests:
python test_enhanced_server.py

Data Format:
The server uses Amazon electronics data with these fields:
- asin: Product unique identifier
- title: Product title
- price: Price
- brand: Brand
- category: Category
- description: Product description
- rating: Rating