python preprocess_scripts/amazon_electronics_preprocess.py --input_dir /data/zhendong_data/cx/ReCall/data/amazon-electronics/ --sample_users 5000 --max_records 5000


/data/agentic-rec/rec-mcp-bench/mcp_servers/amazon-mcp-server# python test_client.py

cd /data/agentic-rec/rec-mcp-bench/mcp_servers/retrieval-mcp-server
pip install -r enhanced_requirements.txt
python enhanced_server.py