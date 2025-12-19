#!/usr/bin/env python3
"""
LLM-Driven Retrieval Task Synthesizer
Integrates GPT-OSS model for natural instruction generation with MCP tool combinations
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

class LLMRetrievalSynthesizer:
    """LLM-powered synthesizer for retrieval tasks combining multiple tools"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 data_path: Optional[str] = None):
        """
        Initialize the LLM-driven synthesizer
        
        Args:
            model_path: Path to GPT-OSS model (auto-detected if None)
            data_path: Path to product data (auto-detected if None)
        """
        # Import config here to avoid circular imports
        try:
            from .llm_config import LLMSynthesisConfig
        except ImportError:
            from llm_config import LLMSynthesisConfig
        
        self.model_path = model_path or LLMSynthesisConfig.get_model_path()
        self.data_path = data_path or LLMSynthesisConfig.get_data_path()
        self.tokenizer = None
        self.model = None
        
        # Load product data
        self.meta_dict = self._load_product_data()
        self.index_to_asin = self._load_index_mapping()
        self.index_to_uid = self._load_user_mapping()
        
        # Load user interaction data for sequential recommendation
        self.user_interactions = self._load_user_interactions()
        self.history_instructions = self._load_history_instructions()
        
        # MCP tool definitions
        self.mcp_tools = self._define_mcp_tools()
        self.tool_combinations = self._create_tool_combinations()
        
    def _load_gpt_oss_model(self):
        """Load GPT-OSS model for instruction generation"""
        if self.tokenizer is None or self.model is None:
            print(f'Loading GPT-OSS model from: {self.model_path}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
            )
        return self.tokenizer, self.model
    
    def _load_product_data(self) -> Dict[str, Any]:
        meta_path = os.path.join(self.data_path, 'item_meta_dict.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        jsonl_path = os.path.join(self.data_path, 'item_meta_5core.jsonl')
        if not os.path.exists(jsonl_path):
            alt = os.path.join(self.data_path, 'processed', 'item_meta_5core.jsonl')
            jsonl_path = alt if os.path.exists(alt) else jsonl_path
        meta = {}
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    o = json.loads(s)
                    a = o.get('asin')
                    if a:
                        meta[a] = o
        return meta
    
    def _load_index_mapping(self) -> Dict[str, str]:
        """Load index to ASIN mapping"""
        index_path = os.path.join(self.data_path, 'index_to_asin.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_user_mapping(self) -> Dict[str, str]:
        """Load index to user ID mapping"""
        user_path = os.path.join(self.data_path, 'index_to_uid.json')
        if os.path.exists(user_path):
            with open(user_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_user_interactions(self) -> Dict[str, Dict[str, Any]]:
        train_interactions = defaultdict(list)
        test_interactions = {}
        train_path = os.path.join(self.data_path, 'train.txt')
        test_path = os.path.join(self.data_path, 'test.txt')
        if os.path.exists(train_path) and os.path.exists(test_path):
            with open(train_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        user_idx, item_idx = parts
                        user_id = self.index_to_uid.get(user_idx, user_idx)
                        item_asin = self.index_to_asin.get(item_idx, item_idx)
                        train_interactions[user_id].append(item_asin)
            with open(test_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        user_idx, item_idx = parts
                        user_id = self.index_to_uid.get(user_idx, user_idx)
                        item_asin = self.index_to_asin.get(item_idx, item_idx)
                        test_interactions[user_id] = item_asin
        else:
            jsonl_path = os.path.join(self.data_path, 'user_sequences_5core.jsonl')
            if not os.path.exists(jsonl_path):
                alt = os.path.join(self.data_path, 'processed', 'user_sequences_5core.jsonl')
                jsonl_path = alt if os.path.exists(alt) else jsonl_path
            if os.path.exists(jsonl_path):
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        o = json.loads(line)
                        uid = o.get('user_id')
                        items = o.get('items', [])
                        if not uid or not items:
                            continue
                        hist = [it.get('asin') for it in items[:-1] if it.get('asin')]
                        tgt = items[-1].get('asin') if items[-1].get('asin') else None
                        if hist:
                            train_interactions[uid].extend(hist)
                        if tgt:
                            test_interactions[uid] = tgt
        user_data = {}
        for user_id in train_interactions:
            user_data[user_id] = {
                'history': train_interactions[user_id],
                'target': test_interactions.get(user_id, None)
            }
        print(f"📊 Loaded interactions for {len(user_data)} users")
        print(f"📊 Users with history: {len(train_interactions)}")
        print(f"📊 Users with target: {len(test_interactions)}")
        return user_data
    
    def _load_history_instructions(self) -> Dict[str, Any]:
        """Load existing history-based instructions"""
        history_path = os.path.join(self.data_path, 'history_tool_instructions.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                data = json.load(f)
                # Convert to dict keyed by user_id for easy lookup
                return {str(item['user_id']): item for item in data}
        return {}
    
    def _define_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define the 4 core MCP retrieval tools"""
        return {
            "bm25_keyword_search": {
                "description": "BM25-based keyword search for products",
                "parameters": ["query", "limit"],
                "complexity": 1,
                "method": "BM25"
            },
            "sbert_vector_search": {
                "description": "Sentence-BERT semantic vector search",
                "parameters": ["query", "limit", "threshold"],
                "complexity": 2,
                "method": "Sentence-BERT"
            },
            "sql_query_products": {
                "description": "Execute SQL queries on product database",
                "parameters": ["query", "limit"],
                "complexity": 3,
                "method": "SQL"
            }
        }
    
    def _create_tool_combinations(self) -> Dict[str, List[str]]:
        """Define meaningful combinations for the 4 core tools"""
        return {
            # Single tool scenarios
            "keyword_search": ["bm25_keyword_search"],
            "semantic_search": ["sbert_vector_search"],
            "sql_query": ["sql_query_products"],
            "similarity_recommendation": ["itemcf_similarity"],
            "missing_price_query": ["bm25_keyword_search"],
            "review_sentiment": ["sbert_vector_search"],
            "review_summary": ["bm25_keyword_search"],
            
            # Two-tool combinations
            "keyword_semantic": ["bm25_keyword_search", "sbert_vector_search"],
            "keyword_similarity": ["bm25_keyword_search", "itemcf_similarity"],
            "semantic_similarity": ["sbert_vector_search", "itemcf_similarity"],
            "sql_semantic": ["sql_query_products", "sbert_vector_search"],
            
            # Three-tool combinations
            "comprehensive_search": ["bm25_keyword_search", "sbert_vector_search", "itemcf_similarity"],
            "advanced_query": ["sql_query_products", "sbert_vector_search", "itemcf_similarity"]
        }
    
   
    def _get_sample_product_context(self) -> Dict[str, Any]:
        """Get a random product from the dataset for context"""
        if not self.meta_dict:
            return {}
        
        # Get a random product
        asin = random.choice(list(self.meta_dict.keys()))
        product = self.meta_dict[asin]
        
        return {
            "asin": asin,
            "title": product.get("title", "").strip(),
            "brand": product.get("brand", "").strip(),
            "price": product.get("price", "").strip(),
            "category": product.get("category", ["Electronics"])[0] if product.get("category") else "Electronics",
            "reviews": product.get("review", [])[:3] if product.get("review") else []
        }
    
    def _get_user_sequence_context(self) -> Dict[str, Any]:
        """Get a real user's historical behavior and target item for sequential recommendation"""
        if not self.user_interactions:
            return self._get_sample_product_context()
        
        # Select a random user with interactions
        user_id = random.choice(list(self.user_interactions.keys()))
        user_data = self.user_interactions[user_id]
        
        history_items = user_data.get('history', [])
        target_asin = user_data.get('target', None)
        
        # Get product details for history items
        history_products = []
        for asin in history_items[:10]:  # Limit to last 10 interactions
            if asin in self.meta_dict:
                product = self.meta_dict[asin]
                history_products.append({
                    "asin": asin,
                    "title": product.get("title", "").strip(),
                    "brand": product.get("brand", "").strip(),
                    "category": product.get("category", ["Beauty"])[0] if product.get("category") else "Beauty"
                })
        
        # Get target item details
        target_product = None
        if target_asin and target_asin in self.meta_dict:
            product = self.meta_dict[target_asin]
            target_product = {
                "asin": target_asin,
                "title": product.get("title", "").strip(),
                "brand": product.get("brand", "").strip(),
                "price": product.get("price", "").strip(),
                "category": product.get("category", ["Electronics"])[0] if product.get("category") else "Electronics"
            }
        
        # Check if we have existing instruction for this user
        existing_instruction = None
        if user_id in self.history_instructions:
            user_data_inst = self.history_instructions[user_id]
            existing_instruction = user_data_inst.get('instructions', [])
            if existing_instruction:
                existing_instruction = existing_instruction[0]  # Take first instruction
        
        return {
            "user_id": user_id,
            "history_items": history_products,
            "target_item": target_product,
            "existing_instruction": existing_instruction,
            "total_interactions": len(history_items) + (1 if target_asin else 0),
            "history_length": len(history_items),
            "has_target": target_product is not None
        }
    
    def _generate_llm_instruction(self, scenario: str, tools: List[str], 
                                product_context: Dict[str, Any], 
                                budget_range: Tuple[int, int]) -> str:
        """Generate natural user instruction using LLM"""
        # Always use LLM path for instruction generation
        tokenizer, model = self._load_gpt_oss_model()
        
        # Create context for instruction generation
        tool_descriptions = []
        for tool in tools:
            if tool in self.mcp_tools:
                tool_descriptions.append(f"- {tool}: {self.mcp_tools[tool]['description']}")
        
        tools_context = "\n".join(tool_descriptions)

        include_sql_info = 'sql_query_products' in tools and scenario != "missing_price_query"

        product_info = []
        if scenario == "semantic_retrieval":
            desc = (product_context.get('prompt_description') or product_context.get('description') or '')
            if isinstance(desc, (list, dict)):
                try:
                    desc = ' '.join(str(i) for i in desc) if isinstance(desc, list) else ' '.join(str(v) for v in desc.values())
                except Exception:
                    desc = str(desc)
            desc = str(desc).replace('\n', ' ').strip()
            if len(desc) > 300:
                desc = desc[:300]
            product_info.append(f"Product description: {desc}")
        elif scenario == "sql_hard_filter":
            brand = (product_context.get('brand') or '').strip()
            category = (product_context.get('category') or '').strip()
            has_price = bool((product_context.get('price') or '').strip()) and budget_range and (sum(budget_range) > 0)
            if brand:
                product_info.append(f"brand: {brand}")
            if category:
                product_info.append(f"category: {category}")
            if has_price:
                price_text = f"${budget_range[0]}-${budget_range[1]}"
                product_info.append(f"price_range: {price_text}")
        elif scenario == "missing_price_query":
            brand = (product_context.get('brand') or '').strip()
            category = (product_context.get('category') or '').strip()
            title = (product_context.get('title') or '').strip()
            if brand:
                product_info.append(f"brand: {brand}")
            if category:
                product_info.append(f"category: {category}")
            if title:
                product_info.append(f"product: {title}")
            # Note: explicitly NOT including price since this is for missing price queries
        elif include_sql_info:
            if product_context.get("brand"):
                product_info.append(f"brand: {product_context['brand']}")
            if product_context.get("price"):
                product_info.append(f"budget_range: ${budget_range[0]}-${budget_range[1]}")
            if product_context.get("category"):
                product_info.append(f"category: {product_context['category']}")
        else:
            product_info.append(f"Product title: { (product_context.get('title') or '').replace('&amp', '') }")
        # Include short reviews preview for review scenarios
        if scenario in ("review_sentiment", "review_summary"):
            reviews = product_context.get("reviews") or []
            if isinstance(reviews, list) and reviews:
                try:
                    preview = " ".join(str(r) for r in reviews[:3])
                except Exception:
                    preview = str(reviews)[:300]
                preview = preview.replace("\n", " ").strip()
                if len(preview) > 300:
                    preview = preview[:300]
                product_info.append(f"reviews_preview: {preview}")
        product_context_str = ", ".join(product_info)
        extra_context_lines = []
        # No negative samples and no extra target info in prompt
        extra_context = "\n" + "\n".join(extra_context_lines) if extra_context_lines else ""
        
        if scenario == "semantic_retrieval":
            prompt = f"""You are generating implicit user queries for a product recommendation system.

Product Context: {product_context_str}{extra_context}

Available Tools:
{tools_context}

Task: Create a natural search query that reflects needs or use-cases based on the product description, without naming the exact product.

Requirements:
- Do NOT mention brand names or model numbers
- Do NOT quote the full title; rely on features/use-cases from description
- Keep it concise and natural (12–20 words)
- The query should logically require using the tools: {', '.join(tools)}
 - Write the query in English only

Output format: Just the query text, enclosed in <answer> </answer> tags.

"""
        elif scenario == "sql_hard_filter":
            prompt = f"""You are generating structured natural queries for hard filtering.

Product Context: {product_context_str}{extra_context}

Available Tools:
{tools_context}

Task: Create a concise natural query specifying category, optional price constraints (only if price is provided), and optionally brand to filter products.

Requirements:
- Always include category; include price constraints only when price appears in the context
- Use patterns like "category X under $Y", "between $A and $B", "brand Z"
- Keep it short and actionable (≤25 words)
- The query should logically require using the tools: {', '.join(tools)}
 - Write the query in English only

Output format: Just the query text, enclosed in <answer> </answer> tags.

"""
        elif scenario == "missing_price_query":
            prompt = f"""You are generating queries to find price information for products.

Product Context: {product_context_str}{extra_context}

Available Tools:
{tools_context}

Task: Create a natural query asking for the price of a specific product or similar products in the same category.

Requirements:
- Ask about price, cost, or pricing information
- Include product name, brand, or category to identify the product
- Make it sound like a user wanting to know how much something costs
- Keep it natural and conversational (≤25 words)
- The query should logically require using the tools: {', '.join(tools)}
 - Write the query in English only

Output format: Just the query text, enclosed in <answer> </answer> tags.

"""
        elif scenario == "review_sentiment":
            prompt = f"""You are generating user queries to analyze sentiment in product reviews.

Product Context: {product_context_str}{extra_context}

Available Tools:
{tools_context}

Task: Create a natural query asking about the overall sentiment or customer satisfaction in the reviews for this product.

Requirements:
 - Refer to opinions or feedback (e.g., overall sentiment, customer satisfaction)
 - Do NOT quote full reviews; keep it general and concise
 - Keep it natural and conversational (≤25 words)
 - The query should logically require using the tools: {', '.join(tools)}
 - Write the query in English only

Output format: Just the query text, enclosed in <answer> </answer> tags.

"""
        elif scenario == "review_summary":
            prompt = f"""You are generating user queries to summarize product reviews.

Product Context: {product_context_str}{extra_context}

Available Tools:
{tools_context}

Task: Create a natural query asking to summarize key pros and cons from user reviews for this product.

Requirements:
 - Ask for a summary or main points (pros and cons, common feedback)
 - Do NOT quote full reviews; keep it general and concise
 - Keep it natural and conversational (≤25 words)
 - The query should logically require using the tools: {', '.join(tools)}
 - Write the query in English only

Output format: Just the query text, enclosed in <answer> </answer> tags.

"""
        else:
            # Default case for keyword search and other scenarios
            scenario_instruction = "Create a natural keyword search query based on the product information."
            prompt = f"""You are generating user keyword queries for a product recommendation system.

Product Context: {product_context_str}{extra_context}

Available Tools:
{tools_context}

Task: {scenario_instruction}

Requirements:
- Extract 2-5 salient keywords from the product title (drop stopwords, modifiers, punctuation, and special characters; ASCII letters/numbers only)
- Form a natural search query incorporating these keywords, adding minimal descriptive words to make it sound like a real user query (e.g., "Best wireless headphones for running")
- Do NOT use the full title or exceed the keyword limit
- The query should logically require using the tools: {', '.join(tools)}
- Keep it short and natural (≤30 words)
 - Write the query in English only

Output format: Just the query text, enclosed in <answer> </answer> tags.

"""

        messages = [{"role": "user", "content": prompt}]
        tokenizer, model = self._load_gpt_oss_model()
        
        print('messages:', messages)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=1.0)
        response = tokenizer.decode(outputs[0])
        # Extract answer
        answer = response.split("<answer>")[-1].split("</answer>")[0].strip()
        print('ANSWER', answer)
        
        return answer
    def _infer_tool_sequence(self, tools: List[str], user_query: str, 
                           budget_range: Tuple[int, int], 
                           product_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer logical tool calling sequence based on query and tools"""
        
        sequence = []
        
        for i, tool in enumerate(tools):
            tool_info = self.mcp_tools[tool]
            
            # Generate purpose based on tool and context
            purposes = {
                "bm25_keyword_search": "Find products using BM25 keyword matching",
                "sbert_vector_search": "Semantic search using sentence embeddings",
                "sql_query_products": "Execute complex database queries",
                "itemcf_similarity": "Find similar items using collaborative filtering"
            }
            
            # Generate parameters based on tool type and context
            parameters = {}
            
            if tool == "bm25_keyword_search":
                # Use the generated user query (title-only) directly
                parameters = {
                    "query": user_query,
                    "limit": random.randint(10, 15)
                }
                
            elif tool == "sbert_vector_search":
                parameters = {
                    "query": user_query,
                    "limit": random.randint(8, 12),
                    "threshold": round(random.uniform(0.1, 0.3), 1)
                }
                
            elif tool == "itemcf_similarity":
                # Use target item or sample item
                if product_context.get('target_item'):
                    item_id = product_context['target_item']['asin']
                elif product_context.get('asin'):
                    item_id = product_context['asin']
                else:
                    item_id = "B00006L9LC"  # Sample ASIN from data
                
                parameters = {
                    "item_id": item_id,
                    "limit": random.randint(6, 10),
                    "threshold": round(random.uniform(0.1, 0.3), 1)
                }
                
            elif tool == "sql_query_products":
                # Generate SQL query based on context
                category = product_context.get('category', 'Electronics')
                has_price = budget_range and (sum(budget_range) > 0) and bool((product_context.get('price') or '').strip())
                if product_context.get('history_items'):
                    # Query based on user history
                    history_brands = [item.get('brand', '') for item in product_context['history_items'] if item.get('brand')]
                    if history_brands:
                        brands_str = "', '".join(set(history_brands))
                        sql_query = f"SELECT * FROM products WHERE brand IN ('{brands_str}')"
                        if has_price:
                            sql_query += f" AND price BETWEEN {budget_range[0]} AND {budget_range[1]}"
                    else:
                        sql_query = f"SELECT * FROM products WHERE category='{category}'"
                        if has_price:
                            sql_query += f" AND price BETWEEN {budget_range[0]} AND {budget_range[1]}"
                else:
                    sql_query = f"SELECT * FROM products WHERE category='{category}'"
                    if has_price:
                        sql_query += f" AND price BETWEEN {budget_range[0]} AND {budget_range[1]}"
                    sql_query += " ORDER BY rating DESC"
                
                parameters = {
                    "query": sql_query,
                    "limit": random.randint(15, 25)
                }
            
            sequence.append({
                "tool": tool,
                "purpose": purposes.get(tool, f"Execute {tool} operation"),
                "parameters": parameters
            })
        
        return sequence
    
    def generate_synthetic_task(self, scenario: str = None, 
                              force_tools: List[str] = None,
                              product_context: Dict[str, Any] = None,
                              use_real_sequences: bool = True) -> Dict[str, Any]:
        """Generate a single synthetic task using LLM"""
        
        # Select scenario and tools
        if scenario is None:
            scenario = random.choice(list(self.tool_combinations.keys()))
        
        # Use forced tools if provided, otherwise use scenario tools
        if force_tools:
            tools = force_tools
        else:
            tools = self.tool_combinations[scenario]

        # Get product context (use provided or generate new)
        if product_context is None:
            if use_real_sequences and ("get_user_history" in tools or "personalized" in scenario):
                product_context = self._get_user_sequence_context()
            else:
                product_context = self._get_sample_product_context()
        
        price_value = product_context.get('price')
        base_price = None
        try:
            if isinstance(price_value, (int, float)):
                base_price = float(price_value)
            else:
                price_str = str(price_value or '').replace('$', '').replace(',', '').strip()
                if price_str:
                    if '-' in price_str:
                        parts = [p.strip() for p in price_str.split('-')]
                        nums = [float(p) for p in parts if p]
                        if nums:
                            base_price = sum(nums) / len(nums)
                    else:
                        base_price = float(price_str)
        except Exception:
            base_price = None
        if base_price is not None:
            if base_price <= 10:
                lower = max(1, int(base_price * 0.5))
                upper = int(base_price * 1.5) + 5
            elif base_price <= 50:
                lower = max(5, int(base_price * 0.6))
                upper = int(base_price * 1.4) + 10
            else:
                lower = max(20, int(base_price * 0.7))
                upper = int(base_price * 1.3) + 20
            budget_range = (lower, upper)
        else:
            budget_range = (0, 0)
        
        print(f"Budget Range: {budget_range}, scenario: {scenario}")
        user_query = self._generate_llm_instruction(scenario, tools, product_context, budget_range)
        print(f"User Query: {user_query}")
        tool_sequence = self._infer_tool_sequence(tools, user_query, budget_range, product_context)

        sample_interactions = []
        for step in tool_sequence:
            tool_name = step['tool']
            params = step.get('parameters', {})
            if params:
                param_str = ', '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" 
                                 for k, v in list(params.items())[:3]])
                sample_interactions.append(f"{tool_name}({param_str})")
            else:
                sample_interactions.append(f"{tool_name}()")

        task = {
            "scenario": scenario,
            "required_tools": tools,
            "product_context": product_context,
            "task_description": f"LLM-generated retrieval task for {scenario}",
            "user_query": user_query,
            "expected_tool_sequence": tool_sequence,
            "sample_interactions": sample_interactions,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "estimated_steps": len(tools) + 1,
                "categories_involved": [product_context.get('category', 'Electronics')],
                "price_range": budget_range,
                "generation_method": "llm_driven",
                "uses_real_sequences": product_context.get("history_items") is not None
            }}
        return task
    def generate_task_batch(self, num_tasks: int, 
                          scenario_distribution: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic tasks"""
        
        if scenario_distribution is None:
            # Default distribution
            scenarios = list(self.tool_combinations.keys())
            scenario_distribution = {scenario: num_tasks // len(scenarios) for scenario in scenarios}
            # Distribute remainder
            remainder = num_tasks % len(scenarios)
            for i in range(remainder):
                scenario_distribution[scenarios[i]] += 1
        
        tasks = []
        
        for scenario, count in scenario_distribution.items():
            for _ in range(count):
                try:
                    task = self.generate_synthetic_task(scenario=scenario)
                    tasks.append(task)
                    print(f"Generated task: {task['task_id']} - {scenario}")
                except Exception as e:
                    print(f"Error generating task for scenario {scenario}: {e}")
                    continue
        
        return tasks
    
    def save_tasks_to_file(self, tasks: List[Dict[str, Any]], filename: str) -> str:
        """Save tasks to JSON file"""
        
        output_path = Path(__file__).parent / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(tasks)} synthetic tasks saved to: {output_path}")
        return str(output_path)
