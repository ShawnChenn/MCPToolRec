#!/usr/bin/env python3
"""
Retrieval Task Synthesizer for Multi-Tool Recommendation Scenarios
Generates complex tasks that require combining multiple retrieval tools
"""

import asyncio
import json
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path

class RetrievalTaskSynthesizer:
    """Synthesizes complex retrieval tasks combining multiple tools"""
    
    def __init__(self):
        self.user_personas = self._create_user_personas()
        self.scenario_templates = self._create_scenario_templates()
        self.tool_combinations = self._create_tool_combinations()
        
    def _create_user_personas(self) -> List[Dict[str, Any]]:
        """Create diverse user personas for realistic scenarios"""
        return [
            {
                "id": "tech_enthusiast",
                "name": "Alex Chen",
                "description": "Tech enthusiast who loves cutting-edge gadgets",
                "preferences": ["high-performance", "latest technology", "premium brands"],
                "budget_range": [500, 3000],
                "categories": ["Electronics", "Computers"],
                "brands": ["Apple", "Samsung", "Sony"],
                "interaction_history": ["smartphone", "laptop", "headphones", "tablet"]
            },
            {
                "id": "budget_conscious",
                "name": "Maria Rodriguez",
                "description": "Budget-conscious student looking for value",
                "preferences": ["affordable", "good value", "reliable"],
                "budget_range": [50, 500],
                "categories": ["Electronics", "Computers", "Audio"],
                "brands": ["Dell", "Lenovo", "Xiaomi"],
                "interaction_history": ["budget laptop", "wireless earbuds", "phone case"]
            },
            {
                "id": "professional",
                "name": "David Kim",
                "description": "Business professional needing productivity tools",
                "preferences": ["productivity", "professional", "reliable", "business-grade"],
                "budget_range": [800, 2500],
                "categories": ["Computers", "Electronics"],
                "brands": ["Apple", "Dell", "Microsoft"],
                "interaction_history": ["MacBook", "business laptop", "wireless mouse", "monitor"]
            },
            {
                "id": "creative_artist",
                "name": "Sophie Turner",
                "description": "Creative professional working in design and media",
                "preferences": ["creative tools", "high-quality display", "performance"],
                "budget_range": [1000, 4000],
                "categories": ["Computers", "Electronics"],
                "brands": ["Apple", "Adobe", "Wacom"],
                "interaction_history": ["MacBook Pro", "graphics tablet", "4K monitor", "color calibrator"]
            },
            {
                "id": "gamer",
                "name": "Jake Wilson",
                "description": "Gaming enthusiast looking for high-performance gear",
                "preferences": ["gaming", "high-performance", "RGB", "mechanical"],
                "budget_range": [300, 2000],
                "categories": ["Computers", "Electronics", "Gaming"],
                "brands": ["NVIDIA", "AMD", "Razer", "Corsair"],
                "interaction_history": ["gaming laptop", "mechanical keyboard", "gaming mouse", "headset"]
            }
        ]
    
    def _create_scenario_templates(self) -> List[Dict[str, Any]]:
        """Create scenario templates for different recommendation contexts"""
        return [
            {
                "name": "similar_item_exploration",
                "description": "User wants to explore items similar to something they liked",
                "tools": ["item_to_item_similarity", "search_products_by_keywords"],
                "complexity": "medium",
                "template": "I really liked {reference_item}. Can you find similar products and also search for {related_keywords}?"
            },
            {
                "name": "personalized_discovery",
                "description": "Personalized recommendations based on user history",
                "tools": ["get_user_history", "vector_similarity_search"],
                "complexity": "high",
                "template": "Based on my purchase history, can you recommend products that match my interests in {interest_area}?"
            },
            {
                "name": "category_exploration_with_similarity",
                "description": "Explore a category and find similar items to favorites",
                "tools": ["search_products_by_keywords", "item_to_item_similarity", "get_product_categories"],
                "complexity": "high",
                "template": "Show me {category} products under ${budget}, then find items similar to the highest-rated one."
            },
            {
                "name": "brand_comparison_with_history",
                "description": "Compare brands while considering user preferences",
                "tools": ["search_products_by_keywords", "get_user_history", "get_product_brands"],
                "complexity": "high",
                "template": "Compare {brand1} vs {brand2} products in {category}, considering my past purchases."
            },
            {
                "name": "semantic_search_with_filtering",
                "description": "Semantic search combined with structured filtering",
                "tools": ["vector_similarity_search", "sql_query_products"],
                "complexity": "medium",
                "template": "Find products that match '{semantic_query}' and also meet these specific criteria: {sql_criteria}."
            },
            {
                "name": "comprehensive_recommendation",
                "description": "Multi-step recommendation using multiple approaches",
                "tools": ["get_user_history", "search_products_by_keywords", "item_to_item_similarity", "vector_similarity_search"],
                "complexity": "very_high",
                "template": "I need a comprehensive recommendation for {category}. Consider my history, search for {keywords}, find similar items, and use semantic matching for '{semantic_query}'."
            },
            {
                "name": "budget_optimization",
                "description": "Find best value products within budget constraints",
                "tools": ["search_products_by_keywords", "sql_query_products", "item_to_item_similarity"],
                "complexity": "high",
                "template": "Find the best {category} products under ${budget} with rating above {min_rating}, then show similar alternatives."
            },
            {
                "name": "trend_analysis_with_personalization",
                "description": "Analyze trends while considering personal preferences",
                "tools": ["get_product_categories", "vector_similarity_search", "get_user_history"],
                "complexity": "high",
                "template": "What are the trending products in {category} that would suit someone with my preferences?"
            }
        ]
    
    def _create_tool_combinations(self) -> List[Dict[str, Any]]:
        """Define specific tool combinations and their use cases"""
        return [
            {
                "tools": ["search_products_by_keywords", "item_to_item_similarity"],
                "use_case": "Search then find similar",
                "description": "Search for products, then find items similar to the best results"
            },
            {
                "tools": ["get_user_history", "vector_similarity_search"],
                "use_case": "Personalized semantic search",
                "description": "Use user history to inform semantic search queries"
            },
            {
                "tools": ["search_products_by_keywords", "sql_query_products"],
                "use_case": "Keyword search with complex filtering",
                "description": "Combine keyword search with advanced SQL filtering"
            },
            {
                "tools": ["get_product_categories", "search_products_by_keywords", "item_to_item_similarity"],
                "use_case": "Category exploration with similarity",
                "description": "Explore categories, search within them, then find similar items"
            },
            {
                "tools": ["vector_similarity_search", "get_user_history", "add_user_interaction"],
                "use_case": "Semantic search with learning",
                "description": "Semantic search that learns from user interactions"
            },
            {
                "tools": ["sql_query_products", "item_to_item_similarity", "vector_similarity_search"],
                "use_case": "Multi-modal recommendation",
                "description": "Combine structured queries, similarity, and semantic search"
            }
        ]
    
    def generate_synthetic_task(self, scenario_name: str = None, user_persona: str = None) -> Dict[str, Any]:
        """Generate a single synthetic task"""
        
        # Select scenario and persona
        if scenario_name:
            scenario = next((s for s in self.scenario_templates if s["name"] == scenario_name), None)
            if not scenario:
                scenario = random.choice(self.scenario_templates)
        else:
            scenario = random.choice(self.scenario_templates)
        
        if user_persona:
            persona = next((p for p in self.user_personas if p["id"] == user_persona), None)
            if not persona:
                persona = random.choice(self.user_personas)
        else:
            persona = random.choice(self.user_personas)
        
        # Generate task details based on scenario and persona
        task_details = self._generate_task_details(scenario, persona)
        
        # Create the complete task
        task = {
            "task_id": f"retrieval_{scenario['name']}_{persona['id']}_{random.randint(1000, 9999)}",
            "scenario": scenario["name"],
            "user_persona": persona["id"],
            "complexity": scenario["complexity"],
            "required_tools": scenario["tools"],
            "user_context": {
                "name": persona["name"],
                "description": persona["description"],
                "preferences": persona["preferences"],
                "budget_range": persona["budget_range"],
                "interaction_history": persona["interaction_history"]
            },
            "task_description": task_details["description"],
            "user_query": task_details["query"],
            "expected_tool_sequence": task_details["tool_sequence"],
            "success_criteria": task_details["success_criteria"],
            "sample_interactions": task_details.get("sample_interactions", []),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "difficulty_level": self._calculate_difficulty(scenario, len(scenario["tools"])),
                "estimated_steps": len(scenario["tools"]) + random.randint(1, 3),
                "categories_involved": task_details.get("categories", []),
                "price_range": task_details.get("price_range", persona["budget_range"])
            }
        }
        
        return task
    
    def _generate_task_details(self, scenario: Dict[str, Any], persona: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific task details based on scenario and persona"""
        
        details = {}
        
        if scenario["name"] == "similar_item_exploration":
            reference_item = random.choice(persona["interaction_history"])
            related_keywords = " ".join(random.sample(persona["preferences"], 2))
            
            details = {
                "description": f"Help {persona['name']} find products similar to {reference_item} and explore related options",
                "query": f"I really liked the {reference_item} I bought recently. Can you find similar products and also search for {related_keywords} options in my price range?",
                "tool_sequence": [
                    {
                        "tool": "search_products_by_keywords",
                        "purpose": "Find products matching the reference item and keywords",
                        "parameters": {
                            "keywords": f"{reference_item} {related_keywords}",
                            "min_price": persona["budget_range"][0],
                            "max_price": persona["budget_range"][1],
                            "limit": 10
                        }
                    },
                    {
                        "tool": "item_to_item_similarity", 
                        "purpose": "Find items similar to the best search result",
                        "parameters": {
                            "product_id": "from_previous_search",
                            "limit": 8,
                            "threshold": 0.2
                        }
                    }
                ],
                "success_criteria": [
                    "Successfully found products matching the reference item",
                    "Identified similar alternatives using similarity search",
                    "Results are within user's budget range",
                    "Recommendations align with user preferences"
                ],
                "sample_interactions": [
                    f"add_user_interaction('{persona['id']}', product_id, 'view')",
                    f"add_user_interaction('{persona['id']}', similar_product_id, 'like')"
                ]
            }
            
        elif scenario["name"] == "personalized_discovery":
            interest_area = random.choice(persona["categories"])
            
            details = {
                "description": f"Provide personalized recommendations for {persona['name']} based on their history",
                "query": f"Based on my purchase history, can you recommend {interest_area.lower()} products that match my interests? I'm particularly interested in {random.choice(persona['preferences'])} options.",
                "tool_sequence": [
                    {
                        "tool": "get_user_history",
                        "purpose": "Analyze user's past interactions and preferences",
                        "parameters": {
                            "user_id": persona["id"],
                            "limit": 20
                        }
                    },
                    {
                        "tool": "vector_similarity_search",
                        "purpose": "Find products matching user's interest profile",
                        "parameters": {
                            "query": f"{interest_area} {' '.join(persona['preferences'][:3])}",
                            "limit": 10,
                            "threshold": 0.15
                        }
                    }
                ],
                "success_criteria": [
                    "Retrieved and analyzed user's interaction history",
                    "Generated personalized recommendations using semantic search",
                    "Recommendations reflect user's past preferences",
                    "Results are relevant to the specified interest area"
                ],
                "sample_interactions": [
                    f"get_user_history('{persona['id']}', limit=20)",
                    "vector_similarity_search based on history analysis"
                ]
            }
            
        elif scenario["name"] == "category_exploration_with_similarity":
            category = random.choice(persona["categories"])
            budget = random.randint(persona["budget_range"][0], persona["budget_range"][1])
            
            details = {
                "description": f"Help {persona['name']} explore {category} products and find similar alternatives",
                "query": f"Show me {category.lower()} products under ${budget}, then find items similar to the highest-rated one. I prefer {random.choice(persona['preferences'])} options.",
                "tool_sequence": [
                    {
                        "tool": "get_product_categories",
                        "purpose": "Confirm available categories",
                        "parameters": {}
                    },
                    {
                        "tool": "search_products_by_keywords",
                        "purpose": "Find products in the specified category and budget",
                        "parameters": {
                            "keywords": category,
                            "category": category,
                            "max_price": budget,
                            "min_rating": 4.0,
                            "limit": 15
                        }
                    },
                    {
                        "tool": "item_to_item_similarity",
                        "purpose": "Find alternatives to the top-rated product",
                        "parameters": {
                            "product_id": "highest_rated_from_search",
                            "limit": 10,
                            "threshold": 0.25
                        }
                    }
                ],
                "success_criteria": [
                    "Successfully explored the specified category",
                    "Found products within budget constraints",
                    "Identified the highest-rated product",
                    "Generated similar product recommendations"
                ],
                "sample_interactions": [
                    f"get_product_categories()",
                    f"search_products_by_keywords('{category}', max_price={budget})",
                    f"item_to_item_similarity(product_id, limit=10)"
                ],
                "categories": [category],
                "price_range": [0, budget]
            }
            
        elif scenario["name"] == "comprehensive_recommendation":
            category = random.choice(persona["categories"])
            keywords = " ".join(random.sample(persona["preferences"], 2))
            semantic_query = f"high-quality {category.lower()} for {persona['description'].split()[0]}"
            
            details = {
                "description": f"Comprehensive recommendation system for {persona['name']}",
                "query": f"I need a comprehensive recommendation for {category.lower()}. Consider my history, search for {keywords}, find similar items, and use semantic matching for '{semantic_query}'.",
                "tool_sequence": [
                    {
                        "tool": "get_user_history",
                        "purpose": "Understand user's preferences and patterns",
                        "parameters": {
                            "user_id": persona["id"],
                            "limit": 15
                        }
                    },
                    {
                        "tool": "search_products_by_keywords",
                        "purpose": "Find products matching explicit keywords",
                        "parameters": {
                            "keywords": keywords,
                            "category": category,
                            "min_price": persona["budget_range"][0],
                            "max_price": persona["budget_range"][1],
                            "limit": 12
                        }
                    },
                    {
                        "tool": "vector_similarity_search",
                        "purpose": "Semantic matching for nuanced preferences",
                        "parameters": {
                            "query": semantic_query,
                            "limit": 10,
                            "threshold": 0.2
                        }
                    },
                    {
                        "tool": "item_to_item_similarity",
                        "purpose": "Find alternatives to promising candidates",
                        "parameters": {
                            "product_id": "best_match_from_previous_tools",
                            "limit": 8,
                            "threshold": 0.3
                        }
                    }
                ],
                "success_criteria": [
                    "Analyzed user history for personalization",
                    "Found products matching explicit keywords",
                    "Applied semantic search for nuanced matching",
                    "Generated similarity-based alternatives",
                    "Provided comprehensive, multi-faceted recommendations"
                ],
                "sample_interactions": [
                    f"get_user_history('{persona['id']}', limit=15)",
                    f"search_products_by_keywords('{keywords}', category='{category}')",
                    f"vector_similarity_search('{semantic_query}')",
                    f"item_to_item_similarity(product_id, limit=8)"
                ],
                "categories": [category]
            }
        
        # Add default details if not specified
        if not details:
            details = {
                "description": f"Generic task for {persona['name']} using {scenario['name']}",
                "query": f"Help me with {scenario['description'].lower()}",
                "tool_sequence": [{"tool": tool, "purpose": f"Use {tool}"} for tool in scenario["tools"]],
                "success_criteria": ["Complete the task successfully"],
                "sample_interactions": [f"{tool}()" for tool in scenario["tools"]]
            }
        
        return details
    
    def _calculate_difficulty(self, scenario: Dict[str, Any], num_tools: int) -> str:
        """Calculate task difficulty based on scenario complexity and number of tools"""
        complexity_scores = {
            "low": 1,
            "medium": 2, 
            "high": 3,
            "very_high": 4
        }
        
        base_score = complexity_scores.get(scenario["complexity"], 2)
        tool_score = min(num_tools, 4)  # Cap at 4
        
        total_score = base_score + (tool_score - 1)
        
        if total_score <= 2:
            return "Easy"
        elif total_score <= 4:
            return "Medium"
        elif total_score <= 6:
            return "Hard"
        else:
            return "Expert"
    
    def generate_task_batch(self, num_tasks: int = 20, 
                           scenario_distribution: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic tasks"""
        
        if scenario_distribution is None:
            # Default distribution
            scenario_distribution = {
                "similar_item_exploration": 4,
                "personalized_discovery": 4,
                "category_exploration_with_similarity": 3,
                "comprehensive_recommendation": 3,
                "brand_comparison_with_history": 2,
                "semantic_search_with_filtering": 2,
                "budget_optimization": 2
            }
        
        tasks = []
        
        for scenario_name, count in scenario_distribution.items():
            for _ in range(count):
                if len(tasks) >= num_tasks:
                    break
                    
                task = self.generate_synthetic_task(scenario_name=scenario_name)
                tasks.append(task)
        
        # Fill remaining slots with random tasks
        while len(tasks) < num_tasks:
            task = self.generate_synthetic_task()
            tasks.append(task)
        
        return tasks[:num_tasks]
    
    def save_tasks_to_file(self, tasks: List[Dict[str, Any]], filename: str = None):
        """Save generated tasks to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retrieval_synthetic_tasks_{timestamp}.json"
        
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tasks": len(tasks),
                "task_types": list(set(task["scenario"] for task in tasks)),
                "user_personas": list(set(task["user_persona"] for task in tasks)),
                "difficulty_distribution": {
                    level: len([t for t in tasks if t["metadata"]["difficulty_level"] == level])
                    for level in ["Easy", "Medium", "Hard", "Expert"]
                }
            },
            "tasks": tasks
        }
        
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {len(tasks)} synthetic tasks saved to: {filepath.absolute()}")
        return filepath

async def main():
    """Main function to generate and save synthetic tasks"""
    
    print("Retrieval Task Synthesizer")
    print("=" * 50)
    
    synthesizer = RetrievalTaskSynthesizer()
    
    # Generate different types of task batches
    
    # 1. Small focused batch for testing
    print("\n1. Generating focused test batch (10 tasks)...")
    test_tasks = synthesizer.generate_task_batch(10, {
        "similar_item_exploration": 3,
        "personalized_discovery": 3,
        "category_exploration_with_similarity": 2,
        "comprehensive_recommendation": 2
    })
    
    test_file = synthesizer.save_tasks_to_file(test_tasks, "retrieval_test_tasks.json")
    
    # 2. Comprehensive batch with all scenarios
    print("\n2. Generating comprehensive batch (30 tasks)...")
    comprehensive_tasks = synthesizer.generate_task_batch(30)
    comp_file = synthesizer.save_tasks_to_file(comprehensive_tasks, "retrieval_comprehensive_tasks.json")
    
    # 3. Show sample tasks
    print("\n3. Sample Generated Tasks:")
    print("-" * 30)
    
    for i, task in enumerate(test_tasks[:3]):
        print(f"\nTask {i+1}: {task['task_id']}")
        print(f"Scenario: {task['scenario']}")
        print(f"User: {task['user_context']['name']} ({task['user_persona']})")
        print(f"Complexity: {task['complexity']} -> Difficulty: {task['metadata']['difficulty_level']}")
        print(f"Tools: {', '.join(task['required_tools'])}")
        print(f"Query: {task['user_query'][:100]}...")
        print(f"Steps: {len(task['expected_tool_sequence'])}")
    
    print(f"\n" + "=" * 50)
    print("Task generation completed!")
    print(f"Test batch: {test_file}")
    print(f"Comprehensive batch: {comp_file}")

if __name__ == "__main__":
    asyncio.run(main())