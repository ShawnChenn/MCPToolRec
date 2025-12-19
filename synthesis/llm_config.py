#!/usr/bin/env python3
"""
Configuration for LLM-Driven Retrieval Task Synthesis
"""

import os
from pathlib import Path

class LLMSynthesisConfig:
    """Configuration settings for LLM synthesis"""
    
    POSSIBLE_MODEL_PATHS = [
        "/data/zhendong_data/cx/ReCall/gpt-oss-20b",
        "/data/cx/Toolrec/LLM-SRec-master/Qwen3-4B-Instruct-2507",
        "/apdcephfs/share_303747097/huggingface_model/gpt-oss-20b"
    ]
    
    # Data paths - try multiple possible locations
    POSSIBLE_DATA_PATHS = [
        "/data/zhendong_data/cx/ReCall/data/amazon-electronics",
    ]
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get the first available model path"""
        for path in cls.POSSIBLE_MODEL_PATHS:
            if os.path.exists(path):
                return path
        
        # If no model found, return the first path (will be handled by the synthesizer)
        return cls.POSSIBLE_MODEL_PATHS[0]
    
    @classmethod
    def get_data_path(cls) -> str:
        """Get the first available data path"""
        for path in cls.POSSIBLE_DATA_PATHS:
            if os.path.exists(path):
                # Check if required files exist
                meta_file = os.path.join(path, 'item_meta_dict.json')
                index_file = os.path.join(path, 'index_to_asin.json')
                if os.path.exists(meta_file) and os.path.exists(index_file):
                    return path
        
        # If no data found, return the first path
        return cls.POSSIBLE_DATA_PATHS[0]
    
    @classmethod
    def check_dependencies(cls) -> dict:
        """Check if required dependencies are available"""
        status = {
            "torch": False,
            "transformers": False,
            "model_path": False,
            "data_path": False
        }
        
        try:
            import torch
            status["torch"] = True
        except ImportError:
            pass
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            status["transformers"] = True
        except ImportError:
            pass
        
        model_path = cls.get_model_path()
        status["model_path"] = os.path.exists(model_path)
        
        data_path = cls.get_data_path()
        status["data_path"] = os.path.exists(data_path)
        
        return status
    
    @classmethod
    def print_status(cls):
        """Print configuration status"""
        print("🔧 LLM Synthesis Configuration Status")
        print("=" * 40)
        
        status = cls.check_dependencies()
        
        print(f"📦 Dependencies:")
        print(f"   torch: {'✅' if status['torch'] else '❌'}")
        print(f"   transformers: {'✅' if status['transformers'] else '❌'}")
        
        print(f"\n📁 Paths:")
        model_path = cls.get_model_path()
        print(f"   Model: {'✅' if status['model_path'] else '❌'} {model_path}")
        
        data_path = cls.get_data_path()
        print(f"   Data: {'✅' if status['data_path'] else '❌'} {data_path}")
        
        if status['data_path']:
            # Show data statistics
            try:
                import json
                meta_file = os.path.join(data_path, 'item_meta_dict.json')
                with open(meta_file, 'r') as f:
                    meta_dict = json.load(f)
                print(f"   Products: {len(meta_dict)} items loaded")
            except Exception as e:
                print(f"   Products: Error loading - {e}")
        
        all_ready = all(status.values())
        print(f"\n🎯 Ready for LLM Synthesis: {'✅' if all_ready else '❌'}")
        
        if not all_ready:
            print("\n💡 Missing Requirements:")
            if not status['torch']:
                print("   - Install PyTorch: pip install torch")
            if not status['transformers']:
                print("   - Install Transformers: pip install transformers")
            if not status['model_path']:
                print(f"   - Download GPT-OSS model to one of: {cls.POSSIBLE_MODEL_PATHS}")
            if not status['data_path']:
                print(f"   - Prepare product data in one of: {cls.POSSIBLE_DATA_PATHS}")
        
        return all_ready

if __name__ == "__main__":
    LLMSynthesisConfig.print_status()