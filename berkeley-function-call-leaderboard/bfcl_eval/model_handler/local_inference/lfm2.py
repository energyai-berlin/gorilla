from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override
import ast  
import json

# Note: This is the handler for the LFM2 in prompting mode. This model does not support function calls.


class LFM2Handler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype=dtype, model_max_len=16384, **kwargs)


    
    @override  
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:  
        functions: list = test_entry["function"]  
        # Skip BFCL's default - LFM2 needs its own format  
        return {"message": [], "function": functions}
    
    @override    
    def _format_prompt(self, messages, function):    
        formatted_prompt = "<|startoftext|>"    
        
        # Add system message with comprehensive instructions  
        formatted_prompt += "<|im_start|>system\n"  
        formatted_prompt += "You are a helpful assistant that can use tools.\n"  
        formatted_prompt += "When you need to use a tool, output it in this exact format:\n"  
        formatted_prompt += "<|tool_call_start|>[function_name(param1=\"value1\", param2=\"value2\")]<|tool_call_end|>\n"  
        formatted_prompt += "If no tool is needed, respond directly in plain text.\n\n"  
        
        if function and len(function) > 0:    
            tool_list = json.dumps(function)    
            formatted_prompt += f"List of tools: <|tool_list_start|>{tool_list}<|tool_list_end|>\n"  
        
        formatted_prompt += "<|im_end|>\n"    
        
        # Process remaining messages    
        for message in messages:    
            role = message["role"]    
            content = message["content"]    
            
            if role == "system":    
                continue    
                
            if role == "tool":    
                formatted_prompt += f"<|im_start|>tool\n<|tool_response_start|>{content}<|tool_response_end|><|im_end|>\n"    
            else:    
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"    
        
        formatted_prompt += "<|im_start|>assistant\n"    
        return formatted_prompt


    
