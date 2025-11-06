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
        
        # Add system message with LFM2's tool list format  
        formatted_prompt += "<|im_start|>system\n"  
        if function and len(function) > 0:  
            tool_list = json.dumps(function)  
            formatted_prompt += f"List of tools: <|tool_list_start|>{tool_list}<|tool_list_end|>"  
        formatted_prompt += "<|im_end|>\n"  
        
        # Process remaining messages  
        for message in messages:  
            role = message["role"]  
            content = message["content"]  
            
            # Skip system messages (already handled)  
            if role == "system":  
                continue  
                
            # Handle tool responses with special tags  
            if role == "tool":  
                formatted_prompt += f"<|im_start|>tool\n<|tool_response_start|>{content}<|tool_response_end|><|im_end|>\n"  
            else:  
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"  
        
        formatted_prompt += "<|im_start|>assistant\n"  
        return formatted_prompt
    


    
    @override  
    def decode_execute(self, result, has_tool_call_tag):  
        # Similar extraction logic  
        if "<|tool_call_start|>" not in result:  
            return []  
        
        tool_call_str = result.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip()  
        tool_call_str = tool_call_str.strip("[]")  
        
        # Return as executable string  
        return [tool_call_str]
    
    @override  
    def _add_execution_results_prompting(  
        self, inference_data: dict, execution_results: list[str], model_response_data: dict  
    ) -> dict:  
        for execution_result in execution_results:  
            inference_data["message"].append({  
                "role": "tool",  
                "content": f"<|tool_response_start|>{execution_result}<|tool_response_end|>"  
            })  
        return inference_data
    

