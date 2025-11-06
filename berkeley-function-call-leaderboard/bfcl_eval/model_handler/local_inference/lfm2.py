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
    
    @override  
    def decode_ast(self, result, language, has_tool_call_tag):  
        # Extract tool calls from LFM2's format  
        if "<|tool_call_start|>" not in result:  
            return []  
        
        tool_call_str = result.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip()  
        
        # Parse Python-style function calls: [func_name(param="value")]  
        # Remove outer brackets if present  
        tool_call_str = tool_call_str.strip("[]")  
        
        # Use ast.parse to safely parse the Python function call  
        try:  
            parsed = ast.parse(tool_call_str, mode='eval')  
            decoded_output = []  
            
            # Handle single function call  
            if isinstance(parsed.body, ast.Call):  
                from bfcl_eval.model_handler.utils import resolve_ast_call  
                decoded_output.append(resolve_ast_call(parsed.body))  
            # Handle multiple function calls (list) - ADD TYPE CHECK HERE  
            elif isinstance(parsed.body, ast.List):  
                for elem in parsed.body.elts:  
                    if isinstance(elem, ast.Call):  
                        from bfcl_eval.model_handler.utils import resolve_ast_call  
                        decoded_output.append(resolve_ast_call(elem))  
            else:  
                raise ValueError(f"Unexpected AST node type: {type(parsed.body)}")  
            
            return decoded_output  
        except SyntaxError as e:  
            raise ValueError(f"Failed to parse tool call: {tool_call_str}. Error: {str(e)}")
    

