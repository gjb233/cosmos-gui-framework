import importlib.util,importlib.metadata
for m in ('vllm','transformers','peft','openai','qwen_vl_utils','safetensors'):
 try:print(m,importlib.util.find_spec(m) is not None,importlib.metadata.version(m))
 except Exception as e:print(m,str(e))
