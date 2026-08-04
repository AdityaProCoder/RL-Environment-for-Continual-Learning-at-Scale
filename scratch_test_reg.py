import urllib3, requests
urllib3.disable_warnings()
old_send = requests.Session.send
requests.Session.send = lambda self, request, **kwargs: old_send(self, request, **dict(kwargs, verify=False))

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

class Qwen35Config(Qwen2Config):
    model_type = "qwen3_5"
    def get_text_config(self, *args, **kwargs):
        return self

class Qwen35ForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen35Config

AutoConfig.register("qwen3_5", Qwen35Config)
AutoModelForCausalLM.register(Qwen35Config, Qwen35ForCausalLM)

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
print("SUCCESS Qwen3.5-2B config:", type(cfg), cfg.model_type)
print("get_text_config type:", type(cfg.get_text_config()), "has to_dict:", hasattr(cfg.get_text_config(), "to_dict"))

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", torch_dtype="float32", low_cpu_mem_usage=True)
print("SUCCESS Qwen3.5-2B model loaded perfectly:", type(model))
