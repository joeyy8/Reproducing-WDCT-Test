import torch
from peft import PeftModel, LoraConfig
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

class LocalModel(torch.nn.Module):
    def __init__(self, model_name, model_path, ckpt_model_path, device, paras, gpu_id=None):
        super(LocalModel, self).__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.ckpt_model_path = ckpt_model_path
        self.device = device
        self.paras = paras
        self.gpu_id = gpu_id
        self.padding_side = "left"
        self.max_sequence_length = 2048
        self.peft_model = False
        if ckpt_model_path and ('lora' in ckpt_model_path.lower() or 'dpo' in ckpt_model_path.lower()):
            self.peft_model = True

        # 加载模型和tokenizer
        self.model, self.tokenizer = self.load()

    def load(self):
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.padding_side = self.padding_side
        # 为ChatGLM模型设置填充token
        if "chatglm" in self.model_name.lower():
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"设置tokenizer.pad_token_id为eos_token_id: {tokenizer.eos_token_id}")
        else:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                print(f"设置pad_token错误: {e}")
                if tokenizer.unk_token:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token

        # 加载基础模型
        if self.gpu_id:
            device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                trust_remote_code=True
            )
            model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                device_map="auto",
                trust_remote_code=True
            )

        # 加载LoRA/DPO模型
        if self.ckpt_model_path:
            if self.peft_model:
                try:
                    # 尝试加载LoRA模型
                    model = PeftModel.from_pretrained(model, self.ckpt_model_path)
                    print(f"成功加载LoRA模型: {self.ckpt_model_path}")
                except Exception as e:
                    print(f"加载LoRA模型失败: {e}")
                    # 检查adapter_config.json
                    config_path = os.path.join(self.ckpt_model_path, 'adapter_config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        print(f"adapter_config.json内容: {config}")
                        # 移除无效参数
                        config.pop('corda_config', None)
                        # 使用默认LoRA配置
                        lora_config = LoraConfig(
                            r=config.get('r', 8),
                            lora_alpha=config.get('lora_alpha', 32),
                            lora_dropout=config.get('lora_dropout', 0.1),
                            target_modules=config.get('target_modules', ["query_key_value"]),
                            task_type=config.get('task_type', "CAUSAL_LM"),
                            bias=config.get('bias', "none")
                        )
                        model = PeftModel.from_pretrained(model, self.ckpt_model_path, config=lora_config)
                        print(f"使用默认LoRA配置重新加载模型: {self.ckpt_model_path}")
            else:
                # 加载普通checkpoint
                state_dict = torch.load(self.ckpt_model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict['state'])
                print(f"成功加载普通模型: {self.ckpt_model_path}")

        model.eval()
        print(f"模型加载完成: {self.model_name}, 设备: {self.device}")
        return model, tokenizer

    def get_single_prompt(self, message):
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def answer(self, message, max_new_tokens=100, use_prompt_template=True, paras={}):
        self.paras.update(paras)
        messages = [message] if isinstance(message, str) else message
        if use_prompt_template:
            prompts = [self.get_single_prompt(msg) for msg in messages]
        else:
            prompts = messages

        # 处理输入
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        max_input_len = self.max_sequence_length - (max_new_tokens + 100)
        if inputs['input_ids'].shape[1] > max_input_len:
            print(f"截断输入从 {inputs['input_ids'].shape[1]} 到 {max_input_len}")
            for key, value in inputs.items():
                inputs[key] = value[:, :max_input_len]

        # 生成回答
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "output_scores": True,
            "return_dict_in_generate": True
        }
        if self.paras.get('temperature', 0) > 0:
            generate_kwargs.update({
                "do_sample": True,
                "top_p": self.paras.get('top_p', 0.9),
                "temperature": self.paras['temperature']
            })
        else:
            generate_kwargs.update({
                "do_sample": False,
                "temperature": None,
                "top_p": None
            })

        output = model.generate(**inputs, **generate_kwargs)

        # 解析结果
        output_ids = output.sequences
        scores = output.scores
        results = []
        probs = []
        for i in range(len(output_ids)):
            # 提取生成的部分
            single_output_ids = output_ids[i][len(inputs["input_ids"][i]):]
            outputs = self.tokenizer.decode(
                single_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            results.append(outputs)

            # 计算A/B的概率
            if scores:
                first_token_score = scores[0][i]
                a_token, b_token = 'A', 'B'
                a_idx = self.tokenizer.convert_tokens_to_ids(a_token)
                b_idx = self.tokenizer.convert_tokens_to_ids(b_token)
                a_score = first_token_score[a_idx].item() if a_idx != self.tokenizer.unk_token_id else 0
                b_score = first_token_score[b_idx].item() if b_idx != self.tokenizer.unk_token_id else 0
                probs.append([a_score, b_score])
                print(f"调试: Token IDs: A={a_idx}, B={b_idx}, Scores: A={a_score}, B={b_score}")
            else:
                probs.append([0, 0])
                print("调试: 未获取到scores")

        return results, probs