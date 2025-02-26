from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

def load_model_and_tokenizer(model_name):
    """加载大模型和对应的分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

def apply_template(messages, tokenizer):
    """应用对话模板格式化历史消息"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def chat_loop(model, tokenizer, max_new_tokens=4096, temperature=0.7, top_p=0.9):
    """支持流式输出的交互式对话循环"""
    messages = []
    
    print("开始对话（输入'exit'退出）")
    while True:
        try:
            user_input = input("\nUser: ")
        except KeyboardInterrupt:
            break
            
        if user_input.lower() == "exit":
            break
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 生成提示词
            prompt = apply_template(messages, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 创建流式生成器
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,    # 跳过输入提示
                skip_special_tokens=True  # 跳过特殊token
            )
            
            # 配置生成参数
            generate_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 启动生成线程
            thread = Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()
            
            # 实时流式输出
            response = []
            print("Assistant: ", end="", flush=True)
            for token in streamer:
                print(token, end="", flush=True)  # 逐词输出
                response.append(token)
            
            # 等待生成完成
            thread.join()
            print()  # 输出换行
            
            # 保存完整回复到对话历史
            messages.append({"role": "assistant", "content": "".join(response)})
            
        except Exception as e:
            print(f"\n生成错误: {str(e)}")
            messages.pop()

if __name__ == "__main__":
    MODEL_NAME = "/data03/irlab_share/deepseek/DeepSeek-R1-Distill-Qwen-14B"
    
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    chat_loop(
        model,
        tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95
    )