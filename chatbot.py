from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "EleutherAI/gpt-j-6B"  # GPT-J 6B is more powerful than DialoGPT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to get chatbot response
def get_response(user_input, chat_history_ids=None):
    # Encode user input and append end-of-sequence token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Append new input to chat history (if available)
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    
    # Generate response with improved hyperparameters
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        top_k=50,
        temperature=0.7
    )
    
    # Decode response to text and clean it up
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    response = response.replace("\n", " ").strip()  # Clean the output
    return response, chat_history_ids
