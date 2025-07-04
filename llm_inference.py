import os
import json
import ollama
import requests

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
API_URL = "https://api.sambanova.ai/v1/chat/completions"

class LLMInference:
    
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"
    
    def _generate_embeddings(self, input_text: str, model_name: str):
        # print('model_name : ', model_name)
        return ollama.embeddings(model=model_name, prompt=input_text).get("embedding", [])
    
    def generate_text_ollama(self, prompt: str, model_name: str):
        if not prompt or not model_name:
            return "Error: Both 'prompt' and 'model_name' are required", 0, 0

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            generated_text = response.get("message", {}).get("content", "Error: No content in response")
            # Extract token counts if available, default to 0 if not
            input_tokens = response.get("prompt_eval_count", 0)  # Ollama might return this
            output_tokens = response.get("eval_count", 0)  # Ollama might return this
            return generated_text, input_tokens, output_tokens
        except Exception as e:
            return f"Error: Failed to generate response using Ollama: {str(e)}", 0, 0
    
    def generate_text_sambanova(self, prompt: str, model_name: str):
        if not prompt or not model_name:
            return "Error: Both 'prompt' and 'model_name' are required", 0, 0

        try:
            response = requests.post(
                url=API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                })
            )

            response_data = response.json()
            print('response_data : ', response_data)
            # Check if there's an error key in the response data
            if "error" in response_data:
                error_message = response_data["error"].get("message", "Unknown error")
                return f"Error: API error: {error_message}", 0, 0
            
            generated_text = response_data["choices"][0]["message"]["content"]
            # Extract token counts
            total_tokens = response_data["usage"].get("prompt_tokens", 0)
            output_tokens = response_data["usage"].get("completion_tokens", 0)
            return generated_text, total_tokens, output_tokens
        except Exception as e:
            return f"Error generating response: {str(e)}", 0, 0

    def generate_text(self, prompt: str, model_name: str, llm_provider: str):
        if llm_provider == 'Ollama':
            return self.generate_text_ollama(prompt, model_name)
        elif llm_provider == 'Sambanova':
            return self.generate_text_sambanova(prompt, model_name)
        else:
            return f"Error: LLM provider '{llm_provider}' is not supported. Choose either 'Ollama' or 'Sambanova'", 0, 0

if __name__ == "__main__":
    llm = LLMInference()
    prompt = "What is the capital of France?"
    
    # response = llm._generate_embeddings(prompt, model_name="mxbai-embed-large:latest")
    response = llm.generate_text(prompt, model_name="QwQ-32B", llm_provider='Sambanova')
    print(response)
