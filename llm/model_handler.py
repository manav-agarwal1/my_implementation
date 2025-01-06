# model_handler.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import warnings
from transformers import logging

class LLMHandler:
    def __init__(self):
        self.debug = True
        try:
            warnings.filterwarnings("ignore", category=FutureWarning)
            logging.set_verbosity_error()
            
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise

    def generate_response(self, question, context):
        # Define the example JSON separately with proper escaping
        example_json = '''{
    "filters": [
        {
            "filter_type": "CURRENT_COMPANY",
            "type": "in",
            "value": ["example_company"]
        },
        {
            "filter_type": "CURRENT_TITLE",
            "type": "in",
            "value": ["example_title"]
        },
        {
            "filter_type": "REGION",
            "type": "in",
            "value": ["example_location"]
        }
    ],
    "page": 1
}'''

        prompt = f"""You are an API documentation expert. Based on the provided API documentation, generate a response that shows how to use the API endpoint with a specific example.

API Documentation:
{context}

Question: {question}

Generate a response in this format:
To search for people, use the api.crustdata.com/screener/person/search endpoint. Here's an example curl request:

curl --location 'https://api.crustdata.com/screener/person/search' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Token $token' \\
--data '{example_json}'

Response:"""

        try:
            response = self.pipeline(prompt)[0]['generated_text']
            
            # Extract the response part
            if "Response:" in response:
                answer = response.split("Response:")[-1].strip()
            else:
                answer = response.split("Question:")[-1].split("To search for")[1].strip()
            
            if not answer or answer.isspace():
                return f"""To search for people, use the api.crustdata.com/screener/person/search endpoint. Here's an example curl request:

curl --location 'https://api.crustdata.com/screener/person/search' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Token $token' \\
--data '{example_json}'"""
            
            return answer
            
        except Exception as e:
            if self.debug:
                print(f"Error generating response: {str(e)}")
            return "Error generating response. Please try again."