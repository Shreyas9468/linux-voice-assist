import json
import logging
import google.generativeai as genai
from llms.llm_base import LLMBase
from typing import Optional

class GeminiService(LLMBase):
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def initialize(self):
        genai.configure(api_key=self.api_key)
        logging.info("Gemini API initialized")
    
    def generate_bash_script(self, query: str, context: Optional[str] = None) -> tuple[str, str]:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Modify prompt to include context if available
        from config import PROMPT_TEMPLATE
        if context:
            enhanced_prompt = f"""Context information:
{context}

Based on the above context and the following query and also use your knowledge, generate a bash script:
{PROMPT_TEMPLATE.format(query=query)}"""
        else:
            enhanced_prompt = PROMPT_TEMPLATE.format(query=query)
            
        response = model.generate_content(enhanced_prompt)
        
        try:
            jsondata = json.loads(response.text)
            bash_script = jsondata.get('bash script')
            description = jsondata.get('description')
            return bash_script, description
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            raise
    
    def interpret_output(self, original_query: str, command_output: str) -> str:
        model = genai.GenerativeModel('gemini-1.5-flash')
        from config import OUTPUT_INTERPRETATION_PROMPT
        prompt = OUTPUT_INTERPRETATION_PROMPT.format(
            query=original_query,
            output=command_output
        )
        response = model.generate_content(prompt)
        return response.text