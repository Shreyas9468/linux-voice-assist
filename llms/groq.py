import json
import logging
from groq import Groq
from llms.llm_base import LLMBase
from typing import Optional

class GroqService(LLMBase):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        
    def initialize(self):
        self.client = Groq(api_key=self.api_key)
        logging.info("Groq API initialized")
    
    def generate_bash_script(self, query: str, context: Optional[str] = None) -> tuple[str, str]:
        # Modify prompt to include context if available
        from config import PROMPT_TEMPLATE
        if context:
            enhanced_prompt = f"""
                Context information:
                {context}

                Follow these instructions carefully to generate a Bash script in JSON format:

                1. **Provide only the JSON output in the response**, without any additional explanation, comments, or code fences (like ```).
                2. Ensure the JSON has the following structure:
                - "bash script": The exact Bash script to execute.
                - "description": A brief explanation of what each command does.

                **Example JSON Output**:
                {{
                "bash script": "echo 'Hello, World!'",
                "description": "Prints 'Hello, World!' to the console."
                }}

                Do not include any text outside of this JSON format. Only return the JSON-formatted response as shown in the example above.
                The query is:
                {PROMPT_TEMPLATE.format(query=query)}
            """

        else:
            enhanced_prompt = PROMPT_TEMPLATE.format(query=query)
        
        # Using Mixtral model from Groq
        completion = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": enhanced_prompt}]
        )

        logging.debug(f"Groq response: {completion.choices[0].message.content}")
        
        try:
            jsondata = json.loads(completion.choices[0].message.content, strict=False)
            bash_script = jsondata.get('bash script')
            description = jsondata.get('description')
            return bash_script, description
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Groq response: {e}")
            raise
    
    def interpret_output(self, original_query: str, command_output: str) -> str:
        from config import OUTPUT_INTERPRETATION_PROMPT
        prompt = OUTPUT_INTERPRETATION_PROMPT.format(
            query=original_query,
            output=command_output
        )
        
        completion = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return completion.choices[0].message.content