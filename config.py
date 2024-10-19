# Gemini API Key
GEMINI_API_KEY = "AIzaSyBvVbc7VHY1TkPm2FWm20EQ51hbhNHcW84"

# Vosk model path
VOSK_MODEL_PATH = r"/home/sidd/dev/vosk-model-en-us-0.22-lgraph"

# Prompt template for Gemini API
PROMPT_TEMPLATE = """
You are given a task where you need to provide a Bash command that can be directly executed in a Bash script. The command should resolve the issue described below, and your response should follow these rules:
1. Provide the commands in the form of bash script to solve the problem.
2. Include a small description explaining what each command does.
3. Return the result in JSON format so that it can be parsed easily.
4. Each JSON object should have the following structure:
   - "bash script": The actual Bash script to be executed.
   - "description": Just small explanation of what the script does. Don't mention the word script itself in the description.

Here is the issue you need to resolve: {query}
Please return only the JSON object in your response.
Don't include any additional text like ``` or comments in your response.
Return in stringified JSON format only so that it can be converted using python json.loads().
"""
