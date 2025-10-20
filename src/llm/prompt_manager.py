"""
Prompt management for LLM interactions
"""
from typing import Dict, Any, Optional
from string import Template


class PromptManager:
    """Manages prompts and templates for LLM interactions"""

    def __init__(self):
        self.templates: Dict[str, Template] = {
            "intent_analysis": Template("""
Analyze the following user request and determine the primary intent.

User request: $request

Common intents:
$intents

Return only the intent category name.
"""),
            "task_decomposition": Template("""
Decompose this complex request into simpler subtasks:

Request: $request

Provide a JSON list of subtasks.
"""),
            "response_formatting": Template("""
Format this response for the user:

Data: $data

Provide a clear, friendly response.
""")
        }

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Get a formatted prompt from template"""
        if template_name in self.templates:
            return self.templates[template_name].substitute(**kwargs)
        return ""

    def add_template(self, name: str, template_str: str) -> None:
        """Add a new template"""
        self.templates[name] = Template(template_str)