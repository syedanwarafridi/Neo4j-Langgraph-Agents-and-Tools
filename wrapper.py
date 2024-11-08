from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage

class CustomLocalLLM(LLM):
    """Custom LLM wrapper that uses OpenAI chat model internally but presents as a local LLM."""
    
    temperature: float = 0.7
    max_tokens: int = 256
    model_name: str = "gpt-4o-mini"  # Changed to chat model
    openai_api_key: str = None
    
    def __init__(self, openai_api_key: str, **kwargs):
        """Initialize the custom LLM."""
        super().__init__(**kwargs)
        self.openai_api_key = openai_api_key
        # Initialize the underlying OpenAI Chat model
        self._llm = ChatOpenAI(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_name=self.model_name,
            openai_api_key=self.openai_api_key
        )
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "custom_local_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call."""
        try:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required but not provided")
            
            # Convert prompt to chat message format
            messages = [HumanMessage(content=prompt)]
            
            # Use the underlying OpenAI Chat model to generate response
            response = self._llm.invoke(messages)
            
            # Extract the content from the response
            return response.content
            
        except Exception as e:
            # Handle errors gracefully
            return f"Error generating response: {str(e)}"
    
    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_name": self.model_name
        }
