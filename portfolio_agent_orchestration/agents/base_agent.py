"""
Base Agent Class for Portfolio Orchestration
Adapted from refs/TradingAgents patterns with LangChain integration
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, AGENT_CONFIG

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all portfolio agents using LangChain patterns
    
    Features:
    - OpenRouter LLM integration
    - Retry logic with exponential backoff
    - Tool binding support
    - Structured output handling
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        role: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        tools: Optional[List] = None
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            model: OpenRouter model ID
            role: Agent role description
            temperature: LLM temperature
            max_tokens: Maximum tokens for response
            tools: List of LangChain tools
        """
        self.name = name
        self.model = model
        self.role = role
        self.tools = tools or []
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/kubera-trading",
                "X-Title": "Kubera Portfolio Agent"
            }
        )
        
        # Bind tools if provided
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
        
        logger.info(f"Initialized {name} with model {model}")
    
    def create_prompt(self, system_message: str) -> ChatPromptTemplate:
        """Create a prompt template with system message"""
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages")
        ])
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def invoke(self, messages: List, **kwargs) -> Any:
        """
        Invoke the LLM with retry logic
        
        Args:
            messages: List of messages to send
            **kwargs: Additional arguments
            
        Returns:
            LLM response
        """
        try:
            response = self.llm.invoke(messages, **kwargs)
            logger.info(f"{self.name} completed successfully")
            return response
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            raise
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state and return updated state
        This should be overridden by child classes
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        raise NotImplementedError("Child classes must implement process()")
    
    def format_output(self, result: Any) -> str:
        """Format agent output for logging and display"""
        if hasattr(result, 'content'):
            return result.content
        return str(result)
    
    def get_tool_calls(self, result: Any) -> List[Dict]:
        """Extract tool calls from LLM result"""
        if hasattr(result, 'tool_calls'):
            return result.tool_calls
        return []

