"""
Base Agent Class
Provides common functionality for all AI agents using OpenRouter
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all AI agents in the trading system.
    Provides OpenRouter integration, retry logic, and error handling.
    """

    # Model configurations (using correct OpenRouter model IDs)
    MODELS = {
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "claude-sonnet": "anthropic/claude-sonnet-4.5",  # Claude Sonnet 4.5
        "gpt-5": "openai/gpt-5",  # GPT-5
        "gemini": "google/gemini-2.5-flash",  # Gemini 2.5 Flash (for YouTube)
        "grok": "x-ai/grok-3-mini",  # Grok 3 Mini (for X/Twitter)
        "deepseek": "deepseek/deepseek-chat-v3.1:free",  # DeepSeek v3.1 (free)
        "qwen": "qwen/qwen3-235b-a22b:free"  # Qwen3 235B (free)
    }

    def __init__(
        self,
        name: str,
        role: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name (e.g., "Market Analyst")
            role: Agent role description
            model: Model key from MODELS dict
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens
        """
        self.name = name
        self.role = role
        self.model = self.MODELS.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenRouter client
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        logger.info(f"Initialized {self.name} with model {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None
    ) -> str:
        """
        Call LLM with retry logic.

        Args:
            system_prompt: System instruction for the LLM
            user_prompt: User message/query
            response_format: Optional format instruction ("json", "text")

        Returns:
            LLM response as string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            logger.debug(f"{self.name} calling {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response.choices[0].message.content
            logger.debug(f"{self.name} received response ({len(content)} chars)")

            return content

        except Exception as e:
            logger.error(f"{self.name} LLM call failed: {e}")
            raise

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method - to be implemented by subclasses.

        Args:
            data: Input data for analysis

        Returns:
            Analysis results as dictionary
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def _create_report(
        self,
        analysis: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized report structure.

        Args:
            analysis: Analysis text from LLM
            confidence: Optional confidence score (0.0-1.0)
            metadata: Optional additional metadata

        Returns:
            Standardized report dictionary
        """
        report = {
            "agent": self.name,
            "role": self.role,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
        }

        if confidence is not None:
            report["confidence"] = confidence

        if metadata:
            report["metadata"] = metadata

        return report

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response that may contain markdown code blocks.

        Args:
            response: LLM response string

        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try direct JSON parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract from markdown code block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not extract JSON from response: {response[:100]}...")

    def __repr__(self) -> str:
        return f"{self.name} (model={self.model})"


if __name__ == "__main__":
    # Test the base agent
    logging.basicConfig(level=logging.INFO)

    try:
        agent = BaseAgent(
            name="Test Agent",
            role="Testing base functionality",
            model="gpt-4o-mini"
        )

        print(f"\n✅ Successfully created: {agent}")
        print(f"Model: {agent.model}")
        print(f"Temperature: {agent.temperature}")

        # Test LLM call
        response = agent._call_llm(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello in exactly 5 words."
        )
        print(f"\n✅ LLM Test Response: {response}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
