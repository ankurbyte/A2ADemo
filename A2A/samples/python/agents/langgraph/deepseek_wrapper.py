from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from typing import List, Optional, Any
import deepseek

class DeepSeekChat(BaseChatModel):
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        super().__init__()
        self.client = deepseek.Client(api_key=api_key)
        self.model_name = model_name

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> AIMessage:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": m.type, "content": m.content} for m in messages],
            **kwargs
        )
        return AIMessage(content=response.choices[0].message.content)