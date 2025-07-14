# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
genai_langchain_processor.py

A lightweight LangChain processor that wraps any BaseChatModel to provide:
  - Conversation memory across turns
  - Real-time streaming responses
  - Multimodal input support (text + images)
  - Flexible Jinja-based prompt templating
  - Output metadata preservation
"""


import base64
from collections.abc import AsyncIterable
from typing import List, Optional, Union

from genai_processors import processor
from genai_processors.content_api import (
    ProcessorContent,
    ProcessorPart,
    is_text,
    is_image,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class GenAILangChainProcessor(processor.Processor):
    def __init__(
        self,
        llm: BaseChatModel,
        system_instruction: Optional[str] = None,
        prompt_template: Optional[str] = None,
        enable_memory: bool = True
    ):
        super().__init__()
        self.system_instruction = system_instruction or ""
        self.enable_memory = enable_memory
        self.conversation_history: List[ProcessorPart] = []
        self.llm = llm
        self.prompt_template = (
            ChatPromptTemplate.from_template(prompt_template)
            if prompt_template
            else None
        )

    async def call(
        self, content_stream: AsyncIterable[ProcessorPart]
    ) -> AsyncIterable[ProcessorPart]:
        content = await self._collect_content(content_stream)
        async for output in self._process_with_llm(content):
            yield output

    async def _collect_content(
        self, stream: AsyncIterable[ProcessorPart]
    ) -> ProcessorContent:
        buffer = ProcessorContent()
        async for part in stream:
            buffer += part
            if self.enable_memory:
                self.conversation_history.append(part)
        return buffer

    async def _process_with_llm(
        self, content: ProcessorContent
    ) -> AsyncIterable[ProcessorPart]:
        msgs = self._convert_to_langchain_messages(content)
        if self.system_instruction:
            msgs.insert(0, SystemMessage(content=self.system_instruction))
        if self.enable_memory and self.conversation_history:
            history_msgs = self._convert_to_langchain_messages(
                ProcessorContent(*self.conversation_history)
            )
            msgs = history_msgs + msgs
        payload = (
            {"input": self.prompt_template.format(messages=msgs)}
            if self.prompt_template
            else msgs
        )
        async for chunk in self.llm.astream(payload):
            model_name = (
                self.llm.model if hasattr(self.llm, "model")
                else type(self.llm).__name__
            )
            yield ProcessorPart(
                chunk.content,
                mimetype="text/plain",
                role="model",
                metadata={"model": model_name},

            )

    def _convert_to_langchain_messages(
        self, content: ProcessorContent
    ) -> List[Union[HumanMessage, SystemMessage, AIMessage]]:
        messages = []
        
        for part in content:
            if is_text(part.mimetype):
                content_fragment = {
                    "type": "text",
                    "text": part.text,
                    "metadata": part.metadata
                }
            elif is_image(part.mimetype) and part.bytes:
                b64 = base64.b64encode(part.bytes).decode("utf-8")
                content_fragment = {
                    "type": "image_url",
                    "image_url": f"data:{part.mimetype};base64,{b64}",
                    "metadata": part.metadata
                }
            else:
                raise ValueError(f"Unsupported mimetype: {part.mimetype}")
            
            if part.role == "system":
                messages.append(SystemMessage(content=[content_fragment]))
            elif part.role == "model":
                messages.append(AIMessage(content=[content_fragment]))
            else:
                messages.append(HumanMessage(content=[content_fragment]))
        
        return messages