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

A lightweight LangChain processor that wraps any BaseChatModel to provide:
  - Turn based, single prompt inference
  - Multimodal input support (text + images)
  - Flexible Jinja-based prompt templating
  
"""


from collections.abc import AsyncIterable
from typing import List, Optional, Union
import base64

from genai_processors import processor
from genai_processors.content_api import ProcessorPart, is_text, is_image
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class LangChainModel(processor.Processor):
    """
    A simple turn based wrapper around any LangChain BaseChatModel.
    Buffers one user turn, then streams the LLM response.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        system_instruction: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ):
        super().__init__()
        self.llm = llm
        self.system_instruction = system_instruction or ""
        self.prompt_template = (
            ChatPromptTemplate.from_template(prompt_template)
            if prompt_template
            else None
        )

    async def call(
        self,
        content_stream: AsyncIterable[ProcessorPart]
    ) -> AsyncIterable[ProcessorPart]:
        parts: List[ProcessorPart] = []
        async for part in content_stream:
            parts.append(part)

        msgs = self._convert_to_langchain_messages(parts)
        if self.system_instruction:
            msgs.insert(0, SystemMessage(content=self.system_instruction))

        payload = (
            {"input": self.prompt_template.format(messages=msgs)}
            if self.prompt_template else msgs
        )

        async for chunk in self.llm.astream(payload):
            model_name = getattr(self.llm, "model", type(self.llm).__name__)

            yield ProcessorPart(
                chunk.content,
                mimetype="text/plain",
                role="model",
                metadata={"model": model_name},
            )

    def _convert_to_langchain_messages(
        self,
        parts: List[ProcessorPart]
    ) -> List[Union[HumanMessage, SystemMessage, AIMessage]]:
        messages: List[Union[HumanMessage, SystemMessage, AIMessage]] = []
        fragments: List[dict] = []
        last_role: Optional[str] = None

        def flush():
            nonlocal fragments, last_role
            if not fragments:
                return
            cls = {
                "system": SystemMessage,
                "model": AIMessage,
            }.get(last_role, HumanMessage)
            messages.append(cls(content=list(fragments)))
            fragments = []

        for part in parts:
            if is_text(part.mimetype):
                frag = {
                    "type": "text",
                    "text": part.text,
                    "metadata": part.metadata
                }
            elif is_image(part.mimetype) and part.bytes:
                b64 = base64.b64encode(part.bytes).decode("utf-8")
                frag = {
                    "type": "image_url",
                    "image_url": f"data:{part.mimetype};base64,{b64}",
                    "metadata": part.metadata
                }
            else:
                raise ValueError(f"Unsupported mimetype: {part.mimetype}")

            if part.role != last_role:
                flush()
                last_role = part.role
            fragments.append(frag)

        flush()
        return messages
