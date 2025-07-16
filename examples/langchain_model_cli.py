"""

Demonstrates how to use LangChainModel for:
  - Multimodal inputs (text + images)
  - Streaming responses
  
"""

import asyncio
import requests

from genai_processors.contrib.langchain_model import LangChainModel 
from genai_processors import streams
from genai_processors.content_api import ProcessorContent, ProcessorPart
from langchain_google_genai import ChatGoogleGenerativeAI


async def test_multimodal_processor():
    """
    Example: send a text prompt and an image for analysis
    and have the model describe the image and suggest similar artworks.
    """
    # 1. Build the text prompt
    text_prompt = ProcessorContent(
        ProcessorPart(
            "Describe this image and suggest similar artworks.",
            mimetype="text/plain",
            role="user",
            metadata={"query_type": "art_analysis"}
        )
    )

    # 2. Fetch the image bytes (or load locally)
    image_url = "https://fastly.picsum.photos/id/11/2500/1667.jpg?hmac=xxjFJtAPgshYkysU_aqx2sZir-kIOjNR9vx0te7GycQ"
    response = requests.get(image_url)
    image_bytes = response.content

    # 3. Wrap the image bytes in a ProcessorPart
    image_prompt = ProcessorContent(
        ProcessorPart(
            image_bytes,
            mimetype="image/jpeg",
            role="user",
            metadata={"source": "web", "artist": "Unknown"}
        )
    )

    # 4. Combine text + image into one content stream
    combined = text_prompt + image_prompt

    # 5. Create the LLM instance
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
    
    # 6. Instantiate the processor (no internal memory or streaming logic here)
    mm_processor = LangChainModel(
        llm=llm,
        system_instruction="You are an art expert assistant."
    )

    # 7. Run the multimodal prompt, streaming back the response
    async for part in mm_processor(streams.stream_content(combined)):
        print(part.text, end="", flush=True)

    print("\nProcessing complete!\n")


def main():
    """Run the example scenario."""
    asyncio.run(test_multimodal_processor())


if __name__ == "__main__":
    main()