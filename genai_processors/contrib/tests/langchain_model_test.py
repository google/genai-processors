import unittest
from absl.testing import parameterized

from genai_processors import content_api, processor
from genai_processors.contrib.langchain_model import LangChainModel
from langchain_core.messages import HumanMessage

class MockChunk:
    """
    Lightweight stand-in for a LangChain chunk with only a `.content` attribute.
    """

    def __init__(self, content: str):
        """
        Initialize a mock chunk.

        Args:
            content (str): The text content of the chunk.
        """
        self.content = content


class MockChatModel:
    """
    Fake BaseChatModel that streams back provided MockChunk objects.
    """

    def __init__(self, chunks):
        """
        Prepare the mock model with a sequence of content chunks.

        Args:
            chunks (List[str]): The strings to stream as LLM output.
        """
        self._chunks = [MockChunk(c) for c in chunks]

    async def astream(self, payload):
        """
        Asynchronously stream chunks, simulating LangChain behavior.

        Args:
            payload: The input to the model (ignored in this mock).

        Yields:
            MockChunk: Each chunk in the preconfigured sequence.
        """
        for chunk in self._chunks:
            yield chunk


class LangChainModelTest(parameterized.TestCase):
    """
    End-to-end tests for the LangChainModel contrib module.
    """

    def test_text_only_sync(self):
        """
        Single-turn, non-streaming inference via `apply_sync`.
        """
        llm = MockChatModel(["Hello", " world"])
        proc = LangChainModel(llm=llm)

        parts = [
            content_api.ProcessorPart(
                "Hi there",
                mimetype="text/plain",
                role="user",
            )
        ]

        output = processor.apply_sync(proc, parts)

        self.assertLen(output, 2)
        self.assertEqual(output[0].text, "Hello")
        self.assertEqual(output[1].text, " world")

    def test_streaming_sync(self):
        """
        A turn that streams three chunks in sequence.
        """
        llm = MockChatModel(["A", "B", "C"])
        proc = LangChainModel(llm=llm)

        parts = [
            content_api.ProcessorPart(
                "Question?",
                mimetype="text/plain",
                role="user",
            )
        ]
        output = processor.apply_sync(proc, parts)

        self.assertLen(output, 3)
        self.assertSequenceEqual([p.text for p in output], ["A", "B", "C"])

    def test_multimodal_message_conversion(self):
        """
        Multiple fragments of the same role should be grouped into one
        LangChain message, preserving order.
        Each fragment now becomes its own HumanMessage.
        """
        llm = MockChatModel([])
        proc = LangChainModel(llm=llm)

        parts = [
            content_api.ProcessorPart("Caption me", mimetype="text/plain", role="user"),
            content_api.ProcessorPart(b"\x89PNG\r\n\x1a\n", mimetype="image/png", role="user"),
        ]
        msgs = proc._convert_to_langchain_messages(parts)

        # now two messages: one text, one image
        self.assertLen(msgs, 2)
        self.assertIsInstance(msgs[0], HumanMessage)
        self.assertEqual(msgs[0].content, "Caption me")

        self.assertIsInstance(msgs[1], HumanMessage)
        self.assertTrue(msgs[1].content.startswith("data:image/png;base64,"))


    @parameterized.parameters(
        ("audio/mpeg", b"\x00\x01"),
    )
    def test_unsupported_mimetype_raises(self, mime_type, raw_data):
        """
        Unsupported MIME types should raise a ValueError.
        """
        llm = MockChatModel([])
        proc = LangChainModel(llm=llm)

        bad_parts = [
            content_api.ProcessorPart(raw_data, mimetype=mime_type, role="user"),
        ]
        with self.assertRaises(ValueError) as cm:
            proc._convert_to_langchain_messages(bad_parts)
        self.assertIn("Unsupported mimetype", str(cm.exception))


if __name__ == "__main__":
    unittest.main()