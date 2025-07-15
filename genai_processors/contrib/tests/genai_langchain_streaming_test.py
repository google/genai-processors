import unittest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors.contrib.genai_langchain_streaming import GenAILangChainProcessor


class DummyChunk:
  """A tiny stand-in for a LangChain chunk with just a `.content` field."""

  def __init__(self, content: str):
    self.content = content


class DummyChatModel:
  """Fake BaseChatModel that streams back given DummyChunk objects."""

  def __init__(self, chunks):
    # chunks: list of strings we want to stream back as LLM output
    self._chunks = [DummyChunk(c) for c in chunks]

  async def astream(self, payload):
    del payload  # Unused.
    # LangChain calls this to stream chunks
    for chunk in self._chunks:
      yield chunk


class GenAILangChainProcessorTest(parameterized.TestCase):
  """End-to-end tests for our GenAILangChainProcessor contrib module."""

  def test_text_only_sync(self):
    """Single-turn, non‐streaming inference via `apply_sync`."""
    # Make the model pretend to return two pieces of text
    llm = DummyChatModel(['Hello', ' world'])
    proc = GenAILangChainProcessor(llm=llm, enable_memory=False)

    # Our one user input
    parts = [
        content_api.ProcessorPart(
            'Hi there', mimetype='text/plain', role='user'
        )
    ]

    # Run synchronously
    output = processor.apply_sync(proc, parts)

    # We should get back one ProcessorPart per chunk
    self.assertLen(output, 2)
    self.assertEqual(output[0].text, 'Hello')
    self.assertEqual(output[1].text, ' world')

  def test_streaming_sync(self):
    """A turn that streams three chunks in sequence."""
    llm = DummyChatModel(['A', 'B', 'C'])
    proc = GenAILangChainProcessor(llm=llm, enable_memory=False)

    parts = [
        content_api.ProcessorPart(
            'Question?', mimetype='text/plain', role='user'
        )
    ]
    output = processor.apply_sync(proc, parts)

    self.assertLen(output, 3)
    self.assertSequenceEqual([p.text for p in output], ['A', 'B', 'C'])

  def test_multimodal_message_conversion(self):
    """Multiple parts of the same role each become their own LangChain message."""
    llm = DummyChatModel([])
    proc = GenAILangChainProcessor(llm=llm)

    parts = [
        content_api.ProcessorPart(
            'Caption me', mimetype='text/plain', role='user'
        ),
        content_api.ProcessorPart(
            b'\x89PNG\r\n\x1a\n', mimetype='image/png', role='user'
        ),
    ]
    msgs = proc._convert_to_langchain_messages(
        content_api.ProcessorContent(*parts)
    )

    # We expect two separate messages—one for text, one for image
    self.assertLen(msgs, 2)

    # First message carries the text fragment
    frag0 = msgs[0].content[0]
    self.assertEqual(frag0['type'], 'text')
    self.assertEqual(frag0['text'], 'Caption me')

    # Second message carries the image_url fragment
    frag1 = msgs[1].content[0]
    self.assertEqual(frag1['type'], 'image_url')
    self.assertTrue(frag1['image_url'].startswith('data:image/png;base64,'))

  @parameterized.parameters(('audio/mpeg', b'\x00\x01'))
  def test_unsupported_mimetype_raises(self, mime_type, raw_data):
    """Unsupported mime types should raise a ValueError."""
    llm = DummyChatModel([])
    proc = GenAILangChainProcessor(llm=llm)

    content = content_api.ProcessorContent(
        content_api.ProcessorPart(raw_data, mimetype=mime_type, role='user')
    )
    with self.assertRaises(ValueError) as cm:
      proc._convert_to_langchain_messages(content)
    self.assertIn('Unsupported mimetype', str(cm.exception))

  def test_memory_accumulates(self):
    """With enable_memory=True, conversation_history retains all turns."""
    llm = DummyChatModel(['ignored'])
    proc = GenAILangChainProcessor(llm=llm, enable_memory=True)

    # First call
    processor.apply_sync(
        proc, [content_api.ProcessorPart('First turn', role='user')]
    )
    # Second call
    processor.apply_sync(
        proc, [content_api.ProcessorPart('Second turn', role='user')]
    )

    texts = [part.text for part in proc.conversation_history]
    self.assertSequenceEqual(texts, ['First turn', 'Second turn'])


if __name__ == '__main__':
  unittest.main()
