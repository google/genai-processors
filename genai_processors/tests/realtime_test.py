import asyncio
from collections.abc import AsyncIterable
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import realtime
from PIL import Image


ProcessorPart = content_api.ProcessorPart
ProcessorContent = content_api.ProcessorContent


def create_image(width, height):
  return Image.new('RGB', (width, height))


def add_function_name(part: ProcessorPart, fn_name: str) -> ProcessorPart:
  part_data = part.text if content_api.is_text(part.mimetype) else part.mimetype
  return ProcessorPart(f'{fn_name}({part_data})', role='model')


# Fake realtime models - simply wraps the parts around a model() call.
@processor.processor_function
async def main_model_fake(
    content: AsyncIterable[ProcessorPart],
) -> AsyncIterable[ProcessorPart]:
  async for part in content:
    if part.role == 'user':
      yield add_function_name(part, 'model')


# Fake of a realtime model raising an error
@processor.part_processor_function
async def main_model_exception_fake(
    part: ProcessorPart,
) -> AsyncIterable[ProcessorPart]:
  yield part
  raise ValueError('model error')


class RealTimeConversationTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.parameters([
      dict(
          input_stream=[
              ProcessorPart('hello', role='user'),
              ProcessorPart(
                  b'\x01\x00\x01\x00',
                  mimetype='audio/wav',
                  role='user',
              ),
              ProcessorPart(
                  create_image(100, 100),
                  mimetype='image/png',
                  role='user',
              ),
          ],
          output_text='model(hello)model(audio/wav)model(image/png)',
      ),
      dict(
          input_stream=[
              ProcessorPart('hello', role='user'),
              ProcessorPart(
                  b'\x01\x00\x01\x00',
                  mimetype='audio/wav',
                  role='user',
              ),
              content_api.END_OF_TURN,
              ProcessorPart('yo', role='user'),
              ProcessorPart(
                  create_image(100, 100),
                  mimetype='image/png',
                  role='user',
              ),
          ],
          # The first model call is cancelled, the second model call is made
          # with the full prompt.
          output_text='model(hello)model(audio/wav)model(yo)model(image/png)',
      ),
  ])
  async def test_realtime_single_ok(self, input_stream, output_text):
    input_stream = streams.stream_content(input_stream)
    output_parts = await streams.gather_stream(
        realtime.LiveModelProcessor(
            main_model_fake.to_processor(),
        )(input_stream)
    )
    actual = content_api.as_text(output_parts)
    self.assertEqual(actual, output_text)

  async def test_realtime_raise_exception(self):
    conversation_mgr = realtime.LiveModelProcessor(
        turn_processor=main_model_exception_fake.to_processor()
    )
    input_stream = streams.stream_content([
        ProcessorPart('hello', role='user'),
    ])
    with self.assertRaises(ValueError):
      await streams.gather_stream(conversation_mgr(input_stream))


@processor.processor_function
async def model_fake(
    content: AsyncIterable[ProcessorPart],
) -> AsyncIterable[ProcessorPart]:
  buffer = content_api.ProcessorContent()
  async for part in content:
    buffer += part
  # Assume a long model call.
  await asyncio.sleep(1)
  yield ProcessorPart(f'model({buffer.as_text()})', role='model')


class RealTimeConversationModelTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.output_queue = asyncio.Queue()
    self.user_not_talking = asyncio.Event()
    self.user_not_talking.set()
    self.rolling_prompt = realtime._RollingPrompt()

  def end_conversation(self):
    # To be called within an asyncio loop.
    async def _end_conversation():
      await asyncio.sleep(5)
      self.output_queue.put_nowait(None)

    processor.create_task(_end_conversation())

  async def test_output_order_ok(self):
    model = realtime._RealTimeConversationModel(
        output_queue=self.output_queue,
        generation=model_fake,
        rolling_prompt=self.rolling_prompt,
        user_not_talking=self.user_not_talking,
    )
    model.user_input(ProcessorPart('hello'))
    model.user_input(ProcessorPart('world'))

    # A turn takes 1 sec (see model_fake).
    model.turn()
    model.user_input(ProcessorPart('done', role='user'))

    self.end_conversation()
    output_parts = await streams.gather_stream(
        streams.dequeue(self.output_queue)
    )
    actual = content_api.as_text(output_parts, substream_name='')
    self.assertEqual(actual, 'model(helloworld)')

  async def test_prompt_order_ok(self):
    model = realtime._RealTimeConversationModel(
        output_queue=self.output_queue,
        generation=model_fake,
        rolling_prompt=self.rolling_prompt,
        user_not_talking=self.user_not_talking,
    )
    model.user_input(ProcessorPart('hello'))
    model.user_input(ProcessorPart('world'))
    # A turn takes 1 sec (see model_fake).
    model.turn()

    # Wait for the conversation to end.
    self.end_conversation()
    _ = await streams.gather_stream(streams.dequeue(self.output_queue))

    # Check that the rolling prompt put all the parts in the correct order.
    self.rolling_prompt.finalize_pending()
    prompt_pending = self.rolling_prompt.pending()
    self.rolling_prompt.finalize_pending()
    prompt_actual = await streams.gather_stream(prompt_pending)
    self.assertEqual(
        content_api.as_text(prompt_actual, substream_name=''),
        'helloworldmodel(helloworld)',
    )


class RealTimePromptTest(unittest.IsolatedAsyncioTestCase):

  async def test_add_part(self):
    rolling_prompt = realtime._RollingPrompt()
    prompt_content = rolling_prompt.pending()
    part_list = [ProcessorPart(str(i)) for i in range(5)]
    for c in part_list:
      rolling_prompt.add_part(c)
    rolling_prompt.finalize_pending()
    prompt_text = ProcessorContent(
        await streams.gather_stream(prompt_content)
    ).as_text()
    # prompt = part0-4.
    self.assertEqual(prompt_text, '01234')

  async def test_stashing(self):
    rolling_prompt = realtime._RollingPrompt()
    prompt_content = rolling_prompt.pending()
    part_list = [ProcessorPart(str(i)) for i in range(5)]
    for c in part_list:
      rolling_prompt.add_part(c)
    rolling_prompt.stash_part(ProcessorPart('while_outputting'))
    for c in part_list:
      rolling_prompt.add_part(c)
    rolling_prompt.apply_stash()
    rolling_prompt.finalize_pending()
    prompt_text = ProcessorContent(
        await streams.gather_stream(prompt_content)
    ).as_text()
    # prompt = part0-4, part0-4, part while outputting, prompt_suffix
    # -> part while outputting should always be put at the end.
    self.assertEqual(prompt_text, '0123401234while_outputting')

  async def test_cut_history(self):
    rolling_prompt = realtime._RollingPrompt(duration_prompt_sec=0.1)
    prompt_content = rolling_prompt.pending()
    part_count = 2
    part_list = [ProcessorPart(str(i)) for i in range(part_count)]
    img_list = [ProcessorPart(create_image(3, 3))] * part_count
    for idx, part in enumerate(part_list):
      rolling_prompt.add_part(part)
      rolling_prompt.add_part(img_list[idx])
      await asyncio.sleep(0.01)
    rolling_prompt.finalize_pending()
    prompt_text = [
        content_api.as_text(c) if content_api.is_text(c.mimetype) else 'img'
        for c in await streams.gather_stream(prompt_content)
    ]
    # First prompt gets the full history.
    self.assertEqual(prompt_text, ['0', 'img', '1', 'img'])
    await asyncio.sleep(0.08)
    rolling_prompt.finalize_pending()
    prompt_content = rolling_prompt.pending()
    for part in part_list:
      rolling_prompt.add_part(part)
    rolling_prompt.finalize_pending()
    prompt_text = [
        content_api.as_text(c) if content_api.is_text(c.mimetype) else 'img'
        for c in await streams.gather_stream(prompt_content)
    ]
    # Second prompt gets the cut history + what was fed now.
    self.assertEqual(prompt_text, ['1', 'img', '0', '1'])


if __name__ == '__main__':
  absltest.main()
