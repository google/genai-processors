# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
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

"""Tests for OpenRouter model processor."""

import json
import unittest
from unittest import mock

from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.contrib import openrouter_model
from google.genai import types as genai_types
import httpx


class OpenRouterModelTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):
  """Tests for OpenRouterModel processor."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.api_key = 'test_api_key'
    self.model_name = 'openai/gpt-4o'
    self.base_config = {
        'temperature': 0.7,
        'max_tokens': 100,
    }

  def test_init_with_default_config(self):
    """Test initialization with default configuration."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )

    self.assertEqual(model._api_key, self.api_key)
    self.assertEqual(model._model_name, self.model_name)
    self.assertEqual(model._base_url, openrouter_model._DEFAULT_BASE_URL)
    self.assertIsNone(model._site_url)
    self.assertIsNone(model._site_name)

  def test_init_with_custom_config(self):
    """Test initialization with custom configuration."""
    site_url = 'https://example.com'
    site_name = 'Test Site'
    base_url = 'https://custom.openrouter.com/api/v1'

    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
        base_url=base_url,
        site_url=site_url,
        site_name=site_name,
        generate_content_config=self.base_config,
    )

    self.assertEqual(model._base_url, base_url)
    self.assertEqual(model._site_url, site_url)
    self.assertEqual(model._site_name, site_name)
    self.assertEqual(model._config, self.base_config)

  def test_key_prefix(self):
    """Test key_prefix property."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )

    expected_prefix = f'OpenRouterModel_{self.model_name}'
    self.assertEqual(model.key_prefix, expected_prefix)

  @parameterized.named_parameters(
      dict(
          testcase_name='text_user',
          part=content_api.ProcessorPart('Hello world', role='user'),
          expected={'role': 'user', 'content': 'Hello world'},
      ),
      dict(
          testcase_name='text_model',
          part=content_api.ProcessorPart('Hi there!', role='model'),
          expected={'role': 'model', 'content': 'Hi there!'},
      ),
      dict(
          testcase_name='text_no_role',
          part=content_api.ProcessorPart('Hello'),
          expected={'role': 'user', 'content': 'Hello'},
      ),
  )
  def test_to_openrouter_message_text(self, part, expected):
    """Test conversion of text parts to OpenRouter message format."""
    result = openrouter_model._to_openrouter_message(part)
    self.assertEqual(result, expected)

  def test_to_openrouter_message_function_call(self):
    """Test conversion of function call parts."""
    func_call = genai_types.Part.from_function_call(
        name='test_function', args={'param1': 'value1', 'param2': 42}
    )
    part = content_api.ProcessorPart(func_call, role='model')

    result = openrouter_model._to_openrouter_message(part)

    expected = {
        'role': 'model',
        'function_call': {
            'name': 'test_function',
            'arguments': json.dumps({'param1': 'value1', 'param2': 42}),
        },
    }
    self.assertEqual(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='comment_line',
          line=': OPENROUTER PROCESSING',
          expected=None,
      ),
      dict(
          testcase_name='empty_line',
          line='',
          expected=None,
      ),
      dict(
          testcase_name='done_event',
          line='data: [DONE]',
          expected={'type': 'done'},
      ),
      dict(
          testcase_name='valid_json',
          line='data: {"choices": [{"delta": {"content": "Hello"}}]}',
          expected={'choices': [{'delta': {'content': 'Hello'}}]},
      ),
  )
  def test_parse_sse_line(self, line, expected):
    """Test parsing of Server-Sent Events lines."""
    result = openrouter_model._parse_sse_line(line)
    self.assertEqual(result, expected)

  def test_parse_sse_line_invalid_json(self):
    """Test that invalid JSON raises an exception."""
    with self.assertRaises(json.JSONDecodeError):
      openrouter_model._parse_sse_line('data: {invalid json}')

  async def test_call_with_mock_response(self):
    """Test the call method with mocked HTTP response."""
    mock_response_data = [
        (
            'data: {"choices": [{"delta": {"content": "Hello"}}], "model":'
            ' "openai/gpt-4o"}'
        ),
        (
            'data: {"choices": [{"delta": {"content": " world"}}], "model":'
            ' "openai/gpt-4o"}'
        ),
        (
            'data: {"choices": [{"finish_reason": "stop"}], "model":'
            ' "openai/gpt-4o"}'
        ),
        'data: [DONE]',
    ]

    async def mock_aiter_lines():
      for chunk in mock_response_data:
        yield chunk

    def mock_raise_for_status():
      return None

    mock_response = mock.AsyncMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = mock_raise_for_status

    mock_context = mock.AsyncMock()
    mock_context.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = mock.AsyncMock(return_value=None)

    with mock.patch.object(
        httpx.AsyncClient, 'stream', return_value=mock_context
    ) as mock_stream:

      model = openrouter_model.OpenRouterModel(
          api_key=self.api_key,
          model_name=self.model_name,
          generate_content_config=self.base_config,
      )

      input_content = [content_api.ProcessorPart('Test input')]
      result = []

      async for part in model(streams.stream_content(input_content)):
        result.append(part)

      # Verify the request was made correctly.
      mock_stream.assert_called_once()
      call_args = mock_stream.call_args
      self.assertEqual(call_args[0][0], 'POST')
      self.assertEqual(call_args[0][1], '/chat/completions')

      # Check the request payload
      request_json = call_args[1]['json']
      self.assertEqual(request_json['model'], self.model_name)
      self.assertEqual(
          request_json['messages'], [{'role': 'user', 'content': 'Test input'}]
      )
      self.assertTrue(request_json['stream'])
      self.assertEqual(request_json['temperature'], 0.7)
      self.assertEqual(request_json['max_tokens'], 100)

      # Verify the response processing.
      # Two content chunks + finish reason.
      self.assertEqual(len(result), 3)  # pylint: disable=g-generic-assert
      self.assertEqual(result[0].text, 'Hello')
      self.assertEqual(result[1].text, ' world')
      self.assertEqual(result[2].text, '')
      self.assertEqual(result[2].metadata['finish_reason'], 'stop')
      self.assertTrue(result[2].metadata['end_of_turn'])

  def test_call_with_http_error(self):
    """Test error parsing functionality."""
    model = openrouter_model.OpenRouterModel(
        api_key='test_key',
        model_name=self.model_name,
    )

    # Test error parsing.
    error_body = b'{"error": {"message": "Invalid API key"}}'
    parsed_error = model._parse_error_response(error_body)
    self.assertEqual(parsed_error, 'Invalid API key')

    # Test with malformed JSON.
    error_body = b'Invalid JSON'
    parsed_error = model._parse_error_response(error_body)
    self.assertEqual(parsed_error, 'Invalid JSON')

  async def test_call_with_empty_input(self):
    """Test call method with empty input."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )

    # Empty input should return immediately without making API calls.
    result = await processor.apply_async(model, [])
    self.assertEqual(result, [])

  def test_build_metadata(self):
    """Test metadata building from response data."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )

    response_data = {
        'id': 'chatcmpl-123',
        'model': 'openai/gpt-4o',
        'created': 1234567890,
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30,
        },
    }

    metadata = model._build_metadata(response_data)

    expected = {
        'request_id': 'chatcmpl-123',
        'model': 'openai/gpt-4o',
        'created': 1234567890,
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30,
        },
    }

    self.assertEqual(metadata, expected)

  async def test_aclose(self):
    """Test proper cleanup of HTTP client."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )

    with mock.patch.object(model._client, 'aclose') as mock_aclose:
      await model.aclose()
      mock_aclose.assert_called_once()

  async def test_async_context_manager(self):
    """Test async context manager functionality."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )

    with mock.patch.object(model._client, 'aclose') as mock_aclose:
      async with model:
        pass  # Just test the context manager works.
      mock_aclose.assert_called_once()

  def test_apply_sync_integration(self):
    """Test integration with processor.apply_sync."""
    mock_response_data = [
        'data: {"choices": [{"delta": {"content": "Sync response"}}]}',
        'data: {"choices": [{"finish_reason": "stop"}]}',
        'data: [DONE]',
    ]

    async def mock_aiter_lines():
      for chunk in mock_response_data:
        yield chunk

    def mock_raise_for_status():
      return None

    mock_response = mock.AsyncMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = mock_raise_for_status

    mock_context = mock.AsyncMock()
    mock_context.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = mock.AsyncMock(return_value=None)

    with mock.patch.object(
        httpx.AsyncClient, 'stream', return_value=mock_context
    ):
      model = openrouter_model.OpenRouterModel(
          api_key=self.api_key,
          model_name=self.model_name,
      )

      input_content = [content_api.ProcessorPart('Test sync input')]
      result = processor.apply_sync(model, input_content)

      # Should get the content response.
      text_parts = [part for part in result if part.text]
      self.assertEqual(len(text_parts), 1)  # pylint: disable=g-generic-assert
      self.assertEqual(text_parts[0].text, 'Sync response')

  def test_response_schema_conversion(self):
    """Test that response_schema is correctly converted to OpenRouter format."""
    # Create a simple schema.
    schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'name': genai_types.Schema(type=genai_types.Type.STRING),
            'age': genai_types.Schema(type=genai_types.Type.INTEGER),
        },
        required=['name'],
    )

    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
        generate_content_config={'response_schema': schema},
    )

    # The schema should be in the config.
    self.assertEqual(model._config['response_schema'], schema)

  def test_tools_conversion(self):
    """Test that tools are correctly converted to OpenRouter format."""
    # Create a simple tool.
    function_decl = genai_types.FunctionDeclaration(
        name='get_weather',
        description='Get weather information',
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                'location': genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description='Location to get weather for',
                ),
            },
            required=['location'],
        ),
    )

    tool = genai_types.Tool(function_declarations=[function_decl])

    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
        generate_content_config={'tools': [tool]},
    )

    # Verify tools were processed correctly.
    self.assertEqual(
        model._payload_args['tools'],
        [{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get weather information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'Location to get weather for',
                        }
                    },
                    'required': ['location'],
                },
            },
        }],
    )


if __name__ == '__main__':
  unittest.main()
