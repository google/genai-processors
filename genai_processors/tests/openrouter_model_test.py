"""Tests for OpenRouter model processor."""

import asyncio
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


class OpenRouterModelTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
  """Tests for OpenRouterModel processor."""

  def setUp(self):
    """Set up test fixtures."""
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
        name='test_function',
        args={'param1': 'value1', 'param2': 42}
    )
    part = content_api.ProcessorPart(func_call, role='model')
    
    result = openrouter_model._to_openrouter_message(part)
    
    expected = {
        'role': 'assistant',
        'function_call': {
            'name': 'test_function',
            'arguments': json.dumps({'param1': 'value1', 'param2': 42}),
        }
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
      dict(
          testcase_name='invalid_json',
          line='data: {invalid json}',
          expected=None,
      ),
  )
  def test_parse_sse_line(self, line, expected):
    """Test parsing of Server-Sent Events lines."""
    result = openrouter_model._parse_sse_line(line)
    self.assertEqual(result, expected)

  async def test_call_with_mock_response(self):
    """Test the call method with mocked HTTP response."""
    mock_response_data = [
        'data: {"choices": [{"delta": {"content": "Hello"}}], "model": "openai/gpt-4o"}\n',
        'data: {"choices": [{"delta": {"content": " world"}}], "model": "openai/gpt-4o"}\n',
        'data: {"choices": [{"finish_reason": "stop"}], "model": "openai/gpt-4o"}\n',
        'data: [DONE]\n',
    ]

    async def mock_aiter_text():
      for chunk in mock_response_data:
        yield chunk

    mock_response = mock.AsyncMock()
    mock_response.aiter_text = mock_aiter_text
    mock_response.raise_for_status = mock.AsyncMock()
    
    mock_context = mock.AsyncMock()
    mock_context.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = mock.AsyncMock(return_value=None)
    
    with mock.patch.object(
        httpx.AsyncClient, 'stream',
        return_value=mock_context
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
      
      # Verify the request was made correctly
      mock_stream.assert_called_once()
      call_args = mock_stream.call_args
      self.assertEqual(call_args[0][0], 'POST')
      self.assertEqual(call_args[0][1], '/chat/completions')
      
      # Check the request payload
      request_json = call_args[1]['json']
      self.assertEqual(request_json['model'], self.model_name)
      self.assertEqual(request_json['messages'], [{'role': 'user', 'content': 'Test input'}])
      self.assertTrue(request_json['stream'])
      self.assertEqual(request_json['temperature'], 0.7)
      self.assertEqual(request_json['max_tokens'], 100)
      
      # Verify the response processing
      self.assertEqual(len(result), 3)  # Two content chunks + finish reason
      self.assertEqual(result[0].text, 'Hello')
      self.assertEqual(result[1].text, ' world')
      self.assertEqual(result[2].text, '')
      self.assertEqual(result[2].metadata['finish_reason'], 'stop')
      self.assertTrue(result[2].metadata['generation_complete'])

  async def test_call_with_http_error(self):
    """Test error handling for HTTP errors."""
    error_response = mock.Mock()
    error_response.aread = mock.AsyncMock(
        return_value=b'{"error": {"message": "Invalid API key"}}'
    )
    
    http_error = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=mock.Mock(),
        response=error_response
    )

    with mock.patch.object(
        httpx.AsyncClient, 'stream',
        side_effect=http_error
    ):
      model = openrouter_model.OpenRouterModel(
          api_key='invalid_key',
          model_name=self.model_name,
      )
      
      input_content = [content_api.ProcessorPart('Test input')]
      
      with self.assertRaises(RuntimeError) as context:
        async for _ in model(streams.stream_content(input_content)):
          pass
      
      self.assertIn('OpenRouter API error: Invalid API key', str(context.exception))

  async def test_call_with_empty_input(self):
    """Test call method with empty input."""
    model = openrouter_model.OpenRouterModel(
        api_key=self.api_key,
        model_name=self.model_name,
    )
    
    # Empty input should return immediately without making API calls
    input_content = []
    result = []
    
    async for part in model(streams.stream_content(input_content)):
      result.append(part)
    
    self.assertEqual(len(result), 0)

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
        }
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
        }
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

  def test_apply_sync_integration(self):
    """Test integration with processor.apply_sync."""
    mock_response_data = [
        'data: {"choices": [{"delta": {"content": "Sync response"}}]}\n',
        'data: {"choices": [{"finish_reason": "stop"}]}\n',
        'data: [DONE]\n',
    ]

    async def mock_aiter_text():
      for chunk in mock_response_data:
        yield chunk

    mock_response = mock.AsyncMock()
    mock_response.aiter_text = mock_aiter_text
    mock_response.raise_for_status = mock.AsyncMock()
    
    mock_context = mock.AsyncMock()
    mock_context.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = mock.AsyncMock(return_value=None)
    
    with mock.patch.object(
        httpx.AsyncClient, 'stream',
        return_value=mock_context
    ):
      model = openrouter_model.OpenRouterModel(
          api_key=self.api_key,
          model_name=self.model_name,
      )
      
      input_content = [content_api.ProcessorPart('Test sync input')]
      result = processor.apply_sync(model, input_content)
      
      # Should get the content response
      text_parts = [part for part in result if part.text]
      self.assertEqual(len(text_parts), 1)
      self.assertEqual(text_parts[0].text, 'Sync response')


if __name__ == '__main__':
  unittest.main()
