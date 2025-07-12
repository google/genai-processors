### OpenRouter Model (`openrouter_model.py`)

The OpenRouter processor provides access to hundreds of AI models through
OpenRouter's unified API, including models from OpenAI, Anthropic, Google, Meta,
and many other providers.

#### Features

- **Unified API Access**: Access 100+ models through a single interface
- **Streaming Support**: Real-time response streaming for better UX
- **Cost Optimization**: OpenRouter automatically selects cost-effective options
- **Fallback Support**: Automatic failover to alternative models
- **Function Calling**: Support for tool use and function calling
- **Rich Configuration**: Fine-tune temperature, tokens, and other parameters

#### Quick Start

```python
from genai_processors.contrib import OpenRouterModel
from genai_processors import content_api, processor

# Initialize the processor
model = OpenRouterModel(
    api_key="your-openrouter-api-key",
    model_name="google/gemma-3-27b-it",  # Example model
    generate_content_config={
        'temperature': 0.7,
        'max_tokens': 500,
    }
)

# Use synchronously
input_content = [content_api.ProcessorPart("What is AI?")]
result = processor.apply_sync(model, input_content)
for part in result:
    if part.text:
        print(part.text)

# Use asynchronously
async def chat():
    async for part in model(streams.stream_content(input_content)):
        if part.text:
            print(part.text, end="", flush=True)
```

#### Popular Models

| Short Name | Full OpenRouter Name | Provider |
|------------|---------------------|----------|
| `gpt-4.1` | `openai/gpt-4.1` | OpenAI |
| `claude-sonnet-4` | `anthropic/claude-sonnet-4` | Anthropic |
| `gemini-2.5-flash` | `google/gemini-2.5-flash` | Google |
| `llama-4-maverick` | `meta-llama/llama-4-maverick` | Meta |
| `devstral-medium` | `mistralai/devstral-medium` | Mistral |

See the full list at [OpenRouter Models](https://openrouter.ai/models).

#### Configuration Options

```python
config = {
    'temperature': 0.7,           # Randomness (0.0-2.0)
    'top_p': 0.9,                # Nucleus sampling (0.0-1.0)
    'max_tokens': 1000,          # Maximum response length
    'frequency_penalty': 0.0,     # Reduce repetition (-2.0 to 2.0)
    'presence_penalty': 0.0,      # Encourage new topics (-2.0 to 2.0)
    'stop': ["###", "END"],       # Stop sequences
    'stream': True,               # Enable streaming (default: True)
    'tools': [...],               # Function calling tools
    'response_format': {"type": "json_object"},  # JSON response format
}
```

#### Advanced Features

**Multi-turn Conversations:**
```python
conversation = [
    content_api.ProcessorPart("Hello!", role='user'),
    content_api.ProcessorPart("Hi there! How can I help?", role='model'),
    content_api.ProcessorPart("What's the weather like?", role='user'),
]
```

**Function Calling:**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

config = {'tools': tools, 'tool_choice': 'auto'}
```

**Site Attribution (for OpenRouter rankings):**
```python
model = OpenRouterModel(
    api_key="your-key",
    model_name="openai/gpt-4o",
    site_url="https://your-site.com",
    site_name="Your App Name"
)
```

#### Example Applications

Check out the example script at `examples/openrouter_example.py` for
comprehensive demonstrations including:

- Basic chat interactions
- Multi-turn conversations
- Creative writing with temperature control
- Synchronous vs asynchronous usage
- Error handling and best practices

#### Getting Started

1. **Get an API Key**: Sign up at [OpenRouter](https://openrouter.ai/keys)

2. **Install Dependencies**:
   ```bash
   pip install genai-processors httpx
   ```

3. **Set Environment Variable**:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

4. **Run the Example**:
   ```bash
   python examples/openrouter_example.py
   ```

#### Error Handling

```python
try:
    async for part in model(input_stream):
        print(part.text, end="")
except RuntimeError as e:
    if "OpenRouter API error" in str(e):
        print(f"API Error: {e}")
    else:
        print(f"Request failed: {e}")
finally:
    await model.aclose()  # Clean up resources
```
