"""Example script demonstrating OpenRouter integration with GenAI Processors.

This example shows how to use the OpenRouter processor to access hundreds of
AI models through a unified API. OpenRouter provides access to models from
OpenAI, Anthropic, Google, Meta, and many other providers.

## Setup

To run this example, you'll need:

1. An OpenRouter API key (get one at https://openrouter.ai/keys)
2. Install the required dependencies

## Environment Variables

Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

Run the script with different models:
```bash
# Use GPT-4.1
python openrouter_example.py --model "openai/gpt-4.1"

# Use Claude 3.5 Sonnet
python openrouter_example.py --model "anthropic/claude-3-5-sonnet"

# Use Llama 3.1
python openrouter_example.py --model "meta-llama/llama-3.1-70b-instruct"

# Use with custom parameters
python openrouter_example.py --temperature 0.9 --max-tokens 500
```
"""

import argparse
import asyncio
import os
import sys

from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.contrib import openrouter_model


# Popular models available on OpenRouter
POPULAR_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "claude-3.5-sonnet": "anthropic/claude-3-5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
    "command-r-plus": "cohere/command-r-plus",
}


def print_banner():
  """Print a welcome banner."""
  print("🚀 OpenRouter GenAI Processors Example")
  print("=" * 50)
  print("Access hundreds of AI models through a unified API!")
  print()


def list_popular_models():
  """List popular models available on OpenRouter."""
  print("📋 Popular Models:")
  print("-" * 30)
  for short_name, full_name in POPULAR_MODELS.items():
    print(f"  {short_name:<20} → {full_name}")
  print()
  print("💡 Use any model name from https://openrouter.ai/models")
  print()


async def demo_basic_chat(model: openrouter_model.OpenRouterModel):
  """Demonstrate basic chat functionality."""
  print("💬 Basic Chat Demo")
  print("-" * 20)

  prompts = [
      "What is artificial intelligence?",
      "Write a haiku about programming.",
      "Explain quantum computing in simple terms.",
  ]

  for i, prompt in enumerate(prompts, 1):
    print(f"\n🤔 Question {i}: {prompt}")
    print("🤖 Response: ", end="", flush=True)

    input_stream = streams.stream_content([content_api.ProcessorPart(prompt)])
    response_text = ""

    async for part in model(input_stream):
      if part.text:
        print(part.text, end="", flush=True)
        response_text += part.text
      elif part.metadata.get("generation_complete"):
        print(
            f"\n   ℹ️  Tokens used: {part.metadata.get('usage', {}).get('total_tokens', 'N/A')}"
        )

    print()


async def demo_conversation(model: openrouter_model.OpenRouterModel):
  """Demonstrate multi-turn conversation."""
  print("\n🗣️  Multi-turn Conversation Demo")
  print("-" * 35)

  conversation_history = []

  # First turn
  user_msg1 = "I'm planning a trip to Japan. What should I know?"
  conversation_history.append(content_api.ProcessorPart(user_msg1, role="user"))

  print(f"\n👤 User: {user_msg1}")
  print("🤖 Assistant: ", end="", flush=True)

  input_stream = streams.stream_content(conversation_history)
  assistant_response = ""

  async for part in model(input_stream):
    if part.text:
      print(part.text, end="", flush=True)
      assistant_response += part.text

  conversation_history.append(
      content_api.ProcessorPart(assistant_response, role="model")
  )

  # Second turn
  user_msg2 = "What about the best time to visit for cherry blossoms?"
  conversation_history.append(content_api.ProcessorPart(user_msg2, role="user"))

  print(f"\n\n👤 User: {user_msg2}")
  print("🤖 Assistant: ", end="", flush=True)

  input_stream = streams.stream_content(conversation_history)

  async for part in model(input_stream):
    if part.text:
      print(part.text, end="", flush=True)

  print("\n")


async def demo_creative_writing(model: openrouter_model.OpenRouterModel):
  """Demonstrate creative writing with higher temperature."""
  print("\n🎨 Creative Writing Demo (Higher Temperature)")
  print("-" * 45)

  # Create a model with higher temperature for creativity
  creative_model = openrouter_model.OpenRouterModel(
      api_key=model._api_key,
      model_name=model._model_name,
      generate_content_config={
          "temperature": 0.9,
          "max_tokens": 200,
      },
  )

  prompt = "Write a short story about a robot who dreams of becoming a chef."
  print(f"📝 Prompt: {prompt}")
  print("📖 Story: ", end="", flush=True)

  input_stream = streams.stream_content([content_api.ProcessorPart(prompt)])

  async for part in creative_model(input_stream):
    if part.text:
      print(part.text, end="", flush=True)

  print("\n")
  await creative_model.aclose()


def demo_sync_usage(model_name: str, api_key: str):
  """Demonstrate synchronous usage with apply_sync."""
  print("\n⚡ Synchronous Usage Demo")
  print("-" * 28)

  sync_model = openrouter_model.OpenRouterModel(
      api_key=api_key,
      model_name=model_name,
      generate_content_config={
          "temperature": 0.5,
          "max_tokens": 100,
      },
  )

  prompt = "Explain the concept of recursion in one paragraph."
  print(f"❓ Question: {prompt}")
  print("💡 Answer: ", end="")

  input_content = [content_api.ProcessorPart(prompt)]
  result = processor.apply_sync(sync_model, input_content)

  for part in result:
    if part.text:
      print(part.text, end="")

  print("\n")


async def main():
  """Main function to run the examples."""
  parser = argparse.ArgumentParser(
      description="OpenRouter GenAI Processors Example",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )

  parser.add_argument(
      "--model",
      default="openai/gpt-4o",
      help="Model to use (default: openai/gpt-4o). Use short names from popular models or full OpenRouter model names.",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=0.7,
      help="Temperature for generation (default: 0.7)",
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=500,
      help="Maximum tokens to generate (default: 500)",
  )
  parser.add_argument(
      "--list-models", action="store_true", help="List popular models and exit"
  )
  parser.add_argument(
      "--site-url", help="Your site URL (for OpenRouter rankings)"
  )
  parser.add_argument(
      "--site-name", help="Your site name (for OpenRouter rankings)"
  )

  args = parser.parse_args()

  print_banner()

  if args.list_models:
    list_popular_models()
    return

  # Get API key from environment
  api_key = os.getenv("OPENROUTER_API_KEY")
  if not api_key:
    print("❌ Error: OPENROUTER_API_KEY environment variable not set!")
    print("   Get your API key at: https://openrouter.ai/keys")
    print("   Then run: export OPENROUTER_API_KEY='your-key-here'")
    sys.exit(1)

  # Resolve model name (check if it's a short name)
  model_name = POPULAR_MODELS.get(args.model, args.model)

  print(f"🤖 Using model: {model_name}")
  print(f"🌡️  Temperature: {args.temperature}")
  print(f"📊 Max tokens: {args.max_tokens}")
  print()

  # Create the model
  model = openrouter_model.OpenRouterModel(
      api_key=api_key,
      model_name=model_name,
      site_url=args.site_url,
      site_name=args.site_name,
      generate_content_config={
          "temperature": args.temperature,
          "max_tokens": args.max_tokens,
      },
  )

  try:
    # Run demos
    await demo_basic_chat(model)
    await demo_conversation(model)
    await demo_creative_writing(model)
    demo_sync_usage(model_name, api_key)

    print("\n✅ All demos completed successfully!")
    print("\n🎯 Next steps:")
    print("   • Try different models with --model parameter")
    print("   • Experiment with temperature and max_tokens")
    print("   • Check out https://openrouter.ai/models for more models")
    print("   • Build your own processors with OpenRouter!")

  except Exception as e:
    print(f"\n❌ Error running demos: {e}")
    print("\n🔧 Troubleshooting:")
    print("   • Check your API key is valid")
    print("   • Ensure you have internet connection")
    print("   • Try a different model if the current one is unavailable")
    sys.exit(1)

  finally:
    # Clean up
    await model.aclose()


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\n\n👋 Demo interrupted by user. Goodbye!")
  except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    sys.exit(1)
