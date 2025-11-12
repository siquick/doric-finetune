# pip install openai
import os
import sys
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DUMMY", "unused"),
    base_url=os.getenv(
        "VLLM_URL", "https://siquick--doric-vllm-inference-serve.modal.run/v1"
    ),
)

MODEL_NAME = "franco334578/unsloth-gemma-3-4b-it-doric-v4-f16"

# Keep your own array and always resend it whole
messages: list[dict[str, str]] = []


def clean(s: str) -> str:
    return s.replace("<start_of_turn>", "").replace("<end_of_turn>", "")


def stream_response(user_message: str) -> str:
    """Send a message to the model and stream the response."""
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        stream=True,
        # # Critical for Gemma-style chats:
        # stop=["<end_of_turn>", "<eos>"],
        # # Gentle repetition control:
        # repetition_penalty=1.07,
    )

    response_parts = []
    print("Assistant: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            response_parts.append(content)

    print()  # New line after streaming
    full_response = clean("".join(response_parts))
    messages.append({"role": "assistant", "content": full_response})
    return full_response


def main():
    print("Chat CLI - Type 'quit' or 'exit' to end the conversation\n")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            stream_response(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()

