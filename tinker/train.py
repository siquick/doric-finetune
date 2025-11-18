import logging
import tinker
from dotenv import load_dotenv
import os
from lib.renderers import get_renderer
from lib.tokenizer_utils import get_tokenizer
from lib.load_dataset import load_dataset
from tinker.types import SamplingParams

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
logger = logging.getLogger(__name__)


def training_client():
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = get_tokenizer(base_model)
    # We need to use the instruct renderer for the Qwen3 intruct models
    renderer = get_renderer("qwen3_instruct", tokenizer)
    dataset = load_dataset("../datasets/doric_synth.jsonl")
    prompt = renderer.build_generation_prompt(dataset[99]["messages"])

    print(prompt)
    print("-" * 10)
    print(tokenizer.decode(prompt.to_ints()))

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model)
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    stop_sequences = renderer.get_stop_sequences()
    print(f"Stop sequences: {stop_sequences}")
    sampling_params = SamplingParams(temperature=0.5, stop=stop_sequences)
    output = sampling_client.sample(
        prompt, sampling_params=sampling_params, num_samples=1
    ).result()
    print(f"Output: {output.sequences[0].tokens}")
    sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
    print(f"Sampled message: {sampled_message}")
    print(f"Parse success: {parse_success}")


if __name__ == "__main__":
    training_client()
