import argparse
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

from src.models.factory import build_model, build_tokenizer
from src.models.text_rank import text_rank_summarize
from src.utils.environment import detect_runtime_env
from src.utils.logger import setup_logger
from src.utils.paths import get_checkpoint_dir
from src.utils.utils import (
    find_latest_checkpoint,
    load_checkpoint,
    text_to_token_ids,
    token_ids_to_text,
)
from src.utils.visualization import plot_attention_heatmap, plot_tsne_embeddings


class InferenceEngine:
    """
    Core engine for running inference across multiple summarization architectures.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = detect_runtime_env()
        self.logger = setup_logger(name="InferenceEngine", log_file="inference.log")

        # Configuration for visualization
        self.show_plots = not args.disable_plots

        # Define model registry
        self.available_models = [
            "TEXT_RANK",
            "POINTER_GENERATOR_NETWORK",
            "NEURAL_INTRA_ATTENTION_MODEL",
            "TRANSFORMER",
        ]

    def _get_input_text(self) -> str:
        """Fetch input text from args or load a default sample from CNN/DailyMail."""
        if self.args.text:
            return self.args.text

        self.logger.info(
            "No custom text provided. Loading sample from CNN/DailyMail test set..."
        )
        dataset = load_dataset(
            "abisee/cnn_dailymail", "3.0.0", split="test", streaming=True
        )
        sample = next(iter(dataset))
        return sample["article"]

    def _load_model_assets(
        self, model_name: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Initialize tokenizer and model architecture with latest weights."""
        if model_name == "TEXT_RANK":
            return None, None

        tokenizer = build_tokenizer(model_name)
        model = build_model(model_name, tokenizer, self.device)

        # Load the most recent checkpoint for the specific model
        ckpt_dir = str(get_checkpoint_dir(model_name))
        checkpoint_file, _ = find_latest_checkpoint(ckpt_dir)

        if checkpoint_file:
            load_checkpoint(model, checkpoint_file, map_location=self.device)
            self.logger.info(f"[{model_name}] Checkpoint loaded: {checkpoint_file}")
        else:
            self.logger.warning(
                f"[{model_name}] No checkpoint found in {ckpt_dir}. Using random weights."
            )

        return tokenizer, model

    def run_inference(self) -> None:
        """Main execution pipeline for inference."""
        input_text = self._get_input_text()
        self.logger.info(f"Input Text (First 200 chars): {input_text[:200]}...")

        models_to_run = (
            self.available_models if self.args.model == "ALL" else [self.args.model]
        )

        for model_name in models_to_run:
            self.logger.info(f"\n{'='*20} Running Inference: {model_name} {'='*20}")

            tokenizer, model = self._load_model_assets(model_name)

            if model_name == "TEXT_RANK":
                summary = text_rank_summarize(input_text, 3)[0]
                self._display_results(model_name, summary)
            else:
                self._run_neural_inference(model_name, model, tokenizer, input_text)

    def _run_neural_inference(
        self, name: str, model: Any, tokenizer: Any, text: str
    ) -> None:
        """Execute neural model forward pass and optional visualizations."""
        oov_list = []
        input_ids = text_to_token_ids(tokenizer, text, oov_list)

        # Model generation
        outputs = model.infer(
            input_ids,
            max_output_length=100,
            beam_width=6,
            return_attention=self.show_plots,
            return_embedding=self.show_plots,
        )

        # Decoding
        summary, output_tokens = token_ids_to_text(
            tokenizer, outputs["output_ids"][0], oov_list, return_output="both"
        )
        input_tokens = token_ids_to_text(
            tokenizer, input_ids, oov_list, return_output="list"
        )

        self._display_results(name, summary)

        if self.show_plots:
            self._visualize(name, outputs, input_tokens, output_tokens)

    def _display_results(self, model_name: str, summary: str) -> None:
        """Standardized log output for generated summaries."""
        print(f"\n[{model_name}] SUMMARY:")
        print("-" * 50)
        print(summary)
        print("-" * 50)

    def _visualize(
        self,
        name: str,
        outputs: Dict[str, Any],
        input_tokens: List[str],
        output_tokens: List[str],
    ) -> None:
        """Handle complex visualization logic for attention maps and embeddings."""
        self.logger.info(f"Generating visualizations for {name}...")

        # Attention Plotting Logic (Kept core logic as requested)
        if "cross_attention_distributions" in outputs:
            fig, ax = plt.subplots(figsize=(10, 10))
            # Support both single heatmap and multi-head for Transformer
            attn_data = outputs["cross_attention_distributions"]
            if name == "TRANSFORMER":
                attn_data = attn_data[0]  # Take first head for simplicity in Step 1

            plot_attention_heatmap(
                fig,
                self.env,
                ax,
                attn_data[: len(output_tokens)],
                output_tokens,
                input_tokens,
                "Input",
                "Generated",
                f"{name} Attention",
            )

        if "input_embeddings" in outputs:
            fig_tsne, ax_tsne = plt.subplots(figsize=(8, 8))
            plot_tsne_embeddings(
                fig_tsne,
                self.env,
                ax_tsne,
                outputs["input_embeddings"][0],
                input_tokens,
                f"T-SNE: {name}",
            )

        if self.env == "gui":
            plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarization Inference")
    parser.add_argument(
        "--text", type=str, default=None, help="Input text to summarize"
    )
    parser.add_argument("--model", type=str, default="ALL", help="Model name or 'ALL'")
    parser.add_argument(
        "--disable-plots", action="store_true", help="Disable visualizations"
    )
    return parser.parse_args()


if __name__ == "__main__":
    engine = InferenceEngine(parse_args())
    engine.run_inference()
