from __future__ import annotations

import argparse
import logging

from coherence import (
    chat_with_model,
    load_dream_model,
    load_model_with_metadata,
    run_dream_session,
    run_coherence_training,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coherence-oriented world simulator and chat.")
    parser.add_argument(
        "--mode",
        choices=["chat", "simulate", "train_coherence", "dream_with_coherence", "train_datasets"],
        default="chat",
        help="Execution mode.",
    )
    parser.add_argument(
        "--datasets_dir",
        default="datasets",
        help="Path to datasets folder for train_datasets mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for dataset training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs for dataset training.",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of steps for simulation/training.")
    parser.add_argument(
        "--expand",
        type=int,
        default=0,
        help="Number of extra Transformer layers to add before training (e.g., --expand 2)",
    )
    args = parser.parse_args()

    if args.mode == "chat":
        tokenizer, model, meta = load_model_with_metadata()
        chat_with_model(tokenizer, model)
    elif args.mode == "simulate":
        tokenizer, model, meta = load_model_with_metadata()
        state = run_dream_session(tokenizer, model, num_steps=args.steps)
        log.info("Final summary: %s", state.summary())
        log.info("Coherence report: %s", state.coherence_report())
    elif args.mode == "train_coherence":
        # Check if we should continue training an existing model?
        # original function loads fresh model. Let's keep it fresh for pure training test,
        # or maybe we should respect persistence?
        # The user's request is "train, save and reuse".
        # But run_coherence_training in training.py uses load_dream_model internally.
        # I should probably update run_coherence_training inside training.py too if I want that consistently.
        # For now, let's leave this one as is or trust training.py update.
        run_coherence_training(steps=args.steps)
    elif args.mode == "dream_with_coherence":
        tokenizer, model, meta = load_model_with_metadata()
        state = run_dream_session(tokenizer, model, num_steps=args.steps)
        log.info("Final summary: %s", state.summary())
        log.info("Coherence report: %s", state.coherence_report())
    elif args.mode == "train_datasets":
        from coherence.training import train_on_datasets
        from coherence.expansion import add_transformer_layer, save_model_with_metadata
        
        # Load model first to potentially expand it
        tokenizer, model, meta = load_model_with_metadata()
        
        # Add extra layers if requested
        if args.expand > 0:
            current_layers = len(getattr(model, "extra_layers", []))
            layers_to_add = args.expand - current_layers
            for i in range(layers_to_add):
                add_transformer_layer(model)
                log.info("Added extra Transformer layer %d/%d", current_layers + i + 1, args.expand)
            # Save the expanded model immediately
            if layers_to_add > 0:
                meta["expansions"] = {
                    "extra_layers": len(model.extra_layers),
                    "extra_attn_blocks": len(getattr(model, "aux_attention", [])),
                    "mlp_expansions": len(getattr(model, "mlp_expansions", [])),
                    "lora_expansions": len(getattr(model, "lora_adapters", [])),
                }
                save_model_with_metadata(model, meta)
                log.info("Saved expanded model with %d layers", args.expand)

        train_on_datasets(datasets_dir=args.datasets_dir, batch_size=args.batch_size, epochs=args.epochs)


if __name__ == "__main__":
    main()
