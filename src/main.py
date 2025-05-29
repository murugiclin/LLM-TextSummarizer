```python
import argparse
import logging
from src.model.transformer import TransformerSummarizer
from src.data.tokenizer import Tokenizer
from src.data.dataset import Dataset
from src.utils.logger import setup_logger
from src.utils.config import Config

def main():
    # Initialize logger
    setup_logger()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Transformer-based text summarization")
    parser.add_argument("mode", choices=["train", "summarize"], help="Mode: train or summarize")
    parser.add_argument("--input", default="data/sample_data.txt", help="Input text file for training or summarization")
    parser.add_argument("--model", default="model.pt", help="Path to save/load model")
    parser.add_argument("--prompt", help="Text to summarize (summarize mode)")
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Initialize tokenizer and dataset
    tokenizer = Tokenizer()
    dataset = Dataset(args.input, tokenizer)

    # Initialize model
    model = TransformerSummarizer(config)

    if args.mode == "train":
        logging.info("Training transformer model...")
        model.train(dataset, epochs=10)
        model.save(args.model)
        logging.info(f"Model saved to {args.model}")
    else:
        if not args.prompt:
            raise ValueError("Prompt required for summarize mode")
        logging.info(f"Summarizing text: {args.prompt[:50]}...")
        model.load(args.model)
        summary = model.summarize(tokenizer, args.prompt)
        print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
```
