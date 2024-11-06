import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time


def run_inference():
    # Load a small sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Sample text for inference
    texts = [
        "This is a great example!",
        "I'm not sure about this.",
        "The weather is beautiful today!",
    ] * 100  # Repeat to make the task measurable

    # Tokenize and move to GPU
    start_time = time.time()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    end_time = time.time()

    # Print results
    print(f"Device used: {device}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Samples processed: {len(texts)}")
    print(f"Samples per second: {len(texts) / (end_time - start_time):.2f}")


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("torch version:", torch.__version__)
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")
    run_inference()
