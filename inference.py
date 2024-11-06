import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from datetime import datetime


def run_inference(batch_size=32, num_batches=10):
    # Load model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Sample texts - mix of positive and negative examples
    texts = [
        "This is absolutely wonderful!",
        "I'm really disappointed.",
        "Best experience ever!",
        "This could be better.",
        "Not sure how I feel about this.",
    ] * (batch_size // 5 + 1)
    texts = texts[:batch_size]

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "model": model_name,
        "predictions": [],
        "performance": {"batch_times": [], "total_samples": 0},
    }

    # Run batches
    for batch_num in range(num_batches):
        start_time = time.time()

        # Tokenize and move to device
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_ids = outputs.logits.argmax(dim=-1)

        # Convert predictions to labels
        predictions = [
            model.config.id2label[class_id.item()] for class_id in predicted_class_ids
        ]

        # Record batch results
        batch_results = {
            "batch_num": batch_num,
            "texts": texts,
            "predictions": predictions,
            "batch_time": time.time() - start_time,
        }

        results["predictions"].append(batch_results)
        results["performance"]["batch_times"].append(batch_results["batch_time"])
        results["performance"]["total_samples"] += len(texts)

    # Calculate summary metrics
    avg_time = sum(results["performance"]["batch_times"]) / len(
        results["performance"]["batch_times"]
    )
    samples_per_second = results["performance"]["total_samples"] / sum(
        results["performance"]["batch_times"]
    )

    # Print human-readable summary
    print("\n=== ML Inference Results ===")
    print(f"Device: {device}")
    print(f"Average batch time: {avg_time:.4f} seconds")
    print(f"Samples per second: {samples_per_second:.1f}")
    print("\nSample Predictions from last batch:")
    for text, pred in zip(texts[:5], predictions[:5]):
        print(f"\nText: {text[:50]}...")
        print(f"Prediction: {pred}")


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("torch version:", torch.__version__)
    run_inference()
