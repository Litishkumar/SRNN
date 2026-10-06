
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

        self.reliability_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        flat = self.flatten(features)

        logits = self.classifier(flat)
        reliability = self.reliability_head(flat)

        return logits, reliability.squeeze()


# ------------------------------------------------------------
# Load trained SRNN
# ------------------------------------------------------------
model = SRNN().to(device)

checkpoint = "srnn_cifar_model.pth"

try:
    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )
except FileNotFoundError:
    print(f"\nERROR: {checkpoint} was not found.")
    print("Place srnn_cifar_model.pth in the same folder as this script.")
    raise SystemExit

model.eval()

softmax = nn.Softmax(dim=1)

# ------------------------------------------------------------
# Pick 5 sample test images
# These are deterministic, so the same five are used each run.
# ------------------------------------------------------------
sample_indices = [0, 100, 500, 1000, 5000]

print("\n" + "=" * 72)
print("              SRNN - 5 IMAGE PREDICTION DEMO")
print("=" * 72)
print(f"Reliability threshold: {THRESHOLD:.2f}")
print()

with torch.no_grad():
    for display_no, idx in enumerate(sample_indices, start=1):

        image, true_label = test_dataset[idx]

        # Add batch dimension
        input_tensor = image.unsqueeze(0).to(device)

        logits, reliability = model(input_tensor)

        # Class probabilities
        probabilities = softmax(logits)

        confidence, predicted_class = torch.max(probabilities, dim=1)

        confidence = confidence.item()
        predicted_class = predicted_class.item()
        reliability = reliability.item()

        true_class = true_label
        predicted_name = CLASS_NAMES[predicted_class]
        true_name = CLASS_NAMES[true_class]

        # Selective decision
        accepted = reliability >= THRESHOLD

        if accepted:
            decision = "ACCEPT - RELIABLE"
        else:
            decision = "REJECT / UNCERTAIN"

        correct = predicted_class == true_class

        print("-" * 72)
        print(f"IMAGE {display_no}  (test index = {idx})")
        print("-" * 72)
        print(f"Actual Class       : {true_name}")
        print(f"Predicted Class    : {predicted_name}")
        print(f"Softmax Confidence : {confidence * 100:.2f}%")
        print(f"SRNN Reliability   : {reliability * 100:.2f}%")
        print(f"Threshold          : {THRESHOLD * 100:.2f}%")
        print(f"Decision            : {decision}")
        print(f"Prediction Correct : {'YES' if correct else 'NO'}")

        # Show top-3 classes so you can understand the prediction.
        top_probs, top_indices = torch.topk(probabilities[0], 3)

        print("Top-3 predictions  :")
        for rank, (p, c) in enumerate(
            zip(top_probs.tolist(), top_indices.tolist()), start=1
        ):
            print(f"  {rank}. {CLASS_NAMES[c]:12s} {p * 100:.2f}%")

print("\n" + "=" * 72)
print("Demo completed.")
print("=" * 72)
print("""
INTERPRETATION:
- Confidence = how strongly the classifier favors the predicted class.
- Reliability = the separate SRNN reliability-head output.
- If reliability >= threshold, the prediction is accepted.
- Otherwise, it is treated as uncertain.
