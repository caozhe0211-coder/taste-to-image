import marimo
from marimo import Html

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    """
    Common imports shared across the app.
    """
    import numpy as np
    import os
    import json
    return json, np, os


@app.cell
def _():
    """
    Define 7 abstract input classes.

    These represent different hypothetical water profiles that the MLP will
    learn to map into 7 "taste" classes.
    """

    input_class_names = [
        "Class 0 – crisp & balanced profile",
        "Class 1 – mineral-forward profile",
        "Class 2 – citrus-infused profile",
        "Class 3 – smoky & metallic profile",
        "Class 4 – ocean-adjacent, slightly briny profile",
        "Class 5 – floral & delicate profile",
        "Class 6 – earthy & robust profile",
    ]

    num_input_classes = len(input_class_names)
    return input_class_names, num_input_classes


@app.cell
def _(np, num_input_classes, os):
    """
    Load a dummy dataset for training a 7-class MLP from disk.

    The dataset is stored in `water_taste_data.csv`, with each row containing
    7 input features followed by an integer class label (0–6).
    """

    data_path = "water_taste_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset file '{data_path}' not found. "
            "Make sure it exists or regenerate it."
        )

    data = np.loadtxt(data_path, delimiter=",")
    if data.shape[1] != num_input_classes + 1:
        raise ValueError(
            f"Expected {num_input_classes + 1} columns in dataset, "
            f"got {data.shape[1]}"
        )

    X = data[:, :num_input_classes]
    y = data[:, num_input_classes].astype(int)
    return X, y


@app.cell
def _():
    """
    Training configuration for the MLP.

    `num_training_steps` controls how many epochs the network trains for.
    """

    num_training_steps = 100
    return (num_training_steps,)


@app.cell
def _(X, np, num_training_steps, y):
    """
    Define and train a simple 2-layer MLP classifier that maps the
    7-dimensional input into 7 output classes.

    This is a minimal NumPy-based implementation (no external ML libraries).
    """

    class SimpleMLP:
        def __init__(
            self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0
        ):
            rng = np.random.default_rng(seed)
            self.W1 = rng.normal(scale=0.1, size=(input_dim, hidden_dim))
            self.b1 = np.zeros(hidden_dim)
            self.W2 = rng.normal(scale=0.1, size=(hidden_dim, output_dim))
            self.b2 = np.zeros(output_dim)

        def _forward(self, X_batch: np.ndarray):
            z1 = X_batch @ self.W1 + self.b1
            h1 = np.maximum(0, z1)  # ReLU
            logits = h1 @ self.W2 + self.b2
            return logits, h1

        def predict_logits(self, X_batch: np.ndarray) -> np.ndarray:
            logits, _ = self._forward(X_batch)
            return logits

        def predict_proba(self, X_batch: np.ndarray) -> np.ndarray:
            logits = self.predict_logits(X_batch)
            logits = logits - logits.max(
                axis=1, keepdims=True
            )  # numerical stability
            exp_logits = np.exp(logits)
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

        def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            lr: float = 0.3,
            epochs: int = 200,
            batch_size: int = 32,
            seed: int = 0,
        ) -> None:
            rng = np.random.default_rng(seed)

            num_samples, input_dim = X_train.shape
            num_classes = int(y_train.max()) + 1

            # One-hot labels
            y_onehot = np.eye(num_classes)[y_train]

            for _ in range(epochs):
                indices = rng.permutation(num_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_onehot[indices]

                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    logits, h1 = self._forward(X_batch)

                    # Softmax
                    logits_shift = logits - logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits_shift)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                    # Cross-entropy gradient
                    batch_size_actual = X_batch.shape[0]
                    grad_logits = (probs - y_batch) / batch_size_actual

                    # Gradients for second layer
                    grad_W2 = h1.T @ grad_logits
                    grad_b2 = grad_logits.sum(axis=0)

                    # Backprop into first layer
                    grad_h1 = grad_logits @ self.W2.T
                    grad_z1 = grad_h1 * (h1 > 0)

                    grad_W1 = X_batch.T @ grad_z1
                    grad_b1 = grad_z1.sum(axis=0)

                    # Parameter updates
                    self.W2 -= lr * grad_W2
                    self.b2 -= lr * grad_b2
                    self.W1 -= lr * grad_W1
                    self.b1 -= lr * grad_b1

    input_dim = X.shape[1]
    num_classes = int(y.max()) + 1

    mlp = SimpleMLP(
        input_dim=input_dim, hidden_dim=16, output_dim=num_classes, seed=0
    )
    mlp.train(
        X,
        y,
        lr=0.3,
        epochs=int(num_training_steps),
        batch_size=32,
        seed=1,
    )

    # Report training accuracy just for confirmation
    train_pred = np.argmax(mlp.predict_proba(X), axis=1)
    train_accuracy = float((train_pred == y).mean())

    train_accuracy
    return mlp, num_classes


@app.cell
def _():
    """
    Map the 7 output classes into 7 distinct water taste profiles.

    These will be used both for display and for constructing prompts for
    the image generation model.
    """

    taste_profiles = {
        0: {
            "name": "Crisp Glacier Melt",
            "description": (
                "Ultra-clean, almost flavorless water with a faint hint of cold "
                "stone and alpine air. Very light, smooth, and refreshing."
            ),
        },
        1: {
            "name": "Mineral-Rich Spring",
            "description": (
                "Water with a noticeable mineral backbone — calcium and magnesium "
                "notes, slightly chalky yet pleasantly grounding."
            ),
        },
        2: {
            "name": "Citrus-Infused Spa Water",
            "description": (
                "Bright, zesty water with hints of lemon and lime peel, giving a "
                "lively, invigorating sensation."
            ),
        },
        3: {
            "name": "Smoky Volcanic Spring",
            "description": (
                "A bold, unusual profile with subtle sulfurous and smoky notes, "
                "evoking volcanic rock and geothermal pools."
            ),
        },
        4: {
            "name": "Ocean Breeze Salinity",
            "description": (
                "Lightly briny water reminiscent of sea spray and coastal air, "
                "saline yet still drinkable and refreshing."
            ),
        },
        5: {
            "name": "Floral Mountain Stream",
            "description": (
                "Delicate floral hints, like wildflowers near a mountain brook, "
                "with soft sweetness and gentle aroma."
            ),
        },
        6: {
            "name": "Earthy Deep Well",
            "description": (
                "Robust, earthy water with faint clay and stone notes, evoking "
                "an old stone well and damp soil after rain."
            ),
        },
    }
    return (taste_profiles,)


@app.cell
def _(taste_profiles):
    """
    Helpers to map the MLP's predicted class index into a taste profile and
    to construct a prompt template for image generation.
    """


    def class_to_taste(class_index: int):
        index = int(class_index)
        return taste_profiles[index]


    def build_prompt_for_class(class_index: int) -> str:
        index = int(class_index)
        profile = taste_profiles[index]

        prompt = (
            "Create a single, highly detailed, photorealistic image that "
            "visually represents the taste of water.\n\n"
            f"Taste name: {profile['name']}\n"
            f"Taste description: {profile['description']}\n\n"
            "Express this taste through environment, colors, lighting, textures, "
            "and mood. Show water as the central subject. "
            "Do not include any text, labels, or typography in the image."
        )

        return prompt
    return build_prompt_for_class, class_to_taste


@app.cell
def _(input_class_names, mo):
    """
    UI control for selecting one of the 7 input classes.
    """

    # Use human-readable labels as the option names, e.g.
    # "0: Class 0 – crisp & balanced profile".
    labels = [f"{i}: {label}" for i, label in enumerate(input_class_names)]

    # The initial value must be one of the option names.
    default_label = labels[0]

    input_selector = mo.ui.dropdown(
        options=labels,
        value=default_label,
        label="Input water class (0–6)",
    )

    input_selector
    return (input_selector,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(class_to_taste, input_selector, mlp, np, num_classes):
    """
    Use the trained MLP to map the selected input class into an output
    taste class. Display logits, probabilities, and the chosen taste
    profile.
    """

    # Dropdown returns a human-readable label like
    # "0: Class 0 – crisp & balanced profile". Parse the
    # integer class index from the prefix before the colon.
    selected_label = input_selector.value
    selected_index = int(str(selected_label).split(":", 1)[0])

    # Build a one-hot input for the chosen class.
    x = np.eye(num_classes)[[selected_index]]

    logits = mlp.predict_logits(x)[0]
    probs = mlp.predict_proba(x)[0]

    predicted_class = int(np.argmax(probs))
    taste = class_to_taste(predicted_class)

    summary = {
        "selected_input_class": selected_index,
        "logits": logits.tolist(),
        "probabilities": probs.tolist(),
        "predicted_output_class": predicted_class,
        "taste_name": taste["name"],
        "taste_description": taste["description"],
    }

    summary
    return (predicted_class,)


@app.cell
def _(build_prompt_for_class, predicted_class):
    """
    Build the exact prompt string that will be sent to the image model
    for the current predicted taste.
    """

    prompt = build_prompt_for_class(predicted_class)
    prompt
    return (prompt,)


@app.cell
def _(json):
    """
    Low-level helper for calling the OpenRouter API using an
    OpenAI-compatible chat completions endpoint with an image-capable model.
    """

    import urllib.request
    import urllib.error


    def generate_taste_image(
        prompt: str,
        api_key: str,
        model: str = "google/gemini-2.5-flash-image",
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> str:
        """
        Call OpenRouter's OpenAI-compatible API to generate an image.

        Returns a data URL or URL string for the generated image.
        """
        url = f"{base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image"],
            "image_config": {"aspect_ratio": "1:1"},
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"HTTP error from OpenRouter: {e.code} {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Network error while calling OpenRouter: {e}"
            ) from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from OpenRouter: {raw}") from e

        try:
            image_url = parsed["choices"][0]["message"]["images"][0]["image_url"][
                "url"
            ]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected OpenRouter response structure: {parsed}"
            ) from e

        return image_url
    return (generate_taste_image,)


@app.cell(hide_code=True)
def _(mo):
    """
    UI control for triggering the OpenRouter image generation.
    """

    run_button = mo.ui.run_button(label="Generate taste image with OpenRouter")

    run_button
    return (run_button,)


@app.cell
def _(generate_taste_image, mo, os, prompt, run_button):
    """
    Interactive step: call OpenRouter with the constructed prompt to generate
    an image that represents the taste of the water.

    Uses a run_button so that the API is only called when explicitly requested.
    """

    if not run_button.value:
        mo.stop(
            "Click the button above to call OpenRouter's image model. "
            "Make sure OPENROUTER_API_KEY is set in your environment."
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        mo.stop(
            "OPENROUTER_API_KEY is not set. "
            "Set it in your environment and re-run this cell."
        )

    try:
        image_url = generate_taste_image(prompt, api_key=api_key)
    except Exception as e:  # noqa: BLE001
        mo.stop(f"Error while calling OpenRouter: {e}")

    # `image_url` is a data URL (e.g. data:image/png;base64,...);
    # use marimo's built-in image helper to render it.
    mo.image(src=image_url, width=512, rounded=True)


if __name__ == "__main__":
    app.run()
