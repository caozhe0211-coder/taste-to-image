import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    """
    Overview for non-technical readers.
    """
    import marimo as mo

    mo.md(
        "# Water Taste Lab (Demo Notebook)\n\n"
        "This notebook is a small, experimental demo that connects **water taste**, "
        "**data-driven recommendations**, and **AI image generation**. The goal is to "
        "make water taste easier to describe, easier to compare, and even easier to "
        "visualize.\n\n"
        "**Flow of the experiment:**\n"
        "1) Collect taste notes and group them into a few easy categories.\n"
        "2) Train a small model to learn how taste categories map to water types.\n"
        "3) Let people pick the tastes they want, then recommend matching waters.\n"
        "4) Use a modern image model to visualize what a taste might *look like*.\n\n"
        "This is a proof‑of‑concept for communication and exploration. "
        "It prioritizes clarity over technical detail. Therefore we by default folds all the codes instead we write a small explanation on what has been done in the code. You can feel free to expand any code cell to see the exact implementation details."
    )
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    """
    Common imports shared across the app.
    """

    import numpy as np
    import os
    import json
    import urllib.request as urllib_request
    import urllib.error as urllib_error
    return json, np, os, urllib_error, urllib_request


@app.cell(hide_code=True)
def _(mo):
    """
    Section: data and categories.
    """

    mo.md(
        "## Step 1: Taste data and categories\n\n"
        "Start with a small dataset of water taste notes. "
        "Those notes are grouped into a small set of taste categories so people "
        "can talk about water in a simple, consistent way.\n\n"
        "If you're curious, you can expand the code cell below to see the exact steps."
    )
    return


@app.cell(hide_code=True)
def _(json, mo, os):
    """
    Load canonical labels (water types and taste categories).

    `labels.json` is the single source of truth for display names.
    """

    labels_path = "labels.json"
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Missing {labels_path}. It defines canonical labels."
        )

    with open(labels_path, encoding="utf-8") as f:
        label_data = json.load(f)

    water_type_names = label_data.get("water_types")
    taste_category_names = label_data.get("taste_categories")

    if not isinstance(water_type_names, list) or not water_type_names:
        raise ValueError(f"Invalid water_types in {labels_path}")
    if not isinstance(taste_category_names, list) or not taste_category_names:
        raise ValueError(f"Invalid taste_categories in {labels_path}")

    num_water_types = len(water_type_names)

    taste_categories = {i: str(name) for i, name in enumerate(taste_category_names)}
    num_taste_categories = len(taste_categories)
    mo.md(
        "Load the official names for water types and taste categories. "
        "This keeps labels consistent everywhere in the notebook."
    )
    return (
        num_taste_categories,
        num_water_types,
        taste_categories,
        water_type_names,
    )


@app.cell(hide_code=True)
def _(mo, np, num_taste_categories, num_water_types, os):
    """
    Load the real water→taste dataset for training.

    The dataset is stored in `water_taste_data.csv` with columns:
      water_type_idx (0–7)
      taste_idx (0–8)

    Returns indices for both directions, plus one-hot inputs for the forward and
    inverse models.
    """
    data_path = "water_taste_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found.")

    # Load with header handling.
    try:
        data = np.genfromtxt(
            data_path, delimiter=",", dtype=int, skip_header=1
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load {data_path}: {e}") from e

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns (water_type_idx, taste_idx), got {data.shape[1]}"
        )

    water_indices = data[:, 0].astype(int)
    taste_indices = data[:, 1].astype(int)

    if water_indices.min() < 0 or water_indices.max() >= num_water_types:
        raise ValueError(
            "water_type_idx in dataset must be in the range "
            f"[0, {num_water_types - 1}]"
        )

    if taste_indices.min() < 0 or taste_indices.max() >= num_taste_categories:
        raise ValueError(
            "taste_idx in dataset must be in the range "
            f"[0, {num_taste_categories - 1}]"
        )

    # Forward model: water -> taste
    X_water = np.eye(num_water_types)[water_indices]
    y_taste = taste_indices

    # Inverse model: taste -> water
    X_taste = np.eye(num_taste_categories)[taste_indices]
    y_water = water_indices

    mo.md(
        "Load the taste dataset and do a quick cleanup check. "
        "This turns the raw rows into the simple format the model can learn from."
    )
    return X_taste, X_water, y_taste, y_water


@app.cell(hide_code=True)
def _(mo):
    """
    Training configuration for the MLP.

    `num_training_steps` controls how many epochs the network trains for.
    """
    num_training_steps = 100
    mo.md(
        "Set how long the small learning model should train. "
        "More steps can help it learn patterns, but too many can overfit."
    )
    return (num_training_steps,)


@app.cell(hide_code=True)
def _(mo):
    """
    Section: learn taste patterns.
    """

    mo.md(
        "## Step 2: Learn taste patterns (small model)\n\n"
        "Train a tiny model that learns how water types relate to the taste "
        "categories. It’s not meant to be perfect—just good enough to show the idea.\n\n"
        "If you're curious, you can expand the code cell below to see the exact steps."
    )
    return


@app.cell(hide_code=True)
def _(X_water, mo, np, num_training_steps, y_taste):
    """
    Define and train a simple 2-layer MLP classifier that maps the
    one-hot water type into taste categories.

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


    fwd_input_dim = X_water.shape[1]
    num_classes = int(y_taste.max()) + 1

    mlp = SimpleMLP(
        input_dim=fwd_input_dim, hidden_dim=16, output_dim=num_classes, seed=0
    )
    mlp.train(
        X_water,
        y_taste,
        lr=0.3,
        epochs=int(num_training_steps),
        batch_size=32,
        seed=1,
    )

    # Report training accuracy just for confirmation
    fwd_train_pred = np.argmax(mlp.predict_proba(X_water), axis=1)
    train_accuracy = float((fwd_train_pred == y_taste).mean())

    mo.vstack(
        [
            mo.md(
                "Train a tiny model that learns a simple mapping: water type → taste. "
                "Think of it as a lightweight pattern‑finder, not a perfect predictor."
            ),
            mo.md(
                f"Training accuracy (water → taste): **{train_accuracy:.1%}**"
            ),
            mo.md(
                "A low number here is expected because the dataset is small and the "
                "taste labels are coarse. The goal is to prove the workflow, not to "
                "claim a production‑ready model."
            ),
        ]
    )
    return SimpleMLP, mlp


@app.cell(hide_code=True)
def _(mo, taste_categories):
    """
    Map the taste categories into prompt-friendly profiles.

    These will be used both for display and for constructing prompts for
    the image generation model.
    """

    # Keep a small amount of "imagery" per category to help the image model.
    # These are intentionally simple and closely aligned with mapping.md labels.
    taste_profiles = {}
    for idx, name in taste_categories.items():
        taste_profiles[int(idx)] = {
            "name": name,
            "description": (
                "Create an evocative visual metaphor for this taste category, "
                "using lighting, textures, and environment cues."
            ),
        }

    mo.md(
        "Attach a short, human‑readable profile to each taste category. "
        "These become the words we later use to describe and visualize tastes."
    )
    return (taste_profiles,)


@app.cell(hide_code=True)
def _(mo, taste_profiles):
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

    mo.md(
        "Create helper functions so tastes can be translated into prompts "
        "for the image model later."
    )
    return build_prompt_for_class, class_to_taste


@app.cell(hide_code=True)
def _(mo):
    """
    Section: choose taste preferences.
    """

    mo.md(
        "## Step 3: Choose tastes and get recommendations\n\n"
        "Pick one or more taste categories. The system treats these as your "
        "target preferences and immediately recommends matching waters below.\n\n"
        "If you're curious, you can expand the code cell below to see the exact steps."
    )
    return


@app.cell(hide_code=True)
def _(mo, water_type_names):
    """
    UI control for selecting a water type (0–7).
    """
    # Use human-readable labels as the option names, e.g.
    # "0: Class 0 – crisp & balanced profile".
    water_labels = [f"{i}: {label}" for i, label in enumerate(water_type_names)]

    # The initial value must be one of the option names.
    default_label = water_labels[0]

    input_selector = mo.ui.dropdown(
        options=water_labels,
        value=default_label,
        label="Water type (0–7)",
    )

    mo.vstack(
        [
            mo.md(
                "Pick a specific water type to see the model’s predicted taste profile. "
                "This is a quick way to sanity‑check the learned mapping."
            ),
            input_selector,
        ]
    )
    return (input_selector,)


@app.cell(hide_code=True)
def _(SimpleMLP, X_taste, mo, np, num_training_steps, y_water):
    """
    Train the inverse model: taste category -> water type recommendation.

    Input: one-hot taste (length 9)
    Output: water type index (0–7)
    """
    inv_input_dim = X_taste.shape[1]
    num_water = int(y_water.max()) + 1

    mlp_inv = SimpleMLP(
        input_dim=inv_input_dim, hidden_dim=16, output_dim=num_water, seed=2
    )
    mlp_inv.train(
        X_taste,
        y_water,
        lr=0.3,
        epochs=int(num_training_steps),
        batch_size=32,
        seed=3,
    )

    inv_train_pred = np.argmax(mlp_inv.predict_proba(X_taste), axis=1)
    inv_train_accuracy = float((inv_train_pred == y_water).mean())

    mo.vstack(
        [
            mo.md(
                "Train the reverse direction: taste → recommended water. "
                "This is the core of the recommendation step."
            ),
            mo.md(
                f"Training accuracy (taste → water): **{inv_train_accuracy:.1%}**"
            ),
            mo.md(
                "A low number here is expected with a tiny dataset. The key point "
                "is that the pipeline still runs end‑to‑end and produces a working demo."
            ),
        ]
    )
    return (mlp_inv,)


@app.cell(hide_code=True)
def _(class_to_taste, input_selector, mlp, mo, np, num_water_types):
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
    water_x = np.eye(num_water_types)[[selected_index]]

    water_logits = mlp.predict_logits(water_x)[0]
    water_probs = mlp.predict_proba(water_x)[0]

    predicted_class = int(np.argmax(water_probs))
    taste = class_to_taste(predicted_class)

    summary = {
        "selected_input_class": selected_index,
        "logits": water_logits.tolist(),
        "probabilities": water_probs.tolist(),
        "predicted_output_class": predicted_class,
        "taste_name": taste["name"],
        "taste_description": taste["description"],
    }

    mo.vstack(
        [
            mo.md(
                "Predicted taste details for the selected water. "
                "This shows the model’s best guess and its confidence scores."
            ),
            mo.json(summary),
        ]
    )
    return (predicted_class,)


@app.cell(hide_code=True)
def _(mo):
    """
    Persist selected taste categories across reruns.
    """
    get_selected_tastes, set_selected_tastes = mo.state([])
    mo.md(
        "Remember selected taste categories so you don’t lose them "
        "when the notebook reruns."
    )
    return get_selected_tastes, set_selected_tastes


@app.cell(hide_code=True)
def _(get_selected_tastes, mo, set_selected_tastes, taste_categories):
    """
    Checkbox list input for desired taste categories.
    """
    names = [taste_categories[i] for i in sorted(taste_categories)]
    selected_taste_set = set(get_selected_tastes())

    def _toggle(name: str):
        def _handler(checked: bool):
            current = set(get_selected_tastes())
            if checked:
                current.add(name)
            else:
                current.discard(name)
            set_selected_tastes(sorted(current))

        return _handler

    checkboxes = [
        mo.ui.checkbox(
            label=name,
            value=name in selected_taste_set,
            on_change=_toggle(name),
        )
        for name in names
    ]

    mo.vstack(
        [
            mo.md("Pick one or more taste categories (check all that apply):"),
            mo.md(
                "Pick the taste categories you’re aiming for. "
                "This becomes the input to the recommendation step."
            ),
            mo.vstack(checkboxes),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    get_selected_tastes,
    mlp_inv,
    mo,
    np,
    num_taste_categories,
    taste_categories,
    water_type_names,
):
    """
    Inverse recommendation: selected tastes -> MLP -> top-3 waters.

    Aggregate logits across the inferred categories to produce one overall
    ranking (handles overlaps naturally).
    """
    name_to_index = {taste_categories[i]: i for i in taste_categories}
    selected_taste_names = get_selected_tastes() or []
    inferred_categories = [
        int(name_to_index[name])
        for name in selected_taste_names
        if name in name_to_index
    ]

    if not inferred_categories:
        mo.stop("Please select at least one taste category.")

    # Aggregate logits across categories (simple average of logits).
    inv_logits_sum = None
    for c in inferred_categories:
        inv_x = np.eye(num_taste_categories)[[int(c)]]
        inv_logits = mlp_inv.predict_logits(inv_x)[0]
        inv_logits_sum = (
            inv_logits
            if inv_logits_sum is None
            else (inv_logits_sum + inv_logits)
        )

    logits_avg = inv_logits_sum / max(1, len(inferred_categories))

    # Convert to probabilities via softmax.
    logits_shift = logits_avg - logits_avg.max()
    exp_logits = np.exp(logits_shift)
    inv_probs = exp_logits / exp_logits.sum()

    topk = 3
    top_indices = np.argsort(-inv_probs)[:topk].tolist()

    recs = [
        {
            "water_type_idx": int(i),
            "water_name": water_type_names[int(i)]
            if int(i) < len(water_type_names)
            else f"water_type_{int(i)}",
            "probability": float(inv_probs[int(i)]),
        }
        for i in top_indices
    ]

    result = {
        "inferred_taste_categories": inferred_categories,
        "inferred_taste_category_names": [
            taste_categories[int(c)] for c in inferred_categories
        ],
        "top_3_recommendations": recs,
    }

    mo.vstack(
        [
            mo.md(
                "Recommendation result (top matches based on your selected tastes). "
                "Combine your chosen tastes, score all water types, and list the top 3."
            ),
            mo.json(result),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Section: visualize taste with AI.
    """

    mo.md(
        "## Step 5: Visualize the taste with AI\n\n"
        "Use a state‑of‑the‑art image model to create a visual metaphor for a "
        "taste. This helps people build intuition and discuss flavors more easily.\n\n"
        "If you're curious, you can expand the code cell below to see the exact steps."
    )
    return


@app.cell(hide_code=True)
def _(build_prompt_for_class, mo, predicted_class):
    """
    Build the exact prompt string that will be sent to the image model
    for the current predicted taste.
    """
    prompt = build_prompt_for_class(predicted_class)
    mo.vstack(
        [
            mo.md(
                "Prompt sent to the image model. This is the text description used to "
                "generate the visual taste metaphor."
            ),
            mo.md(f"```text\n{prompt}\n```"),
        ]
    )
    return (prompt,)


@app.cell(hide_code=True)
def _(json, mo, urllib_error, urllib_request):
    """
    Low-level helper for calling the OpenRouter API using an
    OpenAI-compatible chat completions endpoint with an image-capable model.
    """
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

        request = urllib_request.Request(
            url, data=data, headers=headers, method="POST"
        )

        try:
            with urllib_request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"HTTP error from OpenRouter: {e.code} {error_body}"
            ) from e
        except urllib_error.URLError as e:
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

    mo.md(
        "Define the API call for the image model so we can request "
        "a visual representation of a taste."
    )
    return (generate_taste_image,)


@app.cell(hide_code=True)
def _(mo):
    """
    UI control for triggering the OpenRouter image generation.
    """
    run_button = mo.ui.run_button(label="Generate taste image with OpenRouter")

    mo.vstack(
        [
            mo.md(
                "Add a button so image generation only happens when you want it to."
            ),
            run_button,
        ]
    )
    return (run_button,)


@app.cell(hide_code=True)
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

    image_api_key = os.getenv("OPENROUTER_API_KEY")
    if not image_api_key:
        mo.stop(
            "OPENROUTER_API_KEY is not set. "
            "Set it in your environment and re-run this cell."
        )

    try:
        image_url = generate_taste_image(prompt, api_key=image_api_key)
    except Exception as e:  # noqa: BLE001
        mo.stop(f"Error while calling OpenRouter: {e}")

    # `image_url` is a data URL (e.g. data:image/png;base64,...);
    # use marimo's built-in image helper to render it.
    mo.vstack(
        [
            mo.md(
                "Generated image output. This visual is the model’s attempt to represent "
                "the selected taste in a picture."
            ),
            mo.image(src=image_url, width=512, rounded=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Summary and limitations.
    """

    mo.md(
        "## Summary and limitations\n\n"
        "This demo shows that it’s possible to connect taste categories, data‑driven "
        "recommendations, and AI imagery in one workflow. The goal is to make taste "
        "more visible and easier to discuss.\n\n"
        "**Limitations:** this is an experiment with a small dataset and a small set "
        "of categories. For real‑world use, we would need much more data, richer "
        "taste categories, and broader validation.\n\n"
        "**What’s next:** collect more taste data, expand categories, and test with "
        "real users to improve accuracy and usefulness."
    )
    return


if __name__ == "__main__":
    app.run()
