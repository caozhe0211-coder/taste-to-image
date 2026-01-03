import marimo

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
    import urllib.request as urllib_request
    import urllib.error as urllib_error
    return json, np, os, urllib_error, urllib_request


@app.cell
def _(json, os):
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

    return num_taste_categories, num_water_types, taste_categories, water_type_names


@app.cell
def _(np, num_taste_categories, num_water_types, os):
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

    return X_taste, X_water, y_taste, y_water, water_indices, taste_indices


@app.cell
def _():
    """
    Training configuration for the MLP.

    `num_training_steps` controls how many epochs the network trains for.
    """

    num_training_steps = 100
    return (num_training_steps,)


@app.cell
def _(np, num_training_steps, y_taste, X_water):
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

    train_accuracy
    return SimpleMLP, mlp, num_classes, train_accuracy


@app.cell
def _(taste_categories):
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

    input_selector
    return (input_selector,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(SimpleMLP, X_taste, np, num_training_steps, y_water):
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

    inv_train_accuracy
    return inv_train_accuracy, mlp_inv


@app.cell
def _(class_to_taste, input_selector, mlp, np, num_water_types):
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

    summary
    return (predicted_class,)


@app.cell
def _(mo, taste_categories):
    """
    Persist the free-form text input across cell reruns.
    """

    get_taste_text, set_taste_text = mo.state("")
    return get_taste_text, set_taste_text


@app.cell
def _(get_taste_text, mo, set_taste_text, taste_categories):
    """
    Free-form taste description input.
    """

    taste_text = mo.ui.text_area(
        label="Describe the taste (free form)",
        placeholder="e.g. refreshing, a bit bitter, mineral...",
        value=get_taste_text(),
        rows=4,
        debounce=False,
        on_change=set_taste_text,
    )

    taste_text
    return (taste_text,)


@app.cell
def _(mo, num_taste_categories, taste_categories):
    """
    Fallback structured input (when no API key / as a control baseline).
    """

    options = [f"{i}: {taste_categories[i]}" for i in range(num_taste_categories)]
    taste_selector = mo.ui.dropdown(
        options=options,
        value=options[0],
        label="Or pick a taste category directly (0–8)",
    )

    taste_selector
    return (taste_selector,)


@app.cell
def _(
    get_taste_text,
    json,
    mo,
    np,
    num_taste_categories,
    taste_categories,
    taste_selector,
    taste_text_to_categories,
    mlp_inv,
    water_type_names,
    os,
):
    """
    Inverse recommendation: free-form taste -> categories -> MLP -> top-3 waters.

    We aggregate logits across the inferred categories to produce one overall
    ranking (handles overlaps naturally).
    """

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    text_value = str(get_taste_text() or "").strip()

    # Decide which categories to use.
    inferred_categories: list[int] = []
    used_fallback = False

    if text_value:
        if openrouter_api_key:
            try:
                inferred_categories = taste_text_to_categories(
                    taste_text=text_value,
                    taste_categories=taste_categories,
                    api_key=openrouter_api_key,
                    max_categories=3,
                )
            except Exception as e:  # noqa: BLE001
                used_fallback = True
                inferred_categories = [
                    int(str(taste_selector.value).split(":", 1)[0])
                ]
                mo.md(
                    f"⚠️ Could not map free-form text via LLM ({e}). "
                    "Falling back to the selected category dropdown."
                )
        else:
            used_fallback = True
            inferred_categories = [int(str(taste_selector.value).split(":", 1)[0])]
            mo.md(
                "ℹ️ OPENROUTER_API_KEY not set; using the selected category dropdown."
            )
    else:
        used_fallback = True
        inferred_categories = [int(str(taste_selector.value).split(":", 1)[0])]

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
        "used_fallback_category_dropdown": bool(used_fallback),
        "inferred_taste_categories": inferred_categories,
        "inferred_taste_category_names": [
            taste_categories[int(c)] for c in inferred_categories
        ],
        "top_3_recommendations": recs,
    }

    result
    return


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
def _(json, urllib_error, urllib_request):
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
def _(mo, os):
    """
    Debug: show whether OPENROUTER_API_KEY is visible to the Marimo app.
    """

    has_key = bool(os.getenv("OPENROUTER_API_KEY"))

    status = (
        "✅ OPENROUTER_API_KEY detected in this Marimo session."
        if has_key
        else "⚠️ OPENROUTER_API_KEY not detected in this Marimo session."
    )
    mo.md(status)


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
    mo.image(src=image_url, width=512, rounded=True)
    return




@app.cell
def _(json, urllib_error, urllib_request):
    """
    Helper: use OpenRouter chat completions to map free-form taste text
    into one or more taste categories (0–8).

    We keep the output machine-readable and validate it strictly.
    """

    def _openrouter_chat(
        *,
        api_key: str,
        messages: list[dict],
        model: str = "google/gemini-2.5-flash",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: int = 60,
    ) -> str:
        url = f"{base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
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
            with urllib_request.urlopen(request, timeout=timeout_s) as response:
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

        parsed = json.loads(raw)
        try:
            return parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:  # noqa: PERF203
            raise RuntimeError(
                f"Unexpected OpenRouter response structure: {parsed}"
            ) from e

    def taste_text_to_categories(
        *,
        taste_text: str,
        taste_categories: dict[int, str],
        api_key: str,
        model: str = "google/gemini-2.5-flash",
        max_categories: int = 3,
    ) -> list[int]:
        """
        Returns a list of taste category indices (0–8), length 1..max_categories.
        """

        categories_list = "\n".join(
            [f"{i}: {taste_categories[i]}" for i in sorted(taste_categories)]
        )

        system = (
            "You map a user's free-form water taste description into the "
            "closest taste category indices from a fixed list.\n\n"
            "Rules:\n"
            f"- Return ONLY valid JSON.\n"
            f"- Output schema: {{\"categories\": [int, ...]}}.\n"
            f"- categories length: 1 to {int(max_categories)}.\n"
            "- Each category must be one of the listed indices.\n"
        )

        user = (
            "Taste categories:\n"
            f"{categories_list}\n\n"
            "User description:\n"
            f"{taste_text}\n\n"
            "Return JSON now."
        )

        content = _openrouter_chat(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM returned non-JSON content: {content}"
            ) from e

        cats = parsed.get("categories")
        if not isinstance(cats, list) or not cats:
            raise RuntimeError(f"Invalid categories payload: {parsed}")

        cleaned: list[int] = []
        for c in cats[: int(max_categories)]:
            if isinstance(c, bool) or not isinstance(c, int):
                continue
            if c in taste_categories and c not in cleaned:
                cleaned.append(int(c))

        if not cleaned:
            raise RuntimeError(f"No valid categories parsed from: {parsed}")

        return cleaned

    return taste_text_to_categories,


if __name__ == "__main__":
    app.run()
