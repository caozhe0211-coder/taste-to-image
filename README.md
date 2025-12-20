## Water taste classifier + image prompt demo (Marimo)

This project is an interactive **Marimo** app that:

1. Trains a small NumPy-based **MLP classifier** that maps 7 “input water classes” into 7 “taste classes”.
2. Shows the predicted taste profile (name + description) plus logits/probabilities.
3. Builds a **text prompt** describing the predicted taste.
4. (Optional) Calls **OpenRouter** to generate an image representing that taste prompt.

The goal is to demonstrate a simple end-to-end workflow: **choose an input class → classify → generate a prompt → optionally generate an image**.

---

## What’s in the repo

- `main.py` — The Marimo app (UI + training + prompt construction + optional OpenRouter call)
- `water_taste_data_dummy.csv` — Example dataset (input_class,output_class)
- `pyproject.toml` / `uv.lock` — Python dependencies (uses `marimo`)

Note: `main.py` expects a dataset file named `water_taste_data.csv` at runtime.

---

## Requirements

- Python `>=3.11`
- A way to install dependencies (`uv` recommended, but `pip` works too)
- (Optional) An OpenRouter API key if you want to generate images

---

## Setup

### 1) Create the dataset file `water_taste_data.csv`

`main.py` loads `water_taste_data.csv` (not the dummy file). The simplest way is to copy the provided dummy dataset:

```bash
cp water_taste_data_dummy.csv water_taste_data.csv
```

The CSV format is:

```text
input_class,output_class
```

Both values are integers in the range `0–6`.

### 2) Install dependencies

With `uv`:

```bash
uv sync
```

Or with `pip` (inside a virtual environment):

```bash
pip install -r <(uv pip compile pyproject.toml)
```

---

## Run the app

Start Marimo for `main.py`:

```bash
marimo run main.py
```

Then:

- Pick an **Input water class (0–6)** from the dropdown.
- Review the predicted taste class, probabilities, and the generated prompt.

---

## Optional: Image generation (OpenRouter)

This app can call OpenRouter’s OpenAI-compatible endpoint to generate an image using an image-capable model.

1) Set your API key in the environment:

```bash
export OPENROUTER_API_KEY="YOUR_KEY_HERE"
```

2) In the app, click **“Generate taste image with OpenRouter”**.

Notes:

- The default model is `google/gemini-2.5-flash-image` (see `generate_taste_image` in `main.py`).
- If the key is missing, the app will stop and prompt you to set `OPENROUTER_API_KEY`.

---

## How it works (high level)

- **Inputs**: A one-hot vector representing one of 7 abstract “water profiles”.
- **Model**: A tiny 2-layer MLP implemented in NumPy (ReLU hidden layer, softmax output).
- **Outputs**: One of 7 taste profiles (name + descriptive text).
- **Prompt**: A template that describes the taste visually (no text/labels requested in the image).

---

## Troubleshooting

- **`FileNotFoundError: Dataset file 'water_taste_data.csv' not found`**
  - Run: `cp water_taste_data_dummy.csv water_taste_data.csv`
- **OpenRouter call fails**
  - Verify `OPENROUTER_API_KEY` is set and valid.
  - Confirm you have network access and the selected model is available to your OpenRouter account.
