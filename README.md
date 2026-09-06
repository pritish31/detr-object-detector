---
title: DETR Object Detector
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# DETR Object Detector

A one-page Gradio app: upload a photo, get back the same photo with bounding
boxes around every object DETR finds. Uses the official COCO-pretrained
checkpoint (`facebook/detr-resnet-50`) from Hugging Face — **no training
required**.

This is the companion app for the article *"Following an Image Through
DETR."* If you've read that, this is the exact model, running.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | The Gradio app (loads the model once, exposes an upload → detect UI) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file — also doubles as the Space's config (the `---` block at the top) |

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

The first run downloads the pretrained weights from Hugging Face
(~160 MB) and caches them locally — after that, startup is fast. Then open
the local URL Gradio prints (usually `http://127.0.0.1:7860`).

**Note:** the app needs internet access on first run to download the model
from `huggingface.co`. After the first run, the weights are cached in
`~/.cache/huggingface`, so it works offline from then on.

## Deploy to Hugging Face Spaces (recommended — free CPU tier is enough)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) and create a new Space.
   - **SDK:** Gradio
   - **Hardware:** CPU basic is fine (inference on a single image takes a
     couple of seconds on CPU; DETR is lightweight)
2. Either:
   - **Web UI:** upload `app.py`, `requirements.txt`, and `README.md` directly through the "Files" tab, or
   - **Git:**
     ```bash
     git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
     cd <your-space-name>
     cp /path/to/app.py /path/to/requirements.txt /path/to/README.md .
     git add .
     git commit -m "Add DETR object detector"
     git push
     ```
3. The Space will build automatically (installs `requirements.txt`, then
   runs `app.py`). First build takes a few minutes; after that it's cached.
4. Your app will be live at two URLs:
   - the Space page: `https://huggingface.co/spaces/<your-username>/<your-space-name>`
   - the bare app (used for embedding): `https://<your-username>-<your-space-name>.hf.space`

   Keep the Space **public** so it can be embedded without a login.

### Notes on the free CPU tier

- **Cold starts:** a free Space sleeps after ~48 h of no traffic and takes
  ~30–60 s to wake on the next visit. Fine for a portfolio; pay for
  "always-on" hardware if you need it instant.
- Inference itself is a couple of seconds per image on CPU.

## Embed it on your personal website

Once the Space is public, add it to any page with the Gradio web component
(responsive, resizes to its content — recommended):

```html
<script
  type="module"
  src="https://gradio.s3-us-west-2.amazonaws.com/4.44.1/gradio.js"
></script>

<gradio-app src="https://<your-username>-<your-space-name>.hf.space"></gradio-app>
```

The `gradio.js` version must match `sdk_version` above. Or use a plain
iframe if you'd rather not load a script:

```html
<iframe
  src="https://<your-username>-<your-space-name>.hf.space"
  frameborder="0"
  width="100%"
  height="900"
  title="DETR Object Detector"
></iframe>
```

Hugging Face also generates both snippets for you: on the Space, open the
`⋮` menu (top right) → **Embed this Space**.

## Swapping in your own classes later

Right now this uses the standard 80 COCO classes DETR was pretrained on
(people, cars, dogs, chairs, etc. — see the full list in
`model.config.id2label` after loading). If you later want to detect your
*own* custom categories, you'd fine-tune this checkpoint on a labeled
dataset of your classes rather than starting from scratch — that's a
separate, much smaller training job than pretraining, and this same
`app.py` would only need its `MODEL_NAME` swapped to point at your
fine-tuned checkpoint.

## Credits

- Model: [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50) on the Hugging Face Hub
- Paper: Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020 — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Official implementation: [facebookresearch/detr](https://github.com/facebookresearch/detr)
