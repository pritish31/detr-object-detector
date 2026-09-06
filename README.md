---
title: DETR Object Detector
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# DETR Object Detector

Upload a photo, get it back with bounding boxes around every object DETR
finds, plus a per-object confidence breakdown. Uses the COCO-pretrained
`detr-resnet-50` checkpoint — **no training required**.

There are two versions in this repo:

| File | What it is | Where it runs |
|---|---|---|
| **`index.html`** | Self-contained web page. Runs `Xenova/detr-resnet-50` (ONNX) in the browser via [Transformers.js](https://github.com/huggingface/transformers.js). No server, no cost. **This is what the Space serves** (`sdk: static`). | The visitor's browser |
| **`app.py`** | The original Python [Gradio](https://gradio.app) app using `facebook/detr-resnet-50` + PyTorch. | A Python server (local, or a paid Gradio Space) |
| `requirements.txt` | Python deps for `app.py` (`gradio`, `torch`, `transformers`, `timm`, `pillow`). | — |

## The browser version (`index.html`)

Just open the file — no build step, no dependencies to install.

```bash
# any static server works, e.g.
python -m http.server 8000
# then visit http://localhost:8000/index.html
```

On first use it downloads the quantised ONNX model (~40 MB) from the
Hugging Face Hub and caches it in the browser's Cache Storage; subsequent
runs are instant. Inference is a few seconds on CPU (WebAssembly).

### Deploy it free on Hugging Face (Static Space)

Gradio/Docker Spaces now require a paid plan, but **Static Spaces are free
for everyone** and are all `index.html` needs.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   **SDK → Static**, **Public**.
2. Add the files — either drag `index.html` and `README.md` into the Space's
   **Files** tab (overwrite the auto-generated `README.md`), or push with git:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/detr-object-detector
   git push space main --force   # --force replaces the Space's starter commit
   ```
   When prompted: username = your HF username, password = a **write** token
   from <https://huggingface.co/settings/tokens>.
3. It goes live in a few seconds (no build) at:
   - `https://huggingface.co/spaces/<hf-username>/detr-object-detector`
   - `https://<hf-username>-detr-object-detector.hf.space` ← use this to embed

Static Spaces never sleep and have no cold start.

### Embed on your website

```html
<iframe
  src="https://<hf-username>-detr-object-detector.hf.space"
  style="width:100%;height:1000px;border:0;border-radius:12px"
  title="DETR Object Detector"
  allow="clipboard-read; clipboard-write"
></iframe>
```

The page is self-contained, so you can also just copy `index.html` straight
onto your own site instead of embedding the Space.

## The Python version (`app.py`)

```bash
pip install -r requirements.txt
python app.py
```

First run downloads `facebook/detr-resnet-50` (~160 MB) to
`~/.cache/huggingface`; after that it works offline. Opens a Gradio UI on
`http://127.0.0.1:7860`.

To host `app.py` online you need a Python server that can run PyTorch — a
**paid** Gradio Space (HF PRO), or a VM / container host. The free static
route above avoids all of that.

## Swapping in your own classes later

Both versions use the standard 80 COCO classes. To detect your own
categories you'd fine-tune the checkpoint on a labelled dataset of your
classes (a small job compared to pretraining) and point `MODEL_NAME`
(`app.py`) or the model id (`index.html`) at your fine-tuned checkpoint.

## Credits

- Model: [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50) · ONNX port [`Xenova/detr-resnet-50`](https://huggingface.co/Xenova/detr-resnet-50)
- Paper: Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020 — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Official implementation: [facebookresearch/detr](https://github.com/facebookresearch/detr)
