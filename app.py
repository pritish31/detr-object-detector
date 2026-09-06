"""
DETR Object Detector — Gradio demo
-----------------------------------
Upload an image, get back the same image with bounding boxes drawn around
every object DETR detects, using the official COCO-pretrained checkpoint
from Hugging Face (facebook/detr-resnet-50). No training required.

Run locally:
    pip install -r requirements.txt
    python app.py

Deploy on Hugging Face Spaces:
    See README.md in this folder.
"""

import colorsys
from collections import Counter

# ---------------------------------------------------------------------------
# Workaround for a bug in gradio_client 1.3.x: json_schema_to_python_type()
# crashes with "TypeError: argument of type 'bool' is not iterable" when a
# component's API schema contains a boolean `additionalProperties`. That crash
# 500s Gradio's own startup self-check, which then aborts launch() with a
# misleading "localhost is not accessible" error. Guard get_type() against
# non-dict (boolean) schemas so the API-info endpoint stops crashing.
import gradio_client.utils as _gc_utils

_gc_orig_get_type = _gc_utils.get_type


def _gc_safe_get_type(schema):
    if not isinstance(schema, dict):
        return "boolean"
    return _gc_orig_get_type(schema)


_gc_utils.get_type = _gc_safe_get_type
# ---------------------------------------------------------------------------

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor

MODEL_NAME = "facebook/detr-resnet-50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded once at startup, then reused for every request.
processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model = DetrForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def get_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct color per class id."""
    hue = (class_id * 0.61803398875) % 1.0  # golden-ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


@torch.no_grad()
def detect_objects(image: Image.Image, confidence: float):
    """Run DETR on `image`; return an annotated copy + a list of (label, score)."""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    # (height, width) per image, required by the post-processor to rescale
    # boxes from the model's internal resolution back to the original image.
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence
    )[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = load_font()

    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(v) for v in box.tolist()]
        label = model.config.id2label[label_id.item()]
        conf = score.item()
        color = get_color(label_id.item())

        draw.rectangle(box, outline=color, width=3)

        caption = f"{label} {conf:.2f}"
        text_box = draw.textbbox((box[0], box[1]), caption, font=font)
        # Nudge the label above the box; clamp so it never draws off-image.
        label_top = max(0, text_box[1] - 18)
        draw.rectangle(
            [box[0], label_top, text_box[2] + 4, label_top + (text_box[3] - text_box[1]) + 4],
            fill=color,
        )
        draw.text((box[0] + 2, label_top + 2), caption, fill="white", font=font)

        detections.append((label, conf))

    return annotated, detections


# --------------------------------------------------------------------------- #
#  App
# --------------------------------------------------------------------------- #
INTRO = (
    "Upload an image to begin. Detection runs automatically, and you can "
    "re-run at any time by adjusting the confidence threshold."
)


def run_detection(image, confidence):
    """Gradio callback: annotate the image and summarise what DETR found."""
    if image is None:
        return None, INTRO

    annotated, detections = detect_objects(image, confidence)

    if not detections:
        return annotated, (
            "**No objects found above this threshold.** "
            "Try lowering the confidence slider."
        )

    counts = Counter(label for label, _ in detections)
    n = len(detections)
    headline = f"**{n} object{'s' if n != 1 else ''} detected** — "
    headline += " · ".join(f"{count}× {label}" for label, count in counts.most_common())

    table = "\n".join(
        f"| {label} | {conf:.1%} |"
        for label, conf in sorted(detections, key=lambda d: d[1], reverse=True)
    )
    table = "\n\n| Object | Confidence |\n| --- | --- |\n" + table
    return annotated, headline + table


CREDIT = (
    "Model <b>facebook/detr-resnet-50</b> &nbsp;·&nbsp; pretrained on COCO "
    "(80 classes) &nbsp;·&nbsp; "
    "<a href='https://arxiv.org/abs/2005.12872'>End-to-End Object Detection with Transformers</a>"
)

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
footer { display: none !important; }

#hero { padding: 24px 4px 8px; }
#hero h1 { margin: 0 0 6px; font-size: 1.5rem; font-weight: 700; }
#hero p  { margin: 0; max-width: 640px; line-height: 1.5;
    color: var(--body-text-color-subdued); }

.section-label { margin: 6px 2px 0; font-size: .72rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    color: var(--body-text-color-subdued); }

#credit { margin-top: 8px; padding-top: 12px;
    border-top: 1px solid var(--border-color-primary);
    text-align: center; font-size: .8rem; color: var(--body-text-color-subdued); }
#credit a { color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="DETR Object Detector", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML(
        "<div id='hero'>"
        "<h1>DETR Object Detector</h1>"
        "<p>End-to-end object detection with a Transformer — no anchor boxes, "
        "no non-maximum suppression. Upload a photo and DETR marks every object "
        "it recognises across 80 COCO categories.</p>"
        "</div>"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            gr.HTML("<div class='section-label'>Input</div>")
            image_input = gr.Image(type="pil", label="Image", height=340)
            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.7,
                step=0.05,
                label="Confidence threshold",
                info="Lower to surface more objects · raise to cut false positives",
            )
            run_btn = gr.Button("Detect objects", variant="primary")

        with gr.Column(scale=6):
            gr.HTML("<div class='section-label'>Detections</div>")
            image_output = gr.Image(label="Annotated image", height=340)
            summary = gr.Markdown(INTRO)

    gr.HTML(f"<div id='credit'>{CREDIT}</div>")

    _inputs = [image_input, confidence]
    _outputs = [image_output, summary]
    run_btn.click(run_detection, _inputs, _outputs)
    image_input.change(run_detection, _inputs, _outputs)
    confidence.release(run_detection, _inputs, _outputs)

if __name__ == "__main__":
    demo.launch(show_error=True)
