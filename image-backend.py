import os, io, base64
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import openai

# ── CONFIG ──────────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")           # make sure .env is set

# ── FLASK APP & CORS ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://ai-image-modifier.web.app"]}})

# ── UTILITIES ───────────────────────────────────────────────────────────────
def to_square_rgba(img: Image.Image, size: int = 1024) -> Image.Image:
    """Crop center‑square → resize → convert RGBA."""
    w, h = img.size
    min_dim = min(w, h)
    left, top = (w - min_dim) // 2, (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim)).convert("RGBA")
    return img.resize((size, size), Image.LANCZOS)

def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> io.BytesIO:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/edit-image", methods=["POST"])
def edit_image():
    try:
        # 1. Validate input ---------------------------------------------------
        img_file = request.files.get("image")
        mask_file = request.files.get("mask")
        prompt    = request.form.get("prompt", "").strip()

        if not (img_file and mask_file and prompt):
            return (
                jsonify({"error": "Require image, mask, and prompt fields."}),
                400,
            )

        # 2. Prepare image & mask --------------------------------------------
        img_raw  = Image.open(img_file.stream)
        mask_raw = Image.open(mask_file.stream)

        img_512  = to_square_rgba(img_raw)      # 1024×1024 RGBA
        mask_512 = to_square_rgba(mask_raw.convert("L"))  # ensure grayscale / same size

        # Convert to byte buffers for upload
        img_buf  = pil_to_bytes(img_512, "PNG")
        mask_buf = pil_to_bytes(mask_512, "PNG")

        # 3. Call OpenAI image editing (DALL·E 2) -----------------------------
        print("🔧 Calling openai.images.edit …")
        rsp = openai.images.edit(
            image=img_buf,
            mask=mask_buf,
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="url",
        )

        edited_url = rsp.data[0].url
        return jsonify(
            {
                "image_url": edited_url,
                "method": "DALL·E 2 in‑painting",
                "prompt_used": prompt,
            }
        )

    except Exception as exc:
        print("💥  Edit error:", exc)
        return jsonify({"error": "Failed to edit", "details": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── LOCAL RUN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
