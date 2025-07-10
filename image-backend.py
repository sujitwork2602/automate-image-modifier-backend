import os, io, base64
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import openai

# ── CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Load from .env

# ── FLASK APP ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://ai-image-modifier.web.app"]}})

# ── UTILITIES ─────────────────────────────────────────────────────────────
def to_square_rgba(img: Image.Image, size: int = 1024) -> Image.Image:
    """Crop center square, resize, and convert to RGBA"""
    w, h = img.size
    min_dim = min(w, h)
    left, top = (w - min_dim) // 2, (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim)).convert("RGBA")
    return img.resize((size, size), Image.LANCZOS)

def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> io.BytesIO:
    """Convert PIL image to byte buffer"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ── ROUTES ───────────────────────────────────────────────────────────────
@app.route("/edit-image", methods=["POST"])
def edit_image():
    try:
        # 1. Parse input
        img_file = request.files.get("image")
        mask_file = request.files.get("mask")  # optional
        prompt = request.form.get("prompt", "").strip()

        if not img_file or not prompt:
            return jsonify({"error": "Missing image or prompt"}), 400

        # 2. Load and preprocess original image
        img_raw = Image.open(img_file.stream)
        img_1024 = to_square_rgba(img_raw)
        img_buf = pil_to_bytes(img_1024, "PNG")

        # 3. Create mask
        if mask_file:
            mask_raw = Image.open(mask_file.stream).convert("L")
            mask_1024 = to_square_rgba(mask_raw)
        else:
            mask_1024 = Image.new("L", (1024, 1024), 255)  # Full white mask = edit everything

        mask_buf = pil_to_bytes(mask_1024, "PNG")

        # 4. Call OpenAI API (DALL·E 2 in-painting)
        print("🎨 Editing image with DALL·E 2...")
        rsp = openai.images.edit(
            image=img_buf,
            mask=mask_buf,
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="url"
        )

        return jsonify({
            "image_url": rsp["data"][0]["url"],
            "prompt_used": prompt,
            "method": "DALL·E 2 in-painting"
        })

    except Exception as e:
        print("💥 Error editing image:", str(e))
        return jsonify({"error": "Failed to edit", "details": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── LOCAL DEV MODE ────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
