import os
import io
import base64
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import openai

# ── CONFIG ─────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://ai-image-modifier.web.app"]}})

# ── UTILS ──────────────────────────────
def prepare_image(image_bytes):
    """Convert image to 1024x1024 RGBA square PNG"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    min_dim = min(img.width, img.height)
    left = (img.width - min_dim) // 2
    top = (img.height - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((1024, 1024), Image.LANCZOS)

    output = io.BytesIO()
    img.save(output, format="PNG")
    output.seek(0)
    return output

def generate_white_mask(size=(1024, 1024)):
    """Return a full-white mask to edit entire image"""
    mask = Image.new("L", size, 255)  # Grayscale white mask
    output = io.BytesIO()
    mask.save(output, format="PNG")
    output.seek(0)
    return output

# ── ROUTES ─────────────────────────────
@app.route("/edit-image", methods=["POST"])
def edit_image():
    try:
        img_file = request.files.get("image")
        prompt = request.form.get("prompt", "").strip()

        if not img_file or not prompt:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_bytes = img_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        print("📥 Received image:", img_file.filename)
        print("📝 Prompt:", prompt)

        img = prepare_image(image_bytes)
        mask = generate_white_mask()

        print("🛠️ Sending to OpenAI...")
        response = openai.Image.create_edit(
            image=img,
            mask=mask,
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="url"
        )

        return jsonify({
            "image_url": response["data"][0]["url"],
            "method": "DALL·E 2 Edit",
            "prompt_used": prompt
        })

    except Exception as e:
        print("💥 Error during editing:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Image editor backend is running."})

# ── RUN LOCALLY ───────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
