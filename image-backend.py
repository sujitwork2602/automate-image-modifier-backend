import os
import base64
import openai
from PIL import Image, ImageDraw
import io
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# === Flask Setup ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS FIX

# === Load Secrets ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Helpers ===

def prepare_image_for_edit(image_bytes):
    """Ensure image is 1024x1024 PNG with transparency"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Square crop
    min_dim = min(image.size)
    left = (image.width - min_dim) // 2
    top = (image.height - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))

    # Resize to 1024x1024
    image = image.resize((1024, 1024))

    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output

def generate_mask(size=(1024, 1024)):
    """Generate a simple full-white mask (edit entire image)"""
    mask = Image.new("L", size, 255)  # L mode = grayscale
    output = io.BytesIO()
    mask.save(output, format="PNG")
    output.seek(0)
    return output

# === Routes ===

@app.route("/edit-image", methods=["POST"])
def edit_image():
    try:
        image_file = request.files.get("image")
        prompt = request.form.get("prompt", "").strip()

        if not image_file or not prompt:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        prepared_image = prepare_image_for_edit(image_bytes)
        mask_image = generate_mask()  # You could later customize this

        # Use DALL·E 2 editing
        response = openai.Image.create_edit(
            image=prepared_image,
            mask=mask_image,
            prompt=prompt,
            size="1024x1024",
            n=1
        )

        edited_image_url = response["data"][0]["url"]
        return jsonify({
            "image_url": edited_image_url,
            "method": "OpenAI Edit API (DALL·E 2)",
            "prompt_used": prompt
        })

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/generate-smart", methods=["POST"])
def generate_smart():
    try:
        image_file = request.files.get("image")
        prompt = request.form.get("prompt", "").strip()

        if not image_file or not prompt:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        print("🔍 Generating refined DALL·E prompt with GPT-4o...")

        gpt_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating image prompts. Recreate the original image with just this specific change: " + prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the image precisely and modify ONLY as per the prompt."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )

        dalle_prompt = gpt_response.choices[0].message.content.strip()

        dalle_response = openai.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="hd",
            n=1
        )

        return jsonify({
            "image_url": dalle_response.data[0].url,
            "prompt_used": dalle_prompt,
            "method": "Smart GPT-4o + DALL·E 3"
        })

    except Exception as e:
        return jsonify({"error": "Smart generation failed", "details": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# === Run Locally ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
