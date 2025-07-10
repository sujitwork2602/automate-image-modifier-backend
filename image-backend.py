import os
import base64
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# === Config ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://ai-image-modifier.web.app"]}})  # CORS FIX


def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

@app.route("/edit-image-smart", methods=["POST"])
def edit_image_smart():
    try:
        # Validate input
        image_file = request.files.get("image")
        edit_prompt = request.form.get("prompt", "").strip()

        if not image_file or not edit_prompt:
            return jsonify({"error": "Image or prompt is missing"}), 400

        image_bytes = image_file.read()
        base64_image = encode_image_to_base64(image_bytes)

        # GPT-4o prompt (Hardcoded expert instruction)
        print("🧠 GPT-4o analyzing and preparing edit...")

        system_message = (
            "You are a visual editing expert. Your goal is to describe the image in a way "
            "that replicates it 1:1 in DALL·E 3 while making the exact requested change. "
            "Do not alter any other element, composition, lighting, or visual characteristics."
        )

        user_prompt = (
            f"Study the image carefully. Then write a DALL·E 3 prompt that fully replicates the image's details, "
            f"but with this specific change: {edit_prompt}"
        )

        gpt_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": system_message },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": user_prompt },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.3
        )

        dalle_prompt = gpt_response.choices[0].message.content.strip()

        # DALL·E 3 image generation
        print("🎨 Generating image with DALL·E 3...")
        dalle_response = openai.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="hd",
            n=1
        )

        image_url = dalle_response.data[0].url

        return jsonify({
            "image_url": image_url,
            "prompt_used": dalle_prompt,
            "method": "GPT-4o + DALL·E 3"
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "running", "message": "Smart image edit API is live"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
