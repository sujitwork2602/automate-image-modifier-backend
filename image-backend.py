import os
import base64
import openai
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# === Setup ===
app = Flask(__name__)
CORS(app, origins=["https://ai-image-modifier.web.app"])  # Replace with your frontend domain

load_dotenv()  # This loads the .env file into environment variables


openai.api_key = os.getenv("OPENAI_API_KEY")

# === Utilities ===
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# === Routes ===
@app.route("/generate", methods=["POST"])
def generate():
    try:
        print("➡️ Request received")

        if 'image' not in request.files or 'prompt' not in request.form:
            print("❌ Missing 'image' or 'prompt'")
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files['image']
        prompt = request.form['prompt'].strip()

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        image_bytes = image_file.read()
        base64_image = encode_image_to_base64(image_bytes)

        print("🔍 Calling GPT-4o to generate description...")
        gpt_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe this image and create a prompt for DALL·E 3 with this modification: {prompt}"
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
            max_tokens=500
        )

        description = gpt_response.choices[0].message.content.strip()
        full_prompt = f"{description}. Now {prompt}."

        print("🎨 Prompt for DALL·E:", full_prompt)

        print("🖼️ Calling DALL·E 3 to generate image...")
        dalle_response = openai.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = dalle_response.data[0].url
        print("✅ Image generated:", image_url)

        return jsonify({"image_url": image_url})

    except Exception as e:
        print(f"💥 Server error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

# === Main entry ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
