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

# Initialize OpenAI client (updated syntax)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Utilities ===
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# === Routes ===
@app.route("/generate", methods=["POST"])
def generate():
    try:
        print("➡️ Request received")

        # Check if API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OpenAI API key not found")
            return jsonify({"error": "OpenAI API key not configured"}), 500

        if 'image' not in request.files or 'prompt' not in request.form:
            print("❌ Missing 'image' or 'prompt'")
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files['image']
        prompt = request.form['prompt'].strip()

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Check if image file is valid
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        base64_image = encode_image_to_base64(image_bytes)

        print("🔍 Calling GPT-4o to generate description...")
        try:
            gpt_response = client.chat.completions.create(
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
        except Exception as e:
            print(f"❌ GPT-4o API error: {str(e)}")
            return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500

        description = gpt_response.choices[0].message.content.strip()
        full_prompt = f"{description}. Now {prompt}."

        print("🎨 Prompt for DALL·E:", full_prompt)

        print("🖼️ Calling DALL·E 3 to generate image...")
        try:
            dalle_response = client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
        except Exception as e:
            print(f"❌ DALL·E API error: {str(e)}")
            return jsonify({"error": "Failed to generate image", "details": str(e)}), 500

        image_url = dalle_response.data[0].url
        print("✅ Image generated:", image_url)

        return jsonify({"image_url": image_url})

    except Exception as e:
        print(f"💥 Server error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

# === Main entry ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)