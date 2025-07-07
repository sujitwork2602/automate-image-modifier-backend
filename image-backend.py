# backend/app.py

from flask import Flask, request, jsonify
import openai
import os
import base64

app = Flask(__name__)

# Set your OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files or "prompt" not in request.form:
        return jsonify({"error": "Image and prompt required"}), 400

    image_data = request.files["image"].read()
    prompt = request.form["prompt"]
    base64_image = base64.b64encode(image_data).decode("utf-8")

    try:
        # Step 1: Analyze the image + prompt
        chat_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe and modify this image: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )

        description = chat_response.choices[0].message.content

        # Step 2: Generate image from description
        image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"{description}. Modify it to: {prompt}",
            size="1024x1024",
            quality="standard",
            n=1
        )

        return jsonify({"image_url": image_response.data[0].url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)