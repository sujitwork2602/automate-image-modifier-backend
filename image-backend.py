import os
import base64
import openai
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# === Setup ===
app = Flask(__name__)
CORS(app, origins=["https://ai-image-modifier.web.app"])

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Utilities ===
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# === Routes ===
@app.route("/generate", methods=["POST"])
def generate():
    try:
        print("➡️ Request received")

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

        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        base64_image = encode_image_to_base64(image_bytes)

        print("🔍 Using GPT-4o to analyze and describe the image...")
        try:
            # Better prompt for image analysis
            analysis_prompt = f"""
            Analyze this image in detail and create a comprehensive description that captures:
            1. The main subject(s) and their appearance
            2. The setting/background
            3. Colors, lighting, and mood
            4. Style and artistic elements
            5. Any text or objects present
            
            Then, based on this analysis, create a detailed DALL-E 3 prompt that would recreate this image but with this modification: {prompt}
            
            Make sure the prompt preserves the original image's key characteristics while incorporating the requested change.
            
            Format your response as:
            ANALYSIS: [detailed description of the original image]
            
            DALL-E PROMPT: [modified prompt for DALL-E 3]
            """
            
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
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
        except Exception as e:
            print(f"❌ GPT-4o API error: {str(e)}")
            return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500

        response_text = gpt_response.choices[0].message.content.strip()
        print("📝 GPT-4o Response:", response_text)

        # Extract the DALL-E prompt from the response
        if "DALL-E PROMPT:" in response_text:
            dalle_prompt = response_text.split("DALL-E PROMPT:")[-1].strip()
        else:
            # Fallback if format is not followed
            dalle_prompt = response_text

        print("🎨 Final DALL-E Prompt:", dalle_prompt)

        print("🖼️ Calling DALL·E 3 to generate modified image...")
        try:
            dalle_response = client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
        except Exception as e:
            print(f"❌ DALL·E API error: {str(e)}")
            error_msg = str(e)
            
            if "429" in error_msg:
                return jsonify({
                    "error": "Rate limit exceeded", 
                    "message": "You've reached the maximum number of image generations allowed. Please wait before trying again.",
                    "details": str(e)
                }), 429
            elif "insufficient_quota" in error_msg:
                return jsonify({
                    "error": "Insufficient credits", 
                    "message": "Your OpenAI account has insufficient credits. Please add credits to continue.",
                    "details": str(e)
                }), 402
            else:
                return jsonify({"error": "Failed to generate image", "details": str(e)}), 500

        image_url = dalle_response.data[0].url
        print("✅ Image generated:", image_url)

        return jsonify({
            "image_url": image_url,
            "analysis": response_text.split("DALL-E PROMPT:")[0].replace("ANALYSIS:", "").strip() if "ANALYSIS:" in response_text else "Analysis not available",
            "prompt_used": dalle_prompt
        })

    except Exception as e:
        print(f"💥 Server error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/analyze-only", methods=["POST"])
def analyze_only():
    """Endpoint to just analyze the image without generating a new one"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Missing image"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        base64_image = encode_image_to_base64(image_bytes)

        gpt_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Describe what you see including subjects, setting, colors, style, and any notable elements."
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

        analysis = gpt_response.choices[0].message.content.strip()
        return jsonify({"analysis": analysis})

    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

# === Main entry ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)