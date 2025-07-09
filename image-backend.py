import os
import base64
import openai
from PIL import Image
import io
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://ai-image-modifier.web.app"])

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prepare_image_for_edit(image_bytes):
    """Prepare image for OpenAI edit API - must be square PNG with transparency"""
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Make it square by cropping or padding
        width, height = image.size
        size = min(width, height)
        
        # Crop to square from center
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        square_image = image.crop((left, top, right, bottom))
        
        # Resize to 1024x1024 (required by OpenAI)
        square_image = square_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        square_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
        
    except Exception as e:
        raise Exception(f"Image preparation failed: {str(e)}")

def create_edit_mask(image_bytes, mask_prompt):
    """Create a mask for the areas to edit using GPT-4o analysis"""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        mask_analysis = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying specific areas in images that need to be edited. Provide precise instructions for creating an edit mask."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image and identify the specific areas that should be modified for this request: '{mask_prompt}'. Describe exactly which parts of the image need to be changed."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return mask_analysis.choices[0].message.content.strip()
        
    except Exception as e:
        raise Exception(f"Mask analysis failed: {str(e)}")

@app.route("/edit-image", methods=["POST"])
def edit_image():
    """Use OpenAI's image edit API for more accurate modifications"""
    try:
        print("➡️ Edit request received")

        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key not configured"}), 500

        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files['image']
        edit_prompt = request.form['prompt'].strip()

        if not edit_prompt:
            return jsonify({"error": "Edit prompt cannot be empty"}), 400

        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        print("🔧 Preparing image for editing...")
        try:
            prepared_image = prepare_image_for_edit(image_bytes)
        except Exception as e:
            print(f"❌ Image preparation error: {str(e)}")
            return jsonify({"error": "Failed to prepare image", "details": str(e)}), 500

        print("✏️ Generating edited image...")
        try:
            # Use OpenAI's image edit API
            response = client.images.edit(
                image=prepared_image,
                prompt=edit_prompt,
                n=1,
                size="1024x1024"
            )
            
            edited_image_url = response.data[0].url
            print("✅ Image edited successfully")
            
            return jsonify({
                "image_url": edited_image_url,
                "edit_prompt": edit_prompt,
                "method": "OpenAI Edit API"
            })
            
        except Exception as e:
            print(f"❌ Edit API error: {str(e)}")
            error_msg = str(e)
            
            if "429" in error_msg:
                return jsonify({
                    "error": "Rate limit exceeded", 
                    "message": "Too many edit requests. Please wait before trying again.",
                    "details": str(e)
                }), 429
            else:
                return jsonify({"error": "Failed to edit image", "details": str(e)}), 500

    except Exception as e:
        print(f"💥 Server error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/generate-variation", methods=["POST"])
def generate_variation():
    """Use OpenAI's image variation API"""
    try:
        print("➡️ Variation request received")

        if 'image' not in request.files:
            return jsonify({"error": "Missing image"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        
        prepared_image = prepare_image_for_edit(image_bytes)
        
        print("🔄 Generating variation...")
        response = client.images.create_variation(
            image=prepared_image,
            n=1,
            size="1024x1024"
        )
        
        variation_url = response.data[0].url
        print("✅ Variation generated successfully")
        
        return jsonify({
            "image_url": variation_url,
            "method": "OpenAI Variation API"
        })
        
    except Exception as e:
        print(f"❌ Variation error: {str(e)}")
        return jsonify({"error": "Failed to generate variation", "details": str(e)}), 500

@app.route("/generate-smart", methods=["POST"])
def generate_smart():
    """Improved DALL-E generation with better context preservation"""
    try:
        print("➡️ Smart generation request received")

        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files['image']
        modification_request = request.form['prompt'].strip()
        
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Get a very detailed analysis focusing on preservation
        print("🔍 Analyzing image for context preservation...")
        analysis_prompt = f"""
        CRITICAL: Your job is to create a DALL-E prompt that will produce an image as identical as possible to the uploaded image, with ONLY the specific change requested.

        1. Analyze every detail of this image:
           - Exact clothing, colors, textures, patterns
           - Precise facial features, hair, expressions
           - Exact background elements, lighting, shadows
           - Camera angle, composition, framing
           - Artistic style, photo quality, filters

        2. Create a DALL-E prompt that:
           - Recreates EVERY visual element exactly
           - Only incorporates this change: "{modification_request}"
           - Uses specific descriptive terms, not generic ones
           - Maintains the exact same style and quality

        3. Be extremely specific about details that should NOT change.

        Respond with only the DALL-E prompt, nothing else.
        """

        try:
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating DALL-E prompts that preserve original image details while making minimal specific changes. Focus on exact replication with precise modifications."
                    },
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
                max_tokens=800,
                temperature=0.1  # Very low temperature for consistency
            )
        except Exception as e:
            return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500

        dalle_prompt = gpt_response.choices[0].message.content.strip()
        print("🎨 Generated preservation-focused prompt")

        print("🖼️ Generating with DALL-E 3...")
        try:
            dalle_response = client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                size="1024x1024",
                quality="hd",  # Use HD quality for better detail preservation
                n=1
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                return jsonify({
                    "error": "Rate limit exceeded", 
                    "message": "Too many requests. Please wait before trying again."
                }), 429
            else:
                return jsonify({"error": "Failed to generate image", "details": str(e)}), 500

        image_url = dalle_response.data[0].url
        print("✅ Smart generation complete")

        return jsonify({
            "image_url": image_url,
            "prompt_used": dalle_prompt,
            "method": "Smart DALL-E Generation"
        })

    except Exception as e:
        print(f"💥 Server error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Image Editor API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)