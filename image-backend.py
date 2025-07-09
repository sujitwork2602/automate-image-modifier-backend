import os
import base64
import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://ai-image-modifier.web.app"])

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def create_detailed_prompt(base64_image, modification_request):
    """Create a detailed prompt for DALL-E based on image analysis"""
    
    analysis_prompt = f"""
    You are an expert image analyst and prompt engineer. Your task is to:
    
    1. ANALYZE the uploaded image thoroughly:
       - Identify the main subject(s) and their detailed characteristics
       - Describe the setting, background, and environment
       - Note the artistic style, colors, lighting, and composition
       - Identify any objects, clothing, accessories, or text
       - Describe the mood and atmosphere
    
    2. CREATE a detailed DALL-E 3 prompt that will:
       - Recreate the original image as closely as possible
       - Incorporate this specific modification: "{modification_request}"
       - Maintain the original's style, composition, and key elements
       - Be specific about details to ensure consistency
    
    3. FORMAT your response exactly like this:
    
    ORIGINAL IMAGE ANALYSIS:
    [Provide detailed analysis of what you see]
    
    MODIFIED DALL-E PROMPT:
    [Provide the complete prompt for DALL-E 3 that recreates the image with the modification]
    
    MODIFICATION EXPLANATION:
    [Explain how the modification was incorporated while preserving the original]
    
    Remember: Be extremely detailed and specific to ensure the generated image closely matches the original while incorporating the requested change.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert image analyst and DALL-E prompt engineer. You excel at creating detailed, accurate prompts that preserve original image characteristics while incorporating specific modifications."
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
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent analysis
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise Exception(f"Failed to analyze image: {str(e)}")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        print("➡️ Request received")

        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key not configured"}), 500

        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files['image']
        modification_request = request.form['prompt'].strip()

        if not modification_request:
            return jsonify({"error": "Modification request cannot be empty"}), 400

        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        base64_image = encode_image_to_base64(image_bytes)

        print("🔍 Analyzing image and creating detailed prompt...")
        try:
            analysis_result = create_detailed_prompt(base64_image, modification_request)
            print("📝 Analysis complete")
            
            # Extract the DALL-E prompt
            if "MODIFIED DALL-E PROMPT:" in analysis_result:
                sections = analysis_result.split("MODIFIED DALL-E PROMPT:")
                if len(sections) > 1:
                    prompt_section = sections[1].split("MODIFICATION EXPLANATION:")[0].strip()
                else:
                    prompt_section = analysis_result
            else:
                prompt_section = analysis_result
            
            print("🎨 Generated prompt:", prompt_section[:200] + "...")
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500

        print("🖼️ Generating modified image with DALL-E 3...")
        try:
            dalle_response = client.images.generate(
                model="dall-e-3",
                prompt=prompt_section,
                size="1024x1024",
                quality="standard",
                n=1
            )
        except Exception as e:
            print(f"❌ DALL-E error: {str(e)}")
            error_msg = str(e)
            
            if "429" in error_msg:
                return jsonify({
                    "error": "Rate limit exceeded", 
                    "message": "Too many requests. Please wait before trying again.",
                    "details": str(e)
                }), 429
            elif "insufficient_quota" in error_msg:
                return jsonify({
                    "error": "Insufficient credits", 
                    "message": "Your OpenAI account needs more credits.",
                    "details": str(e)
                }), 402
            else:
                return jsonify({"error": "Failed to generate image", "details": str(e)}), 500

        image_url = dalle_response.data[0].url
        print("✅ Image generated successfully")

        # Parse the analysis result for the response
        analysis = ""
        explanation = ""
        
        if "ORIGINAL IMAGE ANALYSIS:" in analysis_result:
            analysis = analysis_result.split("ORIGINAL IMAGE ANALYSIS:")[1].split("MODIFIED DALL-E PROMPT:")[0].strip()
        
        if "MODIFICATION EXPLANATION:" in analysis_result:
            explanation = analysis_result.split("MODIFICATION EXPLANATION:")[1].strip()

        return jsonify({
            "image_url": image_url,
            "original_analysis": analysis,
            "modification_explanation": explanation,
            "prompt_used": prompt_section,
            "full_analysis": analysis_result
        })

    except Exception as e:
        print(f"💥 Server error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Advanced Image Modifier API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)