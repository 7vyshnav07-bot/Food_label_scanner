
import os
import base64
import json
from health_score import calculate_health_score
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Azure AI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", ""
).rstrip("/")

AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY", ""
)

TEXT_MODEL = os.getenv(
    "TEXT_MODEL_DEPLOYMENT", "gpt-4.1-mini"
)

VISION_MODEL = os.getenv(
    "VISION_MODEL_DEPLOYMENT", "gpt-4.1-mini"
)


# Azure Client
def get_client():

    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "Azure AI is not configured. Check your environment variables."
        )

    endpoint = AZURE_OPENAI_ENDPOINT

    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint[:-len("/openai/v1")]

    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{endpoint.rstrip('/')}/openai/v1/"
    )


# Main Nutrition and Allergen Analysis Prompt
NUTRITION_PROMPT = """
You are NutriLens AI, an intelligent food label analysis assistant.

Analyze the provided food label text or image carefully.

GENERAL RULES:

1. Extract only information that is visible or explicitly stated.
2. Never invent nutrition values, ingredients, serving sizes, or allergens.
3. If a value is missing, return null for numeric values and
   "Not available" for unavailable text.
4. Distinguish between per-serving and per-100g values.
5. Never assume a product is safe for someone with an allergy.
6. Do not provide medical diagnoses.
7. Do not claim that a product is universally healthy or unhealthy.

PRODUCT IDENTIFICATION:

1. Inspect the entire food package.
2. Identify the product name from visible branding or label information.
3. Identify its food category.
4. Do not use "Not available" if the product name can reasonably be read.
5. If only the nutrition panel is visible and the product cannot be
   identified, use "Food Product" as the product name and
   "Packaged Food" as the category.
6. Never invent a brand or product name.

ALLERGEN DETECTION RULES:


ALLERGEN DETECTION RULES:

Your task is to identify allergens from the actual visible food label,
not to generate a general list of common allergens.

Follow these rules strictly:

1. First inspect the ingredients list and allergen declaration.
2. Identify the exact words that provide evidence for each allergen.
3. Report an allergen only when its presence is supported by readable
   label text.
4. Never assume peanuts, milk, soy, wheat, or any other allergen
   is present simply because it is common in packaged foods.
5. Never use a previous product's analysis when analyzing a new image.
6. Do not infer allergens from the product category or product name.
7. Do not add an allergen merely because it is included in the
   examples below.

Examples of evidence:

- "Peanuts" in ingredients → report peanuts.
- "Contains: Peanuts" → report peanuts as a declared allergen.
- "May contain peanuts" → report peanuts as a precautionary statement.
- "Contains milk" → report milk.
- No readable peanut reference → do not report peanuts.

If an allergen is not mentioned in the readable label, do not include it
in the allergen findings.

If the ingredients or allergen declaration are unreadable or missing,
use insufficient_information when the available image does not allow
a reliable assessment.

If no allergen evidence is found in a readable label, use
not_identified.

Do not interpret not_identified as allergen-free or allergy-safe.

For every finding, return:
- name
- evidence
- source_type

Allowed source_type values:
- ingredient
- contains_statement
- may_contain_statement
- other_label_text

Return allergen_assessment with:
- status: findings, not_identified, or insufficient_information
- note: a short explanation based on the available label evidence

IMPORTANT DISTINCTIONS:

1. An allergen appearing in the ingredients list must be identified
   as an ingredient-based finding.

2. Allergens listed under "Contains" must be identified as declared
   allergens.

3. Statements such as "May contain peanuts" or
   "Manufactured in a facility that processes milk" must be
   identified as precautionary cross-contact statements.

4. Do not treat a precautionary statement as proof that the allergen
   is an intentional ingredient.

5. Do not infer allergens simply from the product category or food name.

6. If no allergen evidence is found in a readable label, use
   "not_identified" as the assessment status.

7. If the label is blurry, incomplete, or the ingredients or allergen
   declaration cannot be read reliably, use
   "insufficient_information".

8. Never interpret "not_identified" as "allergen-free" or
   "safe for allergy sufferers".

Return allergen_assessment with:

- status: "findings", "not_identified", or "insufficient_information"
- note: a short explanation of the assessment and its limitations

Return ONLY valid JSON using this structure:

{
  "product_name": "string or Not available",
  "category": "string or Not available",
  "serving_size": "string or Not available",

  "nutrition": {
    "calories": {
      "value": null,
      "unit": "kcal",
      "basis": "per serving or Not available"
    },
    "total_fat": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "saturated_fat": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "trans_fat": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "cholesterol": {
      "value": null,
      "unit": "mg",
      "basis": "per serving or Not available"
    },
    "sodium": {
      "value": null,
      "unit": "mg",
      "basis": "per serving or Not available"
    },
    "carbohydrates": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "fiber": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "total_sugars": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "added_sugars": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    },
    "protein": {
      "value": null,
      "unit": "g",
      "basis": "per serving or Not available"
    }
  },

  "ingredients": [],

  "allergens_declared_or_identified": [
    {
      "name": "Allergen name",
      "evidence": "Exact supporting label text",
      "source_type": "ingredient"
    }
  ],

  "allergen_assessment": {
    "status": "findings",
    "note": "Explanation based on readable label evidence"
  },

  "additives": [
    {
      "name": "ingredient or additive",
      "common_function": "general function"
    }
  ],

  "label_notes": [],

  "summary": "Brief factual explanation",

  "nutrition_observations": [
    "Factual observation based on extracted information"
  ]
}

For blurry images, identify unreadable information.

For non-food images, explain that the image is not a food label.

Do not provide medical diagnoses or claim that a product is
universally healthy or unhealthy.
"""


# AI Model Request
def call_model(model, content):

    response = get_client().responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    raw = (response.output_text or "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    return raw, parsed


# Standard Analysis Response

def analysis_response(raw, parsed):
    health_score = None

    if isinstance(parsed, dict):
        health_score = calculate_health_score(parsed)
        parsed["health_score"] = health_score

    return jsonify(
        success=True,
        result=raw,
        analysis=parsed,
        health_score=health_score
    )

# Home Page
@app.get("/")
def home():

    return render_template("home.html")
@app.route("/history")
def history():
    return render_template("history.html")


# Text Analysis Page
@app.get("/text-analysis")
def text_analysis():

    return render_template("text_analysis.html")


# Text-Based Food Analysis
@app.post("/api/text-analysis")
def api_text_analysis():

    data = request.get_json(silent=True) or {}

    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify(
            success=False,
            error="Please enter food label text."
        ), 400

    if len(text) > 20000:
        return jsonify(
            success=False,
            error="Please keep label text under 20,000 characters."
        ), 400

    try:

        raw, parsed = call_model(
            TEXT_MODEL,
            [
                {
                    "type": "input_text",
                    "text": f"{NUTRITION_PROMPT}\n\nLABEL TEXT:\n{text}"
                }
            ]
        )

        return analysis_response(raw, parsed)

    except Exception:
        app.logger.exception("Text analysis failed")

        return jsonify(
            success=False,
            error="Text analysis failed. Check the Azure configuration and logs."
        ), 500


# Image-Based Food Analysis
@app.post("/api/vision")
def api_vision():

    image = request.files.get("image")

    if not image or not image.filename:
        return jsonify(
            success=False,
            error="Please upload a food label image."
        ), 400

    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.mimetype not in allowed:
        return jsonify(
            success=False,
            error="Upload a JPG, PNG, or WEBP image."
        ), 400

    image_bytes = image.read()

    if not image_bytes:
        return jsonify(
            success=False,
            error="The uploaded image is empty."
        ), 400

    if len(image_bytes) > 7 * 1024 * 1024:
        return jsonify(
            success=False,
            error="Image must be smaller than 7 MB."
        ), 400

    encoded = base64.b64encode(image_bytes).decode("ascii")

    try:

        raw, parsed = call_model(
            VISION_MODEL,
            [
                {
                    "type": "input_text",
                    "text": NUTRITION_PROMPT
                },
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{image.mimetype};base64,{encoded}"
                    )
                }
            ]
        )

        return analysis_response(raw, parsed)

    except Exception:
        app.logger.exception("Image analysis failed")

        return jsonify(
            success=False,
            error="Image analysis failed. Check the Azure configuration and logs."
        ), 500


# AI Nutrition Chatbot
@app.post("/api/chat")
def api_chat():

    data = request.get_json(silent=True) or {}

    question = str(data.get("question", "")).strip()

    food_analysis = str(
        data.get("food_analysis", "")
    ).strip()

    if not question:
        return jsonify(
            success=False,
            error="Please enter a question."
        ), 400

    if not food_analysis:
        return jsonify(
            success=False,
            error="Please analyze a food label first."
        ), 400

    if len(question) > 2000 or len(food_analysis) > 20000:
        return jsonify(
            success=False,
            error="The question or analysis is too long."
        ), 400

    prompt = f"""
You are NutriLens AI, a food-label explainer.

Answer the user's question using the provided food analysis.

Explain nutrition and allergen information in simple language.

Do not invent nutrition values, ingredients, or allergens.
Do not provide medical diagnoses or claim medical safety.

For allergy-related questions:
- Explain whether the analysis found label evidence.
- Distinguish ingredients from precautionary cross-contact statements.
- Never claim that a product is allergy-safe based only on this analysis.
- If information is missing, say so clearly.

FOOD ANALYSIS:
{food_analysis}

USER QUESTION:
{question}
"""

    try:

        response = get_client().responses.create(
            model=TEXT_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return jsonify(
            success=True,
            result=response.output_text
        )

    except Exception:
        app.logger.exception("Chat request failed")

        return jsonify(
            success=False,
            error="Chat request failed. Please try again."
        ), 500


# Health Check
@app.get("/health")
def health():

    return jsonify(
        status="ok",
        azure_configured=bool(
            AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
        ),
        text_model=TEXT_MODEL,
        vision_model=VISION_MODEL
    )


# Upload Error Handler
@app.errorhandler(413)
def too_large(_error):

    return jsonify(
        success=False,
        error="Upload is too large. Maximum request size is 8 MB."
    ), 413


# Application Startup
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )