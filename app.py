import os
import base64

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI


# Load .env
load_dotenv()


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# AZURE AI FOUNDRY CONFIGURATION
# --------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    ""
).rstrip("/")

AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    ""
)

TEXT_MODEL = os.getenv(
    "TEXT_MODEL_DEPLOYMENT",
    "gpt-4.1-mini"
)

VISION_MODEL = os.getenv(
    "VISION_MODEL_DEPLOYMENT",
    "gpt-4.1-mini"
)


# --------------------------------------------------
# AZURE OPENAI CLIENT
# --------------------------------------------------

def get_client():

    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is missing in .env"
        )

    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY is missing in .env"
        )

    endpoint = AZURE_OPENAI_ENDPOINT

    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint[:-len("/openai/v1")]

    endpoint = endpoint.rstrip("/")

    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{endpoint}/openai/v1/"
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/")
def home():

    return render_template("home.html")


# --------------------------------------------------
# TEXT ANALYZER PAGE
# --------------------------------------------------

@app.get("/text-analysis")
def text_analysis():

    return render_template(
        "text_analysis.html"
    )


# --------------------------------------------------
# TEXT ANALYZER API
# --------------------------------------------------

@app.post("/api/text-analysis")
def api_text_analysis():

    try:

        data = request.get_json()

        if not data:

            return jsonify(
                error="No data was provided."
            ), 400


        text = data.get("text", "").strip()

        if not text:

            return jsonify(
                error="Please enter food label text."
            ), 400


        prompt = f"""
You are an AI Food Label Text Analyzer.

Analyze the food label text provided by the user.

Your job is to identify information that is actually
present in the provided text.

IMPORTANT RULES:

- Do not invent information.
- Do not guess missing values.
- If information is not available, write "Not available".
- Only use information from the provided text.
- Keep the explanation simple.
- Identify allergens only when they are explicitly
  mentioned or clearly identifiable from the provided
  ingredient text.

Analyze the following:

1. Product Name
2. Serving Size
3. Calories
4. Total Fat
5. Saturated Fat
6. Trans Fat
7. Cholesterol
8. Sodium
9. Total Carbohydrates
10. Dietary Fiber
11. Total Sugars
12. Added Sugars
13. Protein
14. Ingredients
15. Allergens

Then provide a simple nutritional analysis.

Use this format:

PRODUCT INFORMATION

Product Name:
Serving Size:


NUTRITION INFORMATION

Calories:
Total Fat:
Saturated Fat:
Trans Fat:
Cholesterol:
Sodium:
Total Carbohydrates:
Dietary Fiber:
Total Sugars:
Added Sugars:
Protein:


INGREDIENTS

-


ALLERGENS

-


NUTRITION ANALYSIS

-


FOOD LABEL TEXT:

{text}
"""


        client = get_client()


        response = client.responses.create(

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


        result = response.output_text


        return jsonify(

            success=True,

            result=result

        )


    except Exception as e:

        print(
            "TEXT ANALYZER ERROR:",
            str(e)
        )

        return jsonify(

            success=False,

            error=str(e)

        ), 500


# --------------------------------------------------
# IMAGE ANALYZER API
# --------------------------------------------------

@app.post("/api/vision")
def api_vision():

    try:

        image = request.files.get("image")


        if not image:

            return jsonify(
                error="Please upload a food label image."
            ), 400


        allowed_types = [

            "image/jpeg",
            "image/png",
            "image/webp"

        ]


        if image.mimetype not in allowed_types:

            return jsonify(

                error=(
                    "Please upload a JPG, PNG, "
                    "or WEBP image."
                )

            ), 400


        image_bytes = image.read()


        if not image_bytes:

            return jsonify(

                error="The uploaded image is empty."

            ), 400


        image_data = base64.b64encode(
            image_bytes
        ).decode("utf-8")


        mime_type = image.mimetype


        prompt = """
You are an AI Food Label Image Analyzer.

Carefully analyze the uploaded food product label.

Extract ONLY information that is visible and readable
in the image.

Extract:

1. Product Name
2. Serving Size
3. Calories
4. Total Fat
5. Saturated Fat
6. Trans Fat
7. Cholesterol
8. Sodium
9. Total Carbohydrates
10. Dietary Fiber
11. Total Sugars
12. Added Sugars
13. Protein
14. Ingredients
15. Allergens

IMPORTANT RULES:

- Never guess nutritional values.
- Never invent information.
- Only use information visible in the image.
- If something cannot be read, write "Not available".
- Clearly identify the serving size.
- Do not confuse per-serving values with per-100g values.
- Mention allergens only when clearly visible or
  explicitly stated on the label.
- If the image is too blurry or unclear, say:

"The food label is unclear. Please upload a clearer image."

After extracting the information, provide a simple
nutritional analysis.

Use this format:

PRODUCT INFORMATION

Product Name:
Serving Size:


NUTRITION INFORMATION

Calories:
Total Fat:
Saturated Fat:
Trans Fat:
Cholesterol:
Sodium:
Total Carbohydrates:
Dietary Fiber:
Total Sugars:
Added Sugars:
Protein:


INGREDIENTS

-


ALLERGENS

-


NUTRITION ANALYSIS

-


Do not add information that is not present on the label.
"""


        client = get_client()


        response = client.responses.create(

            model=VISION_MODEL,

            input=[

                {

                    "role": "user",

                    "content": [

                        {

                            "type": "input_text",

                            "text": prompt

                        },

                        {

                            "type": "input_image",

                            "image_url":
                                f"data:{mime_type};base64,{image_data}"

                        }

                    ]

                }

            ]

        )


        result = response.output_text


        return jsonify(

            success=True,

            result=result

        )


    except Exception as e:

        print(
            "IMAGE ANALYZER ERROR:",
            str(e)
        )

        return jsonify(

            success=False,

            error=str(e)

        ), 500
# --------------------------------------------------
# FOOD CHATBOT API
# --------------------------------------------------

@app.post("/api/chat")
def api_chat():

    try:

        data = request.get_json()

        if not data:
            return jsonify(
                error="No data was provided."
            ), 400

        question = data.get("question", "").strip()

        food_analysis = data.get(
            "food_analysis",
            ""
        ).strip()

        if not question:
            return jsonify(
                error="Please enter a question."
            ), 400

        if not food_analysis:
            return jsonify(
                error="Please analyze a food label first."
            ), 400

        prompt = f"""
You are a Food Label Assistant.

You answer questions ONLY about the food product
described in the food analysis below.

IMPORTANT RULES:

1. Use only the information provided in the
   food analysis.
2. Do not invent nutritional values.
3. Do not guess missing ingredients or allergens.
4. If the requested information is not available,
   clearly say that it is not available from the
   analyzed food label.
5. Keep answers simple and easy to understand.
6. You may explain the nutritional information that
   is already present.
7. Do not provide a medical diagnosis.
8. Do not claim that a food is medically safe or
   unsafe for a person.
9. If the question is unrelated to the analyzed food,
   politely say that you can only answer questions
   about the analyzed food.

ANALYZED FOOD:

{food_analysis}


USER QUESTION:

{question}


Answer the user's question using the analyzed food
information above.
"""

        client = get_client()

        response = client.responses.create(

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

        result = response.output_text

        return jsonify(
            success=True,
            result=result
        )

    except Exception as e:

        print(
            "CHATBOT ERROR:",
            str(e)
        )

        return jsonify(
            success=False,
            error=str(e)
        ), 500

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return jsonify(

        status="running",

        text_model=TEXT_MODEL,

        vision_model=VISION_MODEL,

        endpoint_configured=bool(
            AZURE_OPENAI_ENDPOINT
        ),

        api_key_configured=bool(
            AZURE_OPENAI_API_KEY
        )

    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )