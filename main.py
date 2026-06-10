import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
import torch

from google import genai

from PIL import Image

from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
from fastapi import File

from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

from config import cfg
from model import build_model

from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer

from reportlab.lib.styles import getSampleStyleSheet
from agno.agent import Agent
from agno.models.google import Gemini

from openai import OpenAI

app = FastAPI()
#gemini ai LLM 
GOOGLE_API_KEY = "AIzaSyBmTgGQgnumWQnoUsIoYetEJE2n68tDtrQ"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    markdown=True
)

from openai import OpenAI

client = OpenAI(
    api_key="sk-aecb2c7ebfe84d898e74dbe2fb429c42",
    base_url="https://api.deepseek.com"
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


templates = Jinja2Templates(
    directory="templates"
)


os.makedirs(
    "static/uploads",
    exist_ok=True
)

os.makedirs(
    "static/reports",
    exist_ok=True
)
#load models
print("Loading ResNet50 model...")

resnet_model = load_model(
    "models/final_hybrid_resnet50_model.h5"
)

print("ResNet50 loaded successfully.")

#brain validator
print("Loading Brain Validator...")

brain_validator_model = load_model(
    "models/brain_validator_mobilenetv2_final.h5"
)

print("Brain Validator loaded successfully.")

print("Loading Swin Transformer...")

swin_model, device = build_model(
    cfg.model
)

checkpoint = torch.load(
    "outputs/checkpoints/best_model.pth",
    map_location=device
)

if "model_state_dict" in checkpoint:

    swin_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif "state_dict" in checkpoint:

    swin_model.load_state_dict(
        checkpoint["state_dict"]
    )

else:

    swin_model.load_state_dict(
        checkpoint
    )

swin_model.eval()

print("Swin Transformer loaded successfully.")

def preprocess_resnet(image_path):

    img = Image.open(image_path).convert("RGB")

    img = img.resize((224, 224))

    img = np.array(img)

    img = preprocess_input(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

# VALIDATOR PREPROCESS
def preprocess_validator(image_path):

    img = Image.open(image_path).convert("RGB")

    img = img.resize((224, 224))

    img = np.array(img) / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img

def preprocess_swin(image_path):

    img = Image.open(image_path).convert("RGB")

    img = img.resize((224, 224))

    img = np.array(img).astype(np.float32) / 255.0

    img = torch.tensor(img)

    img = img.permute(2, 0, 1)

    img = img.unsqueeze(0)

    return img.to(device)
#generate PDF
def generate_pdf_report(
    filename,
    diagnosis,
    resnet_probability,
    swin_probability,
    average_probability,
    gemini_report
):

    pdf_path = f"static/reports/{filename}.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI MRI Brain Tumor Report",
            styles['Title']
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            f"<b>Diagnosis:</b> {diagnosis}",
            styles['BodyText']
        )
    )

    story.append(
        Paragraph(
            f"<b>ResNet50 Confidence:</b> {resnet_probability}%",
            styles['BodyText']
        )
    )

    story.append(
        Paragraph(
            f"<b>Swin Transformer Confidence:</b> {swin_probability}%",
            styles['BodyText']
        )
    )

    story.append(
        Paragraph(
            f"<b>Combined AI Confidence:</b> {average_probability}%",
            styles['BodyText']
        )
    )

    story.append(Spacer(1, 20))

    clean_report = gemini_report.replace(
        "<br>",
        "<br/>"
    )

    story.append(
        Paragraph(
            clean_report,
            styles['BodyText']
        )
    )

    doc.build(story)

    return pdf_path
#home
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "request": request
        }
    )
#analysis
@app.get("/analysis", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            "request": request
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):


    filename = f"{uuid.uuid4()}_{file.filename}"

    file_path = f"static/uploads/{filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )
    #brain image validator
    validator_img = preprocess_validator(file_path)

    validator_prediction = brain_validator_model.predict(
        validator_img
    )[0][0]

    validator_probability = round(
        float(validator_prediction) * 100,
        2
    )

    print(
        "Brain Validator Confidence:",
        validator_probability
    )

#detect is it tumour or not
    if validator_probability >= 50:

        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "request": request,

                "original_image": "/" + file_path,

                "gradcam_image": "/" + file_path,

                "resnet_prediction": "Rejected",

                "swin_prediction": "Rejected",

                "gemini_report": f"""

                <h2 style='color:red;'>

                Invalid Medical Image

                </h2>

                <br/>

                The uploaded image does not appear
                to be a valid brain MRI scan.

                <br/><br/>

                Brain Validator Confidence:
                <b>{validator_probability}%</b>

                <br/><br/>

                Please upload a valid
                brain MRI image.

                """,

                "pdf_report": ""
            }
        )    
    resnet_img = preprocess_resnet(file_path)

    resnet_prediction = resnet_model.predict(
        resnet_img
    )[0][0]

    resnet_probability = round(
        float(resnet_prediction) * 100,
        2
    )

    swin_img = preprocess_swin(file_path)

    with torch.no_grad():

        swin_logits = swin_model(swin_img)

        swin_prediction = torch.sigmoid(
            swin_logits
        ).item()

    swin_probability = round(
        float(swin_prediction) * 100,
        2
    )

    average_probability = round(
        (
            (resnet_probability * 0.4) +
            (swin_probability * 0.6)
        ),
        2
    )

    if (
        average_probability >= 50
        or swin_probability >= 60
    ):
        diagnosis = "Tumor Detected"

    else:

        diagnosis = "No Tumor Detected"

#gemini report

    if diagnosis == "No Tumor Detected":

        final_confidence = round(
            100 - average_probability,
            2
        )

    else:

        final_confidence = average_probability


    prompt = f"""

    You are a professional AI medical assistant.

    The final MRI analysis result is:

    Diagnosis: {diagnosis}

    Confidence Score: {final_confidence}%

    IMPORTANT RULES:

    - Focus ONLY on the final diagnosis.

    - Do NOT mention:
      AI models,
      neural networks,
      ResNet50,
      Swin Transformer,
      ensemble models,
      model disagreements,
      segmentation,
      tissue boundaries,
      healthy tissue comparison,
      or detailed radiology interpretation.

    - Explain the result in a simple,
      professional,
      medically responsible,
      and realistic way.

    - If no tumour is detected:

      * Provide a calm and reassuring explanation.

      * Explain that no strong
        tumour-related abnormalities
        were identified in the MRI scan.

      * Do NOT create fear,
        panic,
        or excessive uncertainty.

      * Recommend medical consultation
        only if symptoms persist.

    - If tumour is detected:

      * Clearly state that
        tumour-related abnormalities
        were identified in the MRI scan.

      * Explain that the findings may
        indicate the presence of a brain tumour.

      * Recommend professional medical
        evaluation and follow-up imaging.

      * Clearly explain that this is
        NOT a confirmed medical diagnosis.

      * Keep the explanation calm,
        professional,
        and medically responsible.

    - Avoid dramatic or fear-inducing language.

    - Do NOT use phrases such as:
      "critical red flag",
      "life-threatening",
      "high-risk",
      "severe abnormality",
      "missed tumour",
      "vascular malformation".

    The report should include:

    1. Analysis Summary
    2. Diagnosis Report
    3. Recommended Next Steps
    4. Disclaimer

    For the Diagnosis Report section:

    - Clearly display:

      Diagnosis: Tumor Detected

      OR

      Diagnosis: No Tumor Detected

    - Include the confidence score.

    - Explain the meaning of the result
      in a patient-friendly and medically
      responsible way.

    - If tumour detected:
      explain that tumour-related
      abnormalities were identified
      in the MRI scan.

    - If no tumour detected:
      explain that no strong
      tumour-related abnormalities
      were identified.

    Keep the report:

    - concise
    - medically responsible
    - realistic
    - easy to understand

    Use HTML formatting:
    <h2>, <b>, <br>, <ul>, <li>

    """
#here
    try:

        print("Using Gemini API...")

        response = medical_agent.run(prompt)

        gemini_report = response.content

        if (
            "RESOURCE_EXHAUSTED" in gemini_report
            or "error" in gemini_report.lower()
        ):
            raise Exception("Gemini quota exceeded")

        print("Gemini Success")

    except Exception as gemini_error:

        print("Gemini Failed:", gemini_error)

        try:

            print("Switching to DeepSeek API...")

            deepseek_response = client.chat.completions.create(

                model="deepseek-chat",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )

            gemini_report = (
                deepseek_response
                .choices[0]
                .message
                .content
            )

            print("DeepSeek Success")

        except Exception as deepseek_error:

            print("DeepSeek Failed:", deepseek_error)

            print("Using Rule-Based Fallback System...")

            if diagnosis == "Tumor Detected":

                severity = (
                    "high"
                    if average_probability >= 85
                    else "moderate"
                )

                gemini_report = f"""

                <h2 style='color:red;'>
                AI Medical Analysis Report
                </h2>

                <br/>

                <b>Final Diagnosis:</b>
                Tumor Detected

                <br/><br/>

                <b>ResNet50 Confidence:</b>
                {resnet_probability}%

                <br/>

                <b>Swin Transformer Confidence:</b>
                {swin_probability}%

                <br/>

                <b>Combined AI Confidence:</b>
                {average_probability}%

                <br/><br/>

                The uploaded MRI brain scan shows abnormal
                patterns that may indicate the presence
                of a brain tumor.

                <br/><br/>

                The AI models detected tumour-related
                features with a {severity}
                confidence level.

                <br/><br/>

                <b>Recommended Actions:</b>

                <ul>
                    <li>Consult a neurologist.</li>
                    <li>Perform additional MRI scans.</li>
                    <li>Use Grad-CAM for visual interpretation.</li>
                </ul>

                <br/>

                <b>Disclaimer:</b>
                This AI-generated analysis is for
                educational purposes only.

                """

            else:

                gemini_report = f"""

                <h2 style='color:green;'>
                AI Medical Analysis Report
                </h2>

                <br/>

                <b>Final Diagnosis:</b>
                No Tumor Detected

                <br/><br/>

                <b>ResNet50 Confidence:</b>
                {resnet_probability}%

                <br/>

                <b>Swin Transformer Confidence:</b>
                {swin_probability}%

                <br/>

                <b>Combined AI Confidence:</b>
                {average_probability}%

                <br/><br/>

                No significant tumour-related
                abnormalities were detected.

                <br/><br/>

                <b>Recommended Actions:</b>

                <ul>
                    <li>Continue medical monitoring.</li>
                    <li>Consult a doctor if symptoms persist.</li>
                </ul>

                <br/>

                <b>Disclaimer:</b>
                This AI-generated analysis is for
                educational purposes only.

                """
#pdf report
    pdf_path = generate_pdf_report(
        filename=filename,

        diagnosis=diagnosis,

        resnet_probability=resnet_probability,

        swin_probability=swin_probability,

        average_probability=average_probability,

        gemini_report=gemini_report
    )

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "request": request,

            "original_image": "/" + file_path,

            "gradcam_image": "/" + file_path,

            "resnet_prediction": resnet_probability,

            "swin_prediction": swin_probability,

            "gemini_report": gemini_report,

            "pdf_report": "/" + pdf_path
        }
    )