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

app = FastAPI()
#gemini ai LLM 
GEMINI_API_KEY = "AIzaSyAGrwpO4BgaYdhyLVdI6PywpxaxqMdRaUA"

client = genai.Client(
    api_key=GEMINI_API_KEY
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

    # IF IMAGE IS NOT BRAIN MRI

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

#GEMINI AI REPORT
    prompt = f"""

    You are an AI medical imaging assistant.

    Analyze this MRI/X-Ray result.

    ResNet50 confidence:
    {resnet_probability}%

    Swin Transformer confidence:
    {swin_probability}%

    Combined AI confidence:
    {average_probability}%

    Final diagnosis:
    {diagnosis}

    Generate a professional medical report.

    """

    try:

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        gemini_report = response.text

    except Exception as e:

        print("Gemini Error:", e)

        gemini_report = f"""

        <b>AI Medical Analysis Report</b>

        <br/><br/>

        Gemini AI quota exceeded.
        Using local AI medical interpretation instead.

        <br/><br/>

        <b>ResNet50 Confidence:</b>
        {resnet_probability}%

        <br/><br/>

        <b>Swin Transformer Confidence:</b>
        {swin_probability}%

        <br/><br/>

        <b>Combined AI Confidence:</b>
        {average_probability}%

        <br/><br/>

        <b>Final Diagnosis:</b>
        {diagnosis}

        <br/><br/>

        The uploaded MRI/X-Ray image was analyzed
        using dual AI models including ResNet50 CNN
        and Swin Transformer architecture.

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