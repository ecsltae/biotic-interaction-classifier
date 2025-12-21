from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import LukeTokenizer, LukeForSequenceClassification
import torch

app = FastAPI(title="Binary Classifier API (LUKE version)", description="Classifies passages as positive or negative.", docs_url="/")

# Load LUKE tokenizer and model
model_path = "luke_classifier"
tokenizer = LukeTokenizer.from_pretrained(model_path)
model = LukeForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class TextClassificationRequest(BaseModel):
    text: str

class TextClassificationResponse(BaseModel):
    prediction: int
    negative_probability: float
    positive_probability: float

@app.post("/classify/", response_model=TextClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    """
    Classifies the input text as positive or negative using a trained LUKE model.
    """
    try:
        inputs = tokenizer([request.text], truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = int(torch.argmax(logits, dim=-1).cpu().item())

        return TextClassificationResponse(
            prediction=prediction,
            negative_probability=float(probabilities[0]),
            positive_probability=float(probabilities[1])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
