from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_analyzer = pipeline(
    'sentiment-analysis',
    'nlptown/bert-base-multilingual-uncased-sentiment'
)
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

class TextRequest(BaseModel):
    text: str

@app.post('/sentiment-analysis/')
async def sentiment_analysis(request: TextRequest):
    try:
        result = sentiment_analyzer(request.text)[0]
        return {"stars": int(result['label'].split(' ')[0]), "score": result['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/summarization/')
async def summarization(request: TextRequest):
    try:
        inputs = tokenizer(request.text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=150, early_stopping=True)
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {'summarization': summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {'message': 'NLP (Natural Language Processing) API'}

