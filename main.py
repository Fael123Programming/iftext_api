from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_analyzer = pipeline(
    'sentiment-analysis',
    'nlptown/bert-base-multilingual-uncased-sentiment'
)


class TextRequest(BaseModel):
    text: str


@app.post('/sentiment-analysis/')
async def sentiment_analysis(request: TextRequest):
    try:
        result = sentiment_analyzer(request.text)[0]
        return {"stars": int(result['label'].split(' ')[0]), "score": result['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
async def root():
    return {'message': 'Sentiment Analysis API'}

