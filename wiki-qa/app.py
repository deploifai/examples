import typing

from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict

app = FastAPI()


@app.get("/")
async def root():
    return "OK"


class Request(BaseModel):
    question: str


class Result(BaseModel):
    score: float
    title: str
    text: str


class Response(BaseModel):
    results: typing.List[Result]


@app.post("/predict", response_model=Response)
async def p(request: Request):
    results = predict(request.question)
    return Response(
        results=[
            Result(score=r["score"], title=r["title"], text=r["text"]) for r in results
        ]
    )
