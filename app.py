from fastapi import FastAPI, Query

import main

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "connection online"}

# model = main.AIModel()

# # OpenAI API와 ChromaDB를 활용한 추가 엔드포인트 예시
# @app.post("/recommend/")
# def recommend_bakery(prompt: str = Query()):
#     model.request(prompt)




