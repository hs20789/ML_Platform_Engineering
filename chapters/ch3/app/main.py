from fastapi import FastAPI
from pyjokes import get_joke
app = FastAPI()

@app.get("/")
async def root():
    random_joke = get_joke('en', 'neutral')
    return {"random_joke": random_joke,
            "message": "Welcome to the Hello-joker:v2!"}