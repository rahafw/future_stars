from fastapi import FastAPI

test = FastAPI(title="API Testing")

@test.get("/")
def greating():
    return {"message": "Hello World!"}
