from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QueryInput(BaseModel):
    text: str


@app.get("/")
async def root():
    """Check that the app is running."""
    return {"message": "app is running"}


@app.post("/v1/query")
def query_endpoint(q: QueryInput):
    """Accept a natural-language question. The response is left blank."""
    return {
        "query": q.text,
        "items_as_strings": [],
        "properties_as_strings": [],
        "sparql": "",
        "results": [],
    }
