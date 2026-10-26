# NLP course week 07

FastAPI backend on port **8000**. Frontend on port **8001**.

Course data is in `data/`:

- `keyboards.nt`


## Run

```bash
docker compose up --build
```

- API docs: http://127.0.0.1:8000/docs
- Frontend: http://127.0.0.1:8001

```bash
curl -s http://localhost:8000/
```

## QLever

Index and start QLever from `data/` (see `docs/text-to-query.md`):

```bash
cd data
qlever index
qlever start
```

QLever listens on port **7070**.

## Tests

```bash
python -m pytest test/backend/test_main.py
```
