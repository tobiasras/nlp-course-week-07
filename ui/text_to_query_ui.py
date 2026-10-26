r"""Keyboards text-to-SPARQL frontend.

Original frontend brief (verbatim):

You should make a Python-based frontend Web app perhaps with
Javascript that can use another Web service (that is built
independently by students). The Web app should make it easy to
demonstrate and test the Web service. The Web service is
text-to-sparql. This exercise is describe in detail
below.     

I want a frontend Web app with few dependencies so that installation
would be painless. 

If there is any Javascript is should be relatively simple. Do not
include jQuery or other external library unless absolutely necessary.

The Web app may include some styling, but I would like to have it
simple and the style within HTML code rather than as a separate style
file. It is running at DTU where the primary colors are corporate red
(153,0,0), white and black. Some more colors at
https://designguide.dtu.dk/colours if needed. 

The interface language of the Web app should be English. 

The Web app could be implemented in FastAPI and has the least
dependencies (Pydantic is fine). I as a teacher and the students hould 
be able to understand the code even though the course is not about
frontend development. If docstrings are included make it in numpydoc
format and do not be afraid to add doctests if that is relevant.

The SPARQL endpoint contains data from
https://keyboards.wikibase.cloud - a small Wikibase with data on
musical keyboards. The SPARQL endpoint is at:
https://keyboards.wikibase.cloud/query/sparql the MediaWiki API is at
https://keyboards.wikibase.cloud/w/api.php 


The data distributed has the highest QID as
<https://keyboards.wikibase.cloud/entity/Q87>. Note the QID and PID of
the Wikibase is NOT the same as the once on Wikidata. The data has since been
extended. For test the student application there are query in the
knowledge graph with English and Danish questions and SPARQL. The test
questions should be >Q86 e.g., 

For instance https://keyboards.wikibase.cloud/wiki/Item:Q96

Currently there is only one test question - but I will extend this. I
would like to have a list with an indication with columns such as
ground truth SPARQL, student result data, ground truth result data,
indication of result match (say set). Note that the result can be a
single item, a literal value, or list of such values and the the name
of the projection variable might differ between ground truth and
student respons! 

Items have P11 (instance of) Q43 (query)
and P27 (question) literal string value with natural question
and P31 (sparql) literal string value with ground truth SPARQL

The result from executing the test questions should be regarded as
ground through and compared to what the student returns with his/her
application.


To present the result in the frontend:

Labels for URIs can be prefetched with

PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item ?itemLabel {
  ?item rdfs:label ?itemLabel
  FILTER (LANG(?itemLabel) = "en") .
}

Images for keyboards has the PID P18 and values refer to a Commons
file could optionally be used.

The local endpoint is a QLever engine on http://localhost:7070/sparql
with the data.

As dataset-specific prefixes use, e.g.,

PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
CAMPUSAI_MODEL = os.getenv("CAMPUSAI_MODEL")
CAMPUSAI_EMBED_MODEL = os.getenv("CAMPUSAI_EMBED_MODEL")

client = OpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url="https://chat.campusai.compute.dtu.dk/api/v1"
)

The LLM model is "Gemma 4".

Please also make any error message pedagogic, and include appropriate
time out for the response from the Web service. Include operational
metrics, e.g., response latency from the Web service and/or number of
successful request. The Web service ought to handle multiple
asynchronous Web requests, so this could be  

If you use Python or Javascript templates/formatting remember the level
of interpolation and escaping.

Include this prompt (the above text) as part of the generated code,
e.g., in a docstring.
"""

import asyncio
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
from itertools import permutations
import os
import re
from time import perf_counter

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import AnyHttpUrl, BaseModel, Field

PUBLIC_ENDPOINT = "https://keyboards.wikibase.cloud/query/sparql"
PREFIXES = """PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>
"""
DEFAULT_PREFIXES = {
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}
LABEL_QUERY = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item ?itemLabel WHERE {
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en")
}"""

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
MAX_COLUMNS = 8  # At most 8! alignments; ample for the teaching dataset.
MAX_ROWS = 10000


def cell_value(value):
    """Return a hashable, value-based representation of one result cell.

    Parameters
    ----------
    value : object
        JSON scalar, null, or a SPARQL JSON binding.

    Returns
    -------
    tuple
        Value tagged with its comparison category.

    Examples
    --------
    >>> cell_value('1385.0') == cell_value(1385)
    True
    >>> cell_value(True) == cell_value(1)
    False
    """
    if isinstance(value, dict):
        kind = value.get("type")
        if kind not in {"uri", "literal", "typed-literal", "bnode"} or "value" not in value:
            raise ValueError("A result cell must be a JSON scalar or a SPARQL JSON binding.")
        if kind == "bnode":
            raise ValueError("Blank-node identities cannot be compared across requests.")
        if kind == "uri":
            return ("uri", str(value["value"]))
        if value.get("datatype") == "http://www.w3.org/2001/XMLSchema#boolean":
            lexical = value["value"]
            if lexical not in {"true", "false", "1", "0"}:
                raise ValueError("Invalid xsd:boolean lexical value.")
            return ("boolean", lexical in {"true", "1"})
        value = value["value"]
    if value is None:
        return ("unbound", "")
    if isinstance(value, bool):
        return ("boolean", value)
    if not isinstance(value, (str, int, float)):
        raise ValueError("Nested lists and objects are not supported inside result cells.")
    text = str(value)
    if NUMBER.fullmatch(text):
        try:
            return ("number", Decimal(text))
        except InvalidOperation:
            pass
    if isinstance(value, (int, float)):
        raise ValueError("Numeric results must be finite JSON numbers.")
    if text.startswith(("https://", "http://")):
        return ("uri", text)
    return ("text", text)


def result_table(data):
    """Normalize SELECT/ASK JSON or simple rows to column names and tuples.

    Parameters
    ----------
    data : object
        SPARQL Results JSON, a row list, a scalar list, or an ASK boolean.

    Returns
    -------
    tuple
        Column names and a set of normalized row tuples.
    """
    if isinstance(data, dict):
        if "boolean" in data:
            if not isinstance(data["boolean"], bool):
                raise ValueError("A SPARQL ASK result must contain a boolean.")
            return ["boolean"], {(cell_value(data["boolean"]),)}
        bindings = data.get("results", {}).get("bindings") if isinstance(data.get("results"), dict) else None
        columns = data.get("head", {}).get("vars") if isinstance(data.get("head"), dict) else None
        if not isinstance(bindings, list) or not isinstance(columns, list):
            raise ValueError("Expected SPARQL JSON with head.vars and results.bindings.")
        rows = bindings
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Every SPARQL binding row must be an object.")
    elif isinstance(data, bool):
        return ["boolean"], {(cell_value(data),)}
    elif isinstance(data, list):
        rows = data
        if all(isinstance(row, dict) for row in rows):
            columns = list(dict.fromkeys(key for row in rows for key in row))
        elif all(row is None or isinstance(row, (str, int, float, bool)) for row in rows):
            columns = ["value"]
            rows = [{"value": value} for value in rows]
        else:
            raise ValueError("Use results: [{variable: value}, ...] or a list of scalars.")
    else:
        raise ValueError("Results must be a row list, SPARQL Results JSON, or an ASK boolean.")
    if len(rows) > MAX_ROWS or len(columns) > MAX_COLUMNS:
        raise ValueError(f"Comparison supports at most {MAX_ROWS} rows and {MAX_COLUMNS} columns.")
    if any(not isinstance(col, str) for col in columns) or len(set(columns)) != len(columns):
        raise ValueError("Result column names must be unique strings.")
    if any(set(row) - set(columns) for row in rows):
        raise ValueError("A binding contains a variable missing from head.vars.")
    return columns, {tuple(cell_value(row.get(col)) for col in columns) for row in rows}


def compare_results(expected, actual):
    """Compare row sets under a consistent renaming/reordering of columns.

    Parameters
    ----------
    expected, actual : object
        Accepted result structures; see ``result_table``.

    Returns
    -------
    dict
        Match flag, explanation, row counts, and a successful column mapping.

    Notes
    -----
    Row order and duplicates are ignored. A single column permutation must
    work for the entire table; independently sorting each row is incorrect.
    Empty results match even when a simple empty list lacks column metadata.
    """
    ecols, erows = result_table(expected)
    acols, arows = result_table(actual)
    outcome = {"match": False, "expected_rows": len(erows), "actual_rows": len(arows), "mapping": {}}
    if not erows and not arows:
        return {**outcome, "match": True, "message": "Both result sets are empty."}
    if len(erows) != len(arows):
        return {**outcome, "message": "The numbers of distinct rows differ."}
    if len(ecols) != len(acols):
        return {**outcome, "message": "The numbers of projected columns differ."}
    # Column value sets prune the search before checking complete row tuples.
    esig = [{row[i] for row in erows} for i in range(len(ecols))]
    asig = [{row[i] for row in arows} for i in range(len(acols))]
    for order in permutations(range(len(acols))):
        if all(esig[i] == asig[j] for i, j in enumerate(order)):
            if erows == {tuple(row[j] for j in order) for row in arows}:
                return {**outcome, "match": True, "message": "The distinct row sets match.",
                        "mapping": {ecols[i]: acols[j] for i, j in enumerate(order)}}
    return {**outcome, "message": "The values or their relationships within rows differ."}


class Settings(BaseModel):
    student_url: AnyHttpUrl = "http://localhost:8000/v1/query"
    sparql_url: AnyHttpUrl = "http://localhost:7070/sparql"
    questions_url: AnyHttpUrl = PUBLIC_ENDPOINT
    service_timeout: float = Field(default=90, ge=1, le=600)
    sparql_timeout: float = Field(default=30, ge=1, le=120)
    minimum_qid: int = Field(default=86, ge=0)


class Question(BaseModel):
    item: str = ""
    language: str = ""
    text: str = Field(min_length=1, max_length=20000)
    sparql: str = Field(default="", max_length=100000)


class Evaluation(BaseModel):
    settings: Settings
    question: Question


@asynccontextmanager
async def lifespan(app):
    # One shared client reuses connections; the semaphore bounds student load
    # across browser tabs. No automatic retries: they would distort metrics.
    async with httpx.AsyncClient(follow_redirects=False) as client:
        app.state.client = client
        app.state.student_slots = asyncio.Semaphore(8)
        yield


        

app = FastAPI(title="Keyboards text-to-SPARQL tester", lifespan=lifespan)


@app.get("/api/config")
async def config():
    return Settings(
        student_url=os.getenv("STUDENT_URL", "http://localhost:8000/v1/query"),
        sparql_url=os.getenv("SPARQL_URL", "http://localhost:7070/sparql"),
        questions_url=os.getenv("QUESTIONS_URL", PUBLIC_ENDPOINT),
    ).model_dump(mode="json")


async def request_json(client, method, url, timeout, purpose, **kwargs):
    """Make one timed HTTP request and explain common upstream failures.

    Parameters
    ----------
    client : httpx.AsyncClient
        Shared application client.
    method, url, purpose : str
        HTTP method, destination, and a human-readable service name.
    timeout : float
        Total deadline in seconds, in addition to HTTPX's I/O timeouts.
    **kwargs : dict
        Additional HTTPX request arguments.

    Returns
    -------
    dict
        Success, data, status code, error message, and elapsed milliseconds.
    """
    start = perf_counter()
    result = {"ok": False, "data": None, "status_code": None, "error": None}
    try:
        # wait_for also bounds a response that keeps delivering small chunks.
        response = await asyncio.wait_for(
            client.request(method, str(url), timeout=timeout, **kwargs), timeout
        )
        result["status_code"] = response.status_code
        if not 200 <= response.status_code < 300:
            advice = {
                404: "Check the full URL, including /v1/query for the student service.",
                422: 'The student service should accept JSON such as {"text": "your question"}.',
                429: "Reduce concurrency or wait before trying again.",
                405: "Check the URL and whether the endpoint accepts this HTTP method.",
            }.get(response.status_code, "Check the service logs and endpoint configuration.")
            result["error"] = f"{purpose} returned HTTP {response.status_code}. {advice}"
            result["response_excerpt"] = response.text[:1000]
        else:
            try:
                result["data"] = response.json()
                result["ok"] = True
            except (ValueError, UnicodeError):
                result["error"] = f"{purpose} returned non-JSON data. Check that the URL is an API endpoint rather than an HTML page."
                result["response_excerpt"] = response.text[:1000]
    except (asyncio.TimeoutError, httpx.TimeoutException):
        result["error"] = f"{purpose} exceeded {timeout:g} s. Check the service logs; increase the timeout or reduce concurrency."
    except httpx.RequestError as exc:
        result["error"] = f"Could not reach {purpose} ({type(exc).__name__}). Check that it is running and reachable from the frontend's Python process."
    result["latency_ms"] = round((perf_counter() - start) * 1000, 1)
    return result


async def sparql_request(client, endpoint, query, timeout, purpose):
    # GET works with QLever and the Wikibase query service; explicitly request
    # standard SPARQL Results JSON, rather than QLever's native JSON format.
    return await request_json(
        client, "GET", endpoint, timeout, purpose,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
    )


def with_prefixes(query):
    """Add common prefixes omitted by stored Wikibase query examples.

    Parameters
    ----------
    query : str
        SPARQL query, optionally with its own PREFIX declarations.

    Returns
    -------
    str
        Query with missing declarations added; explicit declarations win.
    """
    missing = [f"PREFIX {name}: <{uri}>" for name, uri in DEFAULT_PREFIXES.items()
               if not re.search(r"\bPREFIX\s+" + name + r"\s*:", query, re.I)]
    return "\n".join(missing) + "\n" + query


def bindings(data):
    """Validate a SELECT response and extract its binding objects.

    Parameters
    ----------
    data : dict
        Standard SPARQL Results JSON.

    Returns
    -------
    list of dict
        Binding objects.
    """
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        raise ValueError("Expected SPARQL Results JSON with results.bindings.")
    rows = data["results"].get("bindings")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected a list of SPARQL binding objects.")
    return rows


@app.post("/api/questions")
async def questions(settings: Settings, request: Request):
    # Use the live graph, since the distributed snapshot predates test items.
    query = PREFIXES + f"""SELECT DISTINCT ?item ?question ?sparql ?number WHERE {{
  ?item kbt:P11 kb:Q43 ; kbt:P27 ?question ; kbt:P31 ?sparql .
  FILTER(REGEX(STR(?item), "/Q[0-9]+$"))
  BIND(xsd:integer(STRAFTER(STR(?item), "/entity/Q")) AS ?number)
  FILTER(?number > {settings.minimum_qid})
}} ORDER BY ?number ?question ?sparql"""
    query = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" + query
    result, label_result = await asyncio.gather(
        sparql_request(request.app.state.client, settings.questions_url, query,
                       settings.sparql_timeout, "question catalogue"),
        sparql_request(request.app.state.client, settings.questions_url, LABEL_QUERY,
                       settings.sparql_timeout, "label lookup"),
    )
    tests, labels, warnings = [], {}, []
    if result["ok"]:
        try:
            for row in bindings(result["data"]):
                tests.append(Question(
                    item=row["item"]["value"], text=row["question"]["value"],
                    language=row["question"].get("xml:lang", ""),
                    sparql=row["sparql"]["value"],
                ).model_dump())
            tests.sort(key=lambda q: (int(q["item"].rsplit("Q", 1)[1]), q["language"], q["text"], q["sparql"]))
        except (ValueError, KeyError, TypeError) as exc:
            result.update(ok=False, error=f"The question catalogue has an unexpected result format: {exc}")
            tests = []
    if label_result["ok"]:
        try:
            labels = {row["item"]["value"]: row["itemLabel"]["value"]
                      for row in bindings(label_result["data"])}
        except (ValueError, KeyError, TypeError):
            warnings.append("Labels could not be read; full URIs will still be displayed.")
    else:
        warnings.append(label_result["error"])
    if result["ok"] and not tests:
        warnings.append(f"No query items above Q{settings.minimum_qid} were found. Check P11=Q43, P27 and P31, and use the live graph for the catalogue.")
    return {"questions": tests, "labels": labels, "warnings": warnings,
            "error": result["error"], "latency_ms": result["latency_ms"]}


@app.post("/api/evaluate")
async def evaluate(payload: Evaluation, request: Request):
    settings, question = payload.settings, payload.question
    client = request.app.state.client
    started = perf_counter()

    async def ask_student():
        async with request.app.state.student_slots:
            return await request_json(
                client, "POST", settings.student_url, settings.service_timeout,
                "student service", json={"text": question.text},
            )

    executed_sparql = with_prefixes(question.sparql)
    reference_call = sparql_request(
        client, settings.sparql_url, executed_sparql, settings.sparql_timeout,
        "reference SPARQL endpoint",
    ) if question.sparql.strip() else asyncio.sleep(0, result=None)
    student, reference = await asyncio.gather(ask_student(), reference_call)
    if reference:
        reference["executed_sparql"] = executed_sparql
    comparison = None
    # Keep the original service payload so debugging never depends on what the
    # result renderer understands. The student's SPARQL is NOT executed here.
    if student["ok"]:
        data = student["data"]
        try:
            if not isinstance(data, dict) or not isinstance(data.get("sparql"), str) or not data["sparql"].strip():
                raise ValueError("Return a JSON object containing a nonempty 'sparql' string.")
            if "results" not in data:
                raise ValueError("Return a 'results' field, including [] when there are no rows.")
            result_table(data["results"])
        except (ValueError, TypeError) as exc:
            student.update(ok=False, error=f"Student response format: {exc}")
    if reference and reference["ok"]:
        try:
            result_table(reference["data"])
        except (ValueError, TypeError) as exc:
            reference.update(ok=False, error=f"Reference response format: {exc}")
    if student["ok"] and reference and reference["ok"]:
        comparison = await asyncio.to_thread(compare_results, reference["data"], student["data"]["results"])
    return {"question": question.model_dump(), "student": student,
            "reference": reference, "comparison": comparison,
            "total_ms": round((perf_counter() - started) * 1000, 1)}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Keyboards · text-to-SPARQL tester</title>
  <style>
    :root { --red:#990000; --line:#ddd; --muted:#555; }
    * { box-sizing:border-box; }
    body { margin:0; color:#171717; background:#f5f5f5; font:16px/1.5 system-ui,sans-serif; }
    header { background:var(--red); color:white; padding:22px max(24px,calc((100vw - 1500px)/2)); }
    header p { margin:4px 0 0; opacity:.9; }
    h1 { font-size:1.6rem; margin:0; } h2 { font-size:1.2rem; margin:0 0 14px; }
    main { max-width:1548px; margin:auto; padding:24px; }
    section { background:white; border:1px solid var(--line); border-radius:6px; padding:20px; margin-bottom:20px; }
    .grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    label { display:block; font-size:.9rem; font-weight:600; }
    input,textarea,select,button { font:inherit; }
    input:not([type=checkbox]),textarea,select { width:100%; border:1px solid #aaa; border-radius:4px; padding:8px; background:white; color:#171717; }
    textarea { resize:vertical; } label small { font-weight:400; color:var(--muted); }
    button { padding:8px 15px; border:1px solid #777; border-radius:4px; cursor:pointer; background:white; color:#171717; }
    button.primary { background:var(--red); color:white; border-color:var(--red); }
    button:disabled { opacity:.5; cursor:default; }
    :focus-visible { outline:3px solid #c98200; outline-offset:2px; }
    .toolbar { display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin:14px 0; }
    .toolbar select,.toolbar input[type=number] { width:auto; }
    .hint { color:var(--muted); font-size:.9rem; margin:8px 0; }
    .notice { white-space:pre-wrap; padding:10px 12px; background:#fff4db; border-left:3px solid #a86c00; margin:10px 0; }
    .notice:empty { display:none; }
    .stats { display:flex; gap:28px; flex-wrap:wrap; border-top:1px solid var(--line); padding-top:12px; margin:14px 0; }
    .stats strong { display:block; font-size:1.4rem; } .stats span { font-size:.85rem; color:var(--muted); }
    .scroll { overflow:auto; } table { width:100%; border-collapse:collapse; font-size:.9rem; }
    th,td { text-align:left; padding:12px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
    th { background:#f1f1f1; } td { min-width:125px; } td:first-child { min-width:230px; }
    .result-table td { min-width:70px; padding:5px; } .result-table th { padding:5px; }
    pre { font:12px/1.5 ui-monospace,monospace; white-space:pre-wrap; overflow-wrap:anywhere; margin:8px 0; max-height:360px; overflow:auto; }
    details { margin:7px 0; } summary { cursor:pointer; } a { color:var(--red); overflow-wrap:anywhere; }
    .badge { display:inline-block; padding:3px 9px; border-radius:4px; font-size:.8rem; font-weight:650; background:#eee; }
    .match { background:#e4f1e6; color:#245a2c; } .mismatch,.error { background:#fbe8e8; color:#850000; }
    .error-text { color:#850000; overflow-wrap:anywhere; } .split { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    progress { width:100%; accent-color:var(--red); } code { font-size:.9em; }
    footer { color:var(--muted); font-size:.85rem; margin:10px 0 30px; }
    @media(max-width:800px) { .grid,.split { grid-template-columns:1fr; } main { padding:14px; } section { padding:15px; } }
  </style>
</head>
<body>
<header><h1>Keyboards · text-to-SPARQL tester</h1><p>Demonstrate a service. Compare its answers with executable reference queries.</p></header>
<main>
  <section>
    <h2>Connections</h2>
    <div class="grid">
      <label>Student service URL<input id="student_url" type="url" required value="http://localhost:8000/v1/query"><small>Full POST endpoint, including /v1/query.</small></label>
      <label>Reference SPARQL endpoint<input id="sparql_url" type="url" required value="http://localhost:7070/sparql"><small>Runs ground truth queries on your QLever snapshot.</small></label>
      <label>Question catalogue endpoint<input id="questions_url" type="url" required value="https://keyboards.wikibase.cloud/query/sparql"><small>Use the live Wikibase to find newly added tests and labels.</small></label>
    </div>
    <details><summary>Timeouts and question selection</summary><div class="grid">
      <label>Student timeout (seconds)<input id="service_timeout" type="number" min="1" max="600" value="90" required></label>
      <label>SPARQL timeout (seconds)<input id="sparql_timeout" type="number" min="1" max="120" value="30" required></label>
      <label>Only items above Q…<input id="minimum_qid" type="number" min="0" value="86" required></label>
    </div></details>
    <p class="hint">Connections are made by Python. “localhost” refers to the computer running this frontend. The student receives only the question text.</p>
  </section>

  <section>
    <h2>Try a question</h2>
    <label for="question">Question in English or Danish</label>
    <textarea id="question" rows="2">What is the width of a Yamaha P-150?</textarea>
    <details><summary>Optional ground truth SPARQL</summary><label for="reference-query" class="hint">Paste a SELECT or ASK query. Common kb:, kbt: and xsd: prefixes are added if missing.</label><textarea id="reference-query" rows="6" spellcheck="false"></textarea></details>
    <div class="toolbar"><button id="ask" class="primary">Send question</button><span id="ask-status" role="status"></span></div>
    <div id="ask-result"></div>
  </section>

  <section>
    <h2>Evaluate reference questions</h2>
    <p class="hint">Loads query items with P11 = Q43, question text in P27, and reference SPARQL in P31. Each question–query combination is a test.</p>
    <div class="toolbar">
      <button id="load">Load questions &amp; labels</button>
      <label>Language <select id="language" aria-label="Question language"><option value="">All languages</option></select></label>
      <label>Concurrent tests <input id="concurrency" type="number" min="1" max="8" value="4" required></label>
      <button id="run" class="primary" disabled>Run selected</button>
      <button id="stop" disabled>Stop after active tests</button>
      <button id="export" disabled>Export JSON</button>
    </div>
    <div id="notice" class="notice" role="status"></div>
    <p id="progress-text" role="status">Load the catalogue to begin.</p>
    <progress id="progress" value="0" max="1" aria-label="Completed tests"></progress>
    <div id="metrics" class="stats"></div>
    <div class="scroll"><table>
      <thead><tr><th><input id="select-all" type="checkbox" checked aria-label="Select all visible questions"> Question</th><th>Ground truth SPARQL</th><th>Student SPARQL</th><th>Student result</th><th>Ground truth result</th><th>Match &amp; latency</th></tr></thead>
      <tbody id="tests"><tr><td colspan="6">No questions loaded.</td></tr></tbody>
    </table></div>
    <details><summary>How results are compared</summary><p class="hint">Comparison uses sets of complete rows. It ignores duplicate rows, row order, and variable names, and finds a consistent column alignment across all rows. Numeric strings such as “1385.0” match JSON numbers such as 1385. URI labels are display-only. Language tags and literal datatypes are not compared. Blank nodes are unsupported. Extra projected label columns count as extra columns. Matching results on one dataset do not prove that two queries are equivalent.</p></details>
  </section>
  <footer>DTU teaching tool · Plain JavaScript, FastAPI and HTTPX · No LLM credentials are needed by this frontend.</footer>
</main>
<script>
"use strict";
const $ = id => document.getElementById(id);
let tests = [], labels = {}, busy = false, stopping = false, lastReport = null;
const settingsIds = ["student_url", "sparql_url", "questions_url", "service_timeout", "sparql_timeout", "minimum_qid"];
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function settings() {
  const value = {};
  for (const id of settingsIds) {
    if (!$(id).reportValidity()) throw new Error("Please correct the highlighted connection setting.");
    value[id] = $(id).type === "number" ? Number($(id).value) : $(id).value.trim();
  }
  return value;
}
async function api(path, body) {
  const response = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  let data;
  try { data = await response.json(); } catch { throw new Error("The frontend returned non-JSON data. Check its Python terminal."); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
function expandable(title, value) {
  const el = node("details"); el.append(node("summary", title), node("pre", typeof value === "string" ? value : JSON.stringify(value, null, 2))); return el;
}
function cell(value) {
  const el = node("span");
  const raw = value && typeof value === "object" && "value" in value ? value.value : value;
  if (typeof raw === "string" && /^https?:\/\//i.test(raw)) {
    const a = node("a", labels[raw] || raw); a.href = raw; a.target = "_blank"; a.rel = "noopener noreferrer"; a.title = raw; el.append(a);
  } else el.textContent = raw === null || raw === undefined ? "(unbound)" : String(raw);
  return el;
}
function resultView(data) {
  const wrapper = node("div", undefined, "scroll");
  if (data && !Array.isArray(data) && typeof data === "object" && "boolean" in data) data = data.boolean;
  if (typeof data === "boolean") { wrapper.append(node("span", String(data))); return wrapper; }
  let rows, columns;
  if (Array.isArray(data)) {
    rows = data.map(row => row && typeof row === "object" ? row : {value:row});
    columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
  } else { rows = data?.results?.bindings; columns = data?.head?.vars; }
  if (!Array.isArray(rows) || !Array.isArray(columns)) { wrapper.append(node("pre", JSON.stringify(data, null, 2))); return wrapper; }
  if (!rows.length) { wrapper.append(node("span", "Empty result set (0 rows)")); return wrapper; }
  const table = node("table", undefined, "result-table"), head = node("tr");
  for (const col of columns) head.append(node("th", col));
  table.append(head);
  for (const row of rows.slice(0, 50)) { const tr = node("tr"); for (const col of columns) { const td = node("td"); td.append(cell(row[col])); tr.append(td); } table.append(tr); }
  wrapper.append(table);
  if (rows.length > 50) wrapper.append(node("p", `Showing 50 of ${rows.length} rows; all rows were compared.`, "hint"));
  wrapper.append(expandable("Raw result JSON", data));
  return wrapper;
}
function showOutcome(outcome) {
  const box = node("div");
  if (!outcome) { box.append(node("span", "No reference query supplied.", "hint")); return box; }
  if (outcome.error) box.append(node("p", outcome.error, "error-text"));
  if (outcome.data !== null && outcome.data !== undefined) box.append(resultView(outcome.data));
  box.append(node("p", `${outcome.latency_ms} ms · HTTP ${outcome.status_code ?? "no response"}`, "hint"));
  if (outcome.response_excerpt) box.append(expandable("Response excerpt", outcome.response_excerpt));
  if (outcome.executed_sparql) box.append(expandable("Executed reference query (with prefixes)", outcome.executed_sparql));
  return box;
}
function studentResult(outcome) {
  const box = node("div");
  if (outcome.error) box.append(node("p", outcome.error, "error-text"));
  if (outcome.data && "results" in Object(outcome.data)) box.append(resultView(outcome.data.results));
  if (outcome.data !== null) box.append(expandable("Full service response", outcome.data));
  if (outcome.response_excerpt) box.append(expandable("Response excerpt", outcome.response_excerpt));
  return box;
}
function verdict(result) {
  const box = node("div"), c = result.comparison;
  box.append(node("span", c ? (c.match ? "Match" : "Mismatch") : (result.student.ok && !result.reference ? "Response received" : "Not compared"), "badge " + (c ? (c.match ? "match" : "mismatch") : "")));
  box.append(node("p", c?.message || "", "hint"));
  box.append(node("p", `Student: ${result.student.latency_ms} ms`, "hint"));
  if (result.reference) box.append(node("p", `Reference: ${result.reference.latency_ms} ms`, "hint"));
  if (c) box.append(expandable("Comparison details", c));
  return box;
}
function visible(entry) { return !$("language").value || (entry.question.language || "untagged") === $("language").value; }
function renderRows() {
  const body = $("tests"); body.replaceChildren();
  for (const entry of tests.filter(visible)) {
    const tr = node("tr"), q = entry.question, first = node("td"), check = node("input");
    check.type = "checkbox"; check.checked = entry.selected; check.disabled = busy; check.setAttribute("aria-label", `Select ${q.text}`); check.onchange = () => { entry.selected = check.checked; };
    first.append(check, node("strong", " " + (q.item.split("/").pop() || "Question")), node("span", ` · ${q.language || "no language tag"}`, "hint"), node("p", q.text));
    if (/^https?:\/\//.test(q.item)) first.append(cell(q.item));
    const refQuery = node("td"); refQuery.append(expandable("View SPARQL", q.sparql));
    const query = node("td"), actual = node("td"), expected = node("td"), status = node("td");
    if (entry.result) {
      query.append(expandable("View SPARQL", entry.result.student.data?.sparql || "No SPARQL returned."));
      actual.append(studentResult(entry.result.student)); expected.append(showOutcome(entry.result.reference)); status.append(verdict(entry.result));
    } else { status.textContent = entry.error || entry.status || "Ready"; if (entry.error) status.className = "error-text"; }
    tr.append(first, refQuery, query, actual, expected, status); body.append(tr);
  }
  if (!body.children.length) { const row=node("tr"), td=node("td", "No questions in this selection."); td.colSpan=6; row.append(td); body.append(row); }
}
function setBusy(value) {
  busy = value;
  for (const id of [...settingsIds,"ask","load","language","concurrency","select-all"]) $(id).disabled = value;
  $("run").disabled = value || !tests.length; $("stop").disabled = !value || !lastReport?.running;
  $("export").disabled = value || !lastReport;
  renderRows();
}
function metrics(report) {
  const outcomes = report.results, complete = outcomes.length, valid = outcomes.filter(r => r.student.ok).length;
  const comparable = outcomes.filter(r => r.comparison), matches = comparable.filter(r => r.comparison.match).length;
  const latencies = outcomes.map(r => r.student.latency_ms).sort((a,b)=>a-b);
  const median = complete ? (latencies[Math.floor((complete-1)/2)] + latencies[Math.floor(complete/2)]) / 2 : null;
  $("metrics").replaceChildren();
  const items = [[`${complete} / ${report.selected}`, "Responses collected"], [String(valid), "Valid service responses"], [String(matches), "Matches"], [String(complete - comparable.length), "Not compared"], [median === null ? "—" : `${median.toFixed(0)} ms`, "Median student latency (all responses)"]];
  for (const [value,label] of items) { const div = node("div"); div.append(node("strong", value), node("span", label)); $("metrics").append(div); }
}
$("ask").onclick = async () => {
  try {
    const config = settings(), text = $("question").value.trim(); if (!text) throw new Error("Enter a question first.");
    setBusy(true); $("ask-status").textContent = "Waiting for the student service…"; $("ask-result").replaceChildren();
    const result = await api("/api/evaluate", {settings:config, question:{text, sparql:$("reference-query").value}});
    const columns = node("div", undefined, "split"), a = node("div"), b = node("div");
    a.append(node("h3", "Student result"), studentResult(result.student), expandable("Generated SPARQL", result.student.data?.sparql || "No SPARQL returned."));
    b.append(node("h3", "Ground truth result"), showOutcome(result.reference), verdict(result)); columns.append(a,b); $("ask-result").append(columns);
    $("ask-status").textContent = `Finished in ${(result.total_ms / 1000).toFixed(2)} s.`;
  } catch (error) { $("ask-status").textContent = error.message; } finally { setBusy(false); }
};
$("load").onclick = async () => {
  try {
    const config=settings(); setBusy(true); $("notice").textContent="Loading questions and labels…";
    const data=await api("/api/questions", config);
    if (data.error) throw new Error(data.error);
    labels=data.labels; tests=data.questions.map(question => ({question,selected:true})); lastReport=null;
    $("metrics").replaceChildren(); $("progress").value=0; $("language").replaceChildren(new Option("All languages", ""));
    for (const lang of [...new Set(tests.map(e=>e.question.language || "untagged"))].sort()) $("language").add(new Option(lang,lang));
    $("notice").textContent=data.warnings.join("\n"); $("progress-text").textContent=`${tests.length} test${tests.length === 1 ? "" : "s"} loaded. Select questions and run the service.`;
  } catch (error) { $("notice").textContent=error.message; } finally { setBusy(false); }
};
$("run").onclick = async () => {
  let report;
  try {
    const config=settings(); if (!$("concurrency").reportValidity()) return;
    const queue=tests.filter(e=>visible(e) && e.selected), concurrency=Number($("concurrency").value);
    if (!queue.length) throw new Error("Select at least one visible question.");
    stopping=false; const start=performance.now();
    report={started_at:new Date().toISOString(), settings:config, concurrency, selected:queue.length, results:[], errors:[], running:true}; lastReport=report;
    for (const entry of tests) { entry.result=null; entry.error=null; entry.status="Ready"; }
    setBusy(true); $("notice").textContent=""; $("progress").max=queue.length; $("progress").value=0; metrics(report);
    let next=0, completed=0;
    async function worker() {
      while (!stopping && next<queue.length) {
        const entry=queue[next++]; entry.status="Running…"; renderRows();
        try { entry.result=await api("/api/evaluate", {settings:config,question:entry.question}); report.results.push(entry.result); }
        catch (error) { entry.error=error.message; report.errors.push({question:entry.question,error:error.message}); }
        completed++; $("progress").value=completed; $("progress-text").textContent=`${completed} / ${queue.length} completed${stopping ? " · stopping…" : ""}`; renderRows(); metrics(report);
      }
    }
    await Promise.all(Array.from({length:Math.min(concurrency,queue.length)}, worker));
    report.elapsed_ms=performance.now()-start; report.stopped=stopping;
    $("progress-text").textContent=`${completed} / ${queue.length} completed in ${(report.elapsed_ms/1000).toFixed(2)} s${stopping ? " · stopped" : ""}.`;
    if (report.errors.length) $("notice").textContent=`${report.errors.length} frontend request(s) failed. See the rows or JSON export.`;
  } catch (error) { $("notice").textContent=error.message; } finally { if (report) report.running=false; setBusy(false); }
};
$("stop").onclick = () => { stopping=true; $("stop").disabled=true; $("progress-text").textContent="Stopping after the active tests finish…"; };
$("export").onclick = () => {
  const url=URL.createObjectURL(new Blob([JSON.stringify(lastReport,null,2)],{type:"application/json"}));
  const link=node("a"); link.href=url; link.download="text-to-sparql-results.json"; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$("select-all").onchange = () => { for (const entry of tests.filter(visible)) entry.selected=$("select-all").checked; renderRows(); };
$("language").onchange=renderRows;
fetch("/api/config").then(r=>{if (!r.ok) throw new Error("Could not load default settings."); return r.json();}).then(data=>{ for (const id of settingsIds) $(id).value=data[id]; }).catch(error=>{$("notice").textContent=error.message;});
</script>
</body>
</html>
"""
