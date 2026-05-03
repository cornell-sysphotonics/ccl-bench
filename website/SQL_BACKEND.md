# SQL Backend Prototype

The production website currently loads `website/data.js`, which embeds the full benchmark payload as a JavaScript object. That path is simple and works from static hosting, but every filter and sort operation happens over in-memory JavaScript arrays.

This prototype adds a static SQLite artifact:

```bash
python website/generate_data.py
python website/generate_database.py
python -m http.server
```

Then open:

```text
http://localhost:8000/website/sql_demo.html
```

## What It Generates

`website/generate_database.py` reads `website/benchmark_data.json` and writes:

```text
website/benchmark_data.sqlite
```

The database has normalized tables:

- `metadata`: one row per benchmark trace, with flattened workload-card fields.
- `metrics`: long-form metric values keyed by `(row_id, metric_name)`.
- `metric_info`: labels, units, categories, directionality, and trace compatibility.
- `metric_categories`: display category ordering.
- `run_info`: generator metadata such as `generated_at`.

## Why This Direction

This gives us real SQL semantics for:

- filtering by phase, model, framework, hardware, and communication library;
- sorting by metadata columns or selected metric values;
- joining benchmark rows to selected metrics without reshaping a large JSON object in the browser.

It also keeps the site static. There is no server-side database process; the browser loads the SQLite file and queries it through sql.js.

## Current Limitations

- `sql_demo.html` uses sql.js from a CDN, so it needs network access unless we vendor `sql-wasm.js` and `sql-wasm.wasm`.
- The main `index.html` still uses `data.js`. This is intentional for the first prototype so the production page remains unchanged.
- A full migration should move table rendering behind a small data-access layer so both `data.js` and SQLite can be supported during transition.

## Recommended Migration Plan

1. Keep generating `benchmark_data.json` as the canonical intermediate file.
2. Generate both `data.js` and `benchmark_data.sqlite` from that canonical file.
3. Add a data-access module with methods like `loadRows`, `searchRows`, `sortRows`, and `getMetricInfo`.
4. Switch the main table to the SQL data-access module.
5. Vendor sql.js or DuckDB-WASM if the public artifact must work without third-party CDNs.
