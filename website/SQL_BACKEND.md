# Local SQL Website Backend

The static website can still load `website/data.js`, but local users can now run the same table against a SQLite database. This removes the hard dependency on a Cornell `/data/...` trace layout for local interaction and gives the table SQL-backed search and sorting.

## Local Workflow

From the repository root:

```bash
python website/generate_database.py
python website/local_server.py
```

Then open:

```text
http://127.0.0.1:8081/
```

`index.html` automatically detects the local `/api/bootstrap` endpoint. When it is available, the page uses the SQLite backend. When it is not available, the page falls back to the checked-in `website/data.js` payload, so static hosting still works.

## Regenerating From Local Traces

The historical config contains trace paths under:

```text
/data/ccl-bench_trace_collection
```

For a downloaded local checkout, keep traces anywhere and remap them at generation time:

```bash
export CCLBENCH_TRACE_ROOT=$PWD/traces
python website/generate_data.py
python website/generate_database.py
python website/local_server.py
```

For example, a configured trace:

```text
/data/ccl-bench_trace_collection/foo
```

will be read from:

```text
$CCLBENCH_TRACE_ROOT/foo
```

You can also pass the trace root explicitly:

```bash
python website/generate_data.py --trace-root ./traces
```

## Database Schema

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

## API

`website/local_server.py` serves the repository root plus these JSON endpoints:

- `GET /api/health`
- `GET /api/bootstrap`
- `GET /api/rows?search=<text>&sort_key=<column>&sort_dir=1|-1`

The main table uses `/api/rows` for filtering and sorting. Metric sorts use the existing frontend column key format, for example:

```text
sort_key=__m__avg_step_time&sort_dir=1
```

## Static Hosting

The public static site can keep using:

- `website/benchmark_data.json`
- `website/data.js`
- `index.html`

The local SQL server is an optional runtime for users who download CCL-Bench and want to query the benchmark locally without editing JavaScript data files.

## Remaining Production Choices

- The local server uses Python's standard library and SQLite, so there are no new Python dependencies.
- `website/sql_demo.html` remains a small standalone browser-SQL demo, but the main path is now `index.html` plus `website/local_server.py`.
- If we want SQL in purely static hosting without a Python server, we should vendor sql.js or DuckDB-WASM assets instead of using a CDN.
