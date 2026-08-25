#!/usr/bin/env python
"""Minimal web interface for the ENEXRE Neo4j prototype."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from neo4j import GraphDatabase


DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "enexre12345"


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ENEXRE Prototype</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #16202a;
      --muted: #637083;
      --line: #d8dee8;
      --blue: #2563eb;
      --green: #0f766e;
      --amber: #b45309;
      --danger: #b91c1c;
      --shadow: 0 12px 28px rgba(22, 32, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .wrap {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
    }
    .topbar {
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--amber);
    }
    .dot.ok { background: var(--green); }
    main {
      padding: 22px 0 40px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 180px 1fr 120px auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 16px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }
    select, input, button {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }
    select, input { padding: 0 12px; }
    button {
      padding: 0 16px;
      border-color: var(--blue);
      background: var(--blue);
      color: white;
      cursor: pointer;
      font-weight: 650;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric, .table-shell, .details {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric {
      padding: 14px;
      min-height: 72px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 5px;
      font-size: 22px;
      line-height: 1.1;
    }
    .table-shell {
      overflow: hidden;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      background: #fbfcfe;
    }
    tr:last-child td { border-bottom: 0; }
    .id { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; }
    .confidence {
      color: var(--green);
      font-weight: 700;
    }
    .evidence {
      color: var(--muted);
      line-height: 1.45;
      max-width: 520px;
    }
    .empty {
      padding: 32px;
      color: var(--muted);
      text-align: center;
    }
    .error {
      margin-bottom: 16px;
      padding: 12px 14px;
      border: 1px solid #fecaca;
      border-radius: 8px;
      background: #fff1f2;
      color: var(--danger);
      display: none;
    }
    @media (max-width: 780px) {
      .topbar { align-items: flex-start; flex-direction: column; padding: 14px 0; }
      .toolbar { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <h1>ENEXRE Chemical-Disease Graph</h1>
      <div class="status"><span id="dot" class="dot"></span><span id="status">Checking Neo4j</span></div>
    </div>
  </header>
  <main class="wrap">
    <form id="search" class="toolbar">
      <div>
        <label for="mode">Filter</label>
        <select id="mode">
          <option value="pmid">PMID</option>
          <option value="chemical_id">Chemical MeSH ID</option>
          <option value="disease_id">Disease MeSH ID</option>
          <option value="all">All relations</option>
        </select>
      </div>
      <div>
        <label for="q">Value</label>
        <input id="q" value="18801087" placeholder="18801087 or D004280">
      </div>
      <div>
        <label for="limit">Limit</label>
        <input id="limit" type="number" min="1" max="100" value="10">
      </div>
      <button id="run" type="submit">Search</button>
    </form>
    <div id="error" class="error"></div>
    <section class="metrics">
      <div class="metric"><span>Chemical nodes</span><strong id="m-chem">-</strong></div>
      <div class="metric"><span>Disease nodes</span><strong id="m-dis">-</strong></div>
      <div class="metric"><span>CID relationships</span><strong id="m-rel">-</strong></div>
      <div class="metric"><span>Current result</span><strong id="m-current">-</strong></div>
    </section>
    <section class="table-shell">
      <table>
        <thead>
          <tr>
            <th style="width: 90px;">PMID</th>
            <th style="width: 120px;">Chemical ID</th>
            <th style="width: 120px;">Disease ID</th>
            <th>Chemical</th>
            <th>Disease</th>
            <th style="width: 110px;">Confidence</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody id="rows">
          <tr><td colspan="7" class="empty">Run a search to view relations.</td></tr>
        </tbody>
      </table>
    </section>
  </main>
  <script>
    const state = {
      mode: document.querySelector("#mode"),
      q: document.querySelector("#q"),
      limit: document.querySelector("#limit"),
      rows: document.querySelector("#rows"),
      error: document.querySelector("#error"),
      status: document.querySelector("#status"),
      dot: document.querySelector("#dot"),
      run: document.querySelector("#run")
    };

    function text(value) {
      return value === null || value === undefined || value === "" ? "-" : String(value);
    }

    function showError(message) {
      state.error.textContent = message;
      state.error.style.display = message ? "block" : "none";
    }

    async function api(path) {
      const response = await fetch(path);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    async function loadHealth() {
      try {
        const data = await api("/api/health");
        document.querySelector("#m-chem").textContent = data.chemical_nodes;
        document.querySelector("#m-dis").textContent = data.disease_nodes;
        document.querySelector("#m-rel").textContent = data.cid_relationships;
        state.status.textContent = "Neo4j connected";
        state.dot.classList.add("ok");
      } catch (error) {
        state.status.textContent = "Neo4j unavailable";
        showError(error.message);
      }
    }

    function renderRows(rows) {
      document.querySelector("#m-current").textContent = rows.length;
      if (!rows.length) {
        state.rows.innerHTML = '<tr><td colspan="7" class="empty">No CID relationships found.</td></tr>';
        return;
      }
      state.rows.innerHTML = rows.map(row => `
        <tr>
          <td class="id">${text(row.pmid)}</td>
          <td class="id">${text(row.chemical_id)}</td>
          <td class="id">${text(row.disease_id)}</td>
          <td>${text(row.chemical)}</td>
          <td>${text(row.disease)}</td>
          <td class="confidence">${Number(row.confidence).toFixed(4)}</td>
          <td class="evidence">${text(row.evidence)}</td>
        </tr>
      `).join("");
    }

    async function search(event) {
      event.preventDefault();
      showError("");
      state.run.disabled = true;
      try {
        const params = new URLSearchParams({ mode: state.mode.value, limit: state.limit.value });
        if (state.mode.value !== "all") params.set("q", state.q.value.trim());
        const data = await api(`/api/relations?${params.toString()}`);
        renderRows(data.rows);
      } catch (error) {
        showError(error.message);
      } finally {
        state.run.disabled = false;
      }
    }

    state.mode.addEventListener("change", () => {
      const disabled = state.mode.value === "all";
      state.q.disabled = disabled;
      state.q.placeholder = disabled ? "No value needed" : "18801087 or D004280";
    });
    document.querySelector("#search").addEventListener("submit", search);
    loadHealth().then(() => document.querySelector("#search").requestSubmit());
  </script>
</body>
</html>
"""


class GraphService:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def health(self) -> dict[str, Any]:
        query = """
        MATCH (c:Chemical)
        WITH count(c) AS chemical_nodes
        MATCH (d:Disease)
        WITH chemical_nodes, count(d) AS disease_nodes
        MATCH ()-[r:CID]->()
        RETURN chemical_nodes, disease_nodes, count(r) AS cid_relationships
        """
        with self.driver.session() as session:
            row = session.run(query).single()
            return dict(row) if row else {}

    def relations(self, mode: str, value: str, limit: int) -> list[dict[str, Any]]:
        filters = []
        params: dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if mode == "pmid":
            filters.append("r.pmid = $value")
            params["value"] = value
        elif mode == "chemical_id":
            filters.append("c.mesh_id = $value")
            params["value"] = value
        elif mode == "disease_id":
            filters.append("d.mesh_id = $value")
            params["value"] = value
        elif mode != "all":
            raise ValueError("Unsupported filter mode")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"""
        MATCH (c:Chemical)-[r:CID]->(d:Disease)
        {where_clause}
        RETURN
          c.mesh_id AS chemical_id,
          c.label AS chemical,
          d.mesh_id AS disease_id,
          d.label AS disease,
          r.confidence AS confidence,
          r.pmid AS pmid,
          r.evidence AS evidence
        ORDER BY r.confidence DESC, r.pmid
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(record) for record in session.run(query, params)]


class PrototypeHandler(BaseHTTPRequestHandler):
    service: GraphService

    def log_message(self, format: str, *args: Any) -> None:
        return

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.write_html(INDEX_HTML)
                return
            if parsed.path == "/api/health":
                self.write_json(HTTPStatus.OK, self.service.health())
                return
            if parsed.path == "/api/relations":
                params = parse_qs(parsed.query)
                mode = params.get("mode", ["pmid"])[0]
                value = params.get("q", [""])[0].strip()
                limit = int(params.get("limit", ["10"])[0])
                if mode != "all" and not value:
                    self.write_json(HTTPStatus.BAD_REQUEST, {"error": "Filter value is required."})
                    return
                self.write_json(
                    HTTPStatus.OK,
                    {"rows": self.service.relations(mode=mode, value=value, limit=limit)},
                )
                return
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ENEXRE prototype web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", DEFAULT_NEO4J_USER))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = GraphService(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    service.driver.verify_connectivity()
    PrototypeHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), PrototypeHandler)
    print(f"ENEXRE prototype running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
