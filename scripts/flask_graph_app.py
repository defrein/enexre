#!/usr/bin/env python
"""Flask knowledge graph viewer for ENEXRE Chemical-Disease relations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = PROJECT_DIR / "data" / "graph"
MAX_ABSTRACT_CHARS = 12000
MAX_ABSTRACTS_PER_REQUEST = 20
EXAMPLE_CASES = [
    {
        "id": "case-connected-lithium",
        "title": "Connected: lithium adverse effects",
        "category": "connected",
        "abstracts": [
            "Lithium caused nephrogenic diabetes insipidus and induced depressive symptoms in a treated patient.",
            "Lithium treatment was associated with mania and hypomania in patients with bipolar disorder.",
            "Lithium induced tremor and renal tubular acidosis during long-term therapy.",
        ],
    },
    {
        "id": "case-connected-dobutamine",
        "title": "Connected: dobutamine arrhythmia",
        "category": "connected",
        "abstracts": [
            "Dobutamine induced QT prolongation and torsade de pointes ventricular tachycardia in a patient with heart failure.",
            "Intermittent dobutamine treatment caused ventricular tachycardia and QT prolongation.",
        ],
    },
    {
        "id": "case-disconnected",
        "title": "Disconnected: separate chemicals",
        "category": "not connected",
        "abstracts": [
            "Lithium caused nephrogenic diabetes insipidus in a patient receiving maintenance therapy.",
            "5-fluorouracil induced hyperammonemic encephalopathy during chemotherapy.",
            "Dobutamine caused torsade de pointes ventricular tachycardia in severe heart failure.",
        ],
    },
    {
        "id": "case-mixed",
        "title": "Mixed: one bridge plus isolated relation",
        "category": "mixed",
        "abstracts": [
            "Lithium caused nephrogenic diabetes insipidus and was associated with depressive symptoms.",
            "Lithium treatment was related to mania and hypomania.",
            "5-fluorouracil caused hyperammonemic encephalopathy during continuous infusion.",
        ],
    },
    {
        "id": "case-cooccurrence",
        "title": "Co-occurrence candidate: weak wording",
        "category": "candidate",
        "abstracts": [
            "Lithium was discussed in a case report mentioning bipolar disorder and nephrogenic diabetes insipidus.",
            "Dobutamine therapy was described in a patient with QT prolongation and heart failure.",
        ],
    },
]
RELATION_CUES = [
    ("caused by", "caused by"),
    ("induced by", "induced by"),
    ("associated with", "associated with"),
    ("related to", "related to"),
    ("resulting from", "resulting from"),
    ("due to", "due to"),
    ("secondary to", "secondary to"),
    ("causes", "causes"),
    ("cause", "causes"),
    ("caused", "caused"),
    ("induces", "induces"),
    ("induce", "induces"),
    ("induced", "induced"),
    ("triggered", "triggered"),
    ("provoked", "provoked"),
    ("produced", "produced"),
    ("developed", "developed"),
    ("complicated by", "complicated by"),
    ("risk of", "risk of"),
    ("increased risk", "increased risk"),
    ("toxicity", "toxicity"),
    ("adverse effect", "adverse effect"),
]


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ENEXRE Knowledge Graph</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #fafafa;
      --surface: #ffffff;
      --surface-2: #f1f4f7;
      --text: #202123;
      --muted: #6b7280;
      --line: #e5e7eb;
      --chemical: #1f7a8c;
      --disease: #9b3d3d;
      --edge: #7b8794;
      --focus: #111827;
      --good: #287455;
      --shadow: 0 18px 46px rgba(17, 24, 39, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: transparent;
      border-bottom: 0;
    }
    .wrap {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
    }
    .topbar {
      min-height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 17px;
      line-height: 1.15;
      letter-spacing: 0;
      font-weight: 700;
    }
    .status {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--good);
    }
    main {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      padding: 18px 0 42px;
    }
    .intro {
      width: min(860px, 100%);
      margin: 28px auto 4px;
      text-align: center;
    }
    .intro h2 {
      margin: 0;
      font-size: clamp(28px, 5vw, 44px);
      line-height: 1.08;
      font-weight: 760;
      letter-spacing: 0;
    }
    .intro p {
      margin: 12px auto 0;
      max-width: 640px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
    }
    .abstract-shell {
      width: min(860px, 100%);
      margin: 0 auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 12px;
    }
    .abstract-form {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 8px;
      align-items: stretch;
    }
    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 2px 4px 0;
    }
    textarea {
      width: 100%;
      min-height: 160px;
      resize: vertical;
      border: 0;
      border-radius: 12px;
      padding: 12px 13px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      line-height: 1.45;
      letter-spacing: 0;
      outline: none;
    }
    .app-tabs {
      width: min(860px, 100%);
      margin: 0 auto;
      display: flex;
      justify-content: center;
      gap: 8px;
    }
    .tab-button {
      width: auto;
      min-height: 36px;
      padding: 0 14px;
      background: var(--surface);
      color: var(--text);
      border-color: var(--line);
      border-radius: 999px;
      font-weight: 650;
    }
    .tab-button.active {
      background: var(--focus);
      border-color: var(--focus);
      color: #fff;
    }
    .view-section {
      grid-column: 1 / -1;
      display: none;
      grid-template-columns: 1fr;
      gap: 18px;
    }
    .view-section.active { display: grid; }
    .view-section > .graph-shell { grid-column: 1; }
    .view-section > aside {
      grid-column: 1;
      grid-row: auto;
    }
    .input-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 0;
      padding: 0;
    }
    .tool-button {
      width: auto;
      min-height: 32px;
      padding: 0 10px;
      background: transparent;
      border-color: transparent;
      color: var(--text);
      font-weight: 650;
      border-radius: 999px;
    }
    .tool-button:hover { background: #f3f4f6; }
    .examples {
      width: min(860px, 100%);
      margin: 16px auto 0;
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }
    .example-button, .case-button {
      width: 100%;
      min-height: 70px;
      padding: 12px 14px;
      background: #fff;
      border-color: var(--line);
      color: var(--text);
      text-align: left;
      font-weight: 650;
      line-height: 1.3;
      border-radius: 14px;
    }
    .example-button:hover, .case-button:hover { background: #f7f7f8; }
    .example-button span, .case-button span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .example-button small, .case-button small {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
      line-height: 1.35;
    }
    #analyzeView.has-results .examples { display: none; }
    .case-library {
      grid-column: 1 / -1;
      width: min(900px, 100%);
      margin: 0 auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: none;
      padding: 16px;
    }
    .case-list {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      max-height: 72vh;
      overflow: auto;
      padding-right: 4px;
    }
    .toolbar {
      width: min(980px, 100%);
      margin: 0 auto;
      display: none;
      grid-template-columns: 168px minmax(220px, 1fr) 142px 118px 106px;
      gap: 10px;
      align-items: end;
    }
    #analyzeView:not(.has-results) > aside,
    #analyzeView:not(.has-results) > .graph-shell,
    #analyzeView:not(.has-results) > .table-shell {
      display: none;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    select, input, button {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 10px;
      font: inherit;
      letter-spacing: 0;
    }
    select, input {
      padding: 0 11px;
      background: var(--surface);
      color: var(--text);
    }
    button {
      padding: 0 14px;
      background: var(--focus);
      border-color: var(--focus);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      border-radius: 999px;
    }
    #runAbstract {
      width: auto;
      min-width: 106px;
    }
    button:disabled {
      opacity: .62;
      cursor: wait;
    }
    .graph-shell, aside, .table-shell {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: none;
      overflow: hidden;
    }
    .graph-shell {
      width: min(1180px, 100%);
      margin: 0 auto;
      min-height: 600px;
      position: relative;
    }
    svg {
      display: block;
      width: 100%;
      height: 600px;
      background:
        linear-gradient(var(--surface-2) 1px, transparent 1px),
        linear-gradient(90deg, var(--surface-2) 1px, transparent 1px);
      background-size: 28px 28px;
    }
    .edge {
      stroke: var(--edge);
      stroke-opacity: .42;
    }
    .edge-label-bg {
      fill: rgba(255, 255, 255, .88);
      stroke: var(--line);
      stroke-width: 1px;
      rx: 4px;
    }
    .edge-label {
      fill: var(--text);
      font-size: 11px;
      font-weight: 700;
      pointer-events: none;
    }
    .node {
      stroke: #fff;
      stroke-width: 2px;
      cursor: pointer;
    }
    .node.chemical { fill: var(--chemical); }
    .node.disease { fill: var(--disease); }
    .label {
      fill: var(--text);
      font-size: 11px;
      paint-order: stroke;
      stroke: rgba(255, 255, 255, .88);
      stroke-width: 4px;
      pointer-events: none;
    }
    .empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: var(--muted);
      padding: 24px;
      text-align: center;
    }
    aside {
      width: min(1180px, 100%);
      margin: 0 auto;
      align-self: start;
      display: grid;
      grid-template-columns: 1.1fr 1fr 1.4fr;
    }
    .panel-section {
      padding: 14px;
      border-bottom: 0;
      border-right: 1px solid var(--line);
    }
    .panel-section:last-child { border-right: 0; }
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .metric {
      min-height: 66px;
      padding: 10px;
      border: 0;
      border-radius: 12px;
      background: #f7f7f8;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 22px;
      line-height: 1.1;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .swatch {
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }
    .swatch::before {
      content: "";
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: var(--chemical);
    }
    .swatch.disease::before { background: var(--disease); }
    .details dl {
      margin: 0;
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr);
      gap: 8px 10px;
      font-size: 13px;
    }
    .details dt { color: var(--muted); }
    .details dd {
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .evidence {
      color: var(--muted);
      line-height: 1.45;
      max-height: 170px;
      overflow: auto;
    }
    .abstract-preview {
      color: var(--muted);
      line-height: 1.5;
      max-height: 150px;
      overflow: auto;
      font-size: 13px;
    }
    .table-shell {
      width: min(1180px, 100%);
      margin: 0 auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      color: var(--muted);
      background: #fbfcfd;
      font-size: 12px;
    }
    tr:last-child td { border-bottom: 0; }
    .id {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
    }
    .confidence {
      color: var(--good);
      font-weight: 700;
    }
    .error {
      grid-column: 1 / -1;
      display: none;
      padding: 11px 12px;
      border: 1px solid #f2b8b5;
      border-radius: 8px;
      background: #fff3f2;
      color: #a33d35;
      font-size: 14px;
    }
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
      .view-section { grid-template-columns: 1fr; }
      .view-section > .graph-shell, .view-section > aside {
        grid-column: 1;
        grid-row: auto;
      }
      .toolbar { grid-template-columns: 1fr 1fr; }
      .toolbar button { grid-column: 1 / -1; }
      .abstract-form { grid-template-columns: 1fr; }
      aside { grid-template-columns: 1fr; }
      .panel-section {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .panel-section:last-child { border-bottom: 0; }
      .examples { grid-template-columns: 1fr 1fr; }
      .case-list { grid-template-columns: 1fr; }
      svg { height: 560px; }
      .graph-shell { min-height: 560px; }
    }
    @media (max-width: 620px) {
      .topbar { align-items: flex-start; flex-direction: column; padding: 12px 0; }
      .toolbar { grid-template-columns: 1fr; }
      .composer-actions {
        align-items: stretch;
        flex-direction: column;
      }
      .input-tools { grid-template-columns: 1fr; }
      .examples { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr 1fr; }
      th:nth-child(3), td:nth-child(3), th:nth-child(6), td:nth-child(6) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <h1>ENEXRE Knowledge Graph</h1>
      <div class="status"><span class="dot"></span><span id="status">Loading graph artifacts</span></div>
    </div>
  </header>

  <main class="wrap">
    <nav class="app-tabs" aria-label="Views">
      <button id="tabAnalyze" class="tab-button active" type="button">Analyze</button>
      <button id="tabCases" class="tab-button" type="button">Case Library</button>
    </nav>

    <div id="error" class="error"></div>

    <section id="analyzeView" class="view-section active">
      <section class="intro">
        <h2>Analyze biomedical abstracts</h2>
        <p>Paste one or more abstracts, then extract Chemical-Disease entities, relation cues, and an interactive knowledge graph.</p>
      </section>

      <section class="abstract-shell">
        <form id="abstractControls" class="abstract-form">
          <div>
            <label for="abstractText">Abstract input</label>
            <textarea id="abstractText" placeholder="Paste one or many abstracts. Use ==== on its own line to separate abstracts, or click an example below."></textarea>
            <div class="composer-actions">
              <div class="input-tools">
                <button id="clearInput" class="tool-button" type="button">Clear</button>
                <button id="appendSeparator" class="tool-button" type="button">Add Separator</button>
                <button id="loadMixedCase" class="tool-button" type="button">Load Mixed Case</button>
              </div>
              <button id="runAbstract" type="submit">Analyze</button>
            </div>
          </div>
        </form>
        <div id="examples" class="examples"></div>
      </section>

      <aside>
        <section class="panel-section">
          <h2>Result</h2>
          <div class="metrics">
            <div class="metric"><span>Abstracts</span><strong id="mAbstracts">0</strong></div>
            <div class="metric"><span>Chemicals</span><strong id="mChem">0</strong></div>
            <div class="metric"><span>Diseases</span><strong id="mDisease">0</strong></div>
            <div class="metric"><span>Relations</span><strong id="mEdges">0</strong></div>
          </div>
        </section>
        <section class="panel-section">
          <h2>Legend</h2>
          <div class="legend">
            <span class="swatch chemical">Chemical</span>
            <span class="swatch disease">Disease</span>
            <span>Edge label = relation cue/type</span>
          </div>
        </section>
        <section class="panel-section details">
          <h2>Selected</h2>
          <dl id="details">
            <dt>Item</dt><dd>Click a node or relation.</dd>
          </dl>
        </section>
      </aside>

      <section class="graph-shell">
        <svg id="graph" role="img" aria-label="Chemical disease knowledge graph"></svg>
        <div id="empty" class="empty">Paste an abstract or choose an example.</div>
      </section>

      <form id="controls" class="toolbar">
        <div>
          <label for="mode">Artifact filter</label>
          <select id="mode">
            <option value="all">All relations</option>
            <option value="pmid">PMID</option>
            <option value="chemical">Chemical ID/name</option>
            <option value="disease">Disease ID/name</option>
          </select>
        </div>
        <div>
          <label for="q">Value</label>
          <input id="q" placeholder="D004280, torsade, or 10087562">
        </div>
        <div>
          <label for="minConfidence">Min confidence</label>
          <input id="minConfidence" type="number" min="0" max="1" step="0.01" value="0.70">
        </div>
        <div>
          <label for="limit">Edges</label>
          <input id="limit" type="number" min="1" max="250" value="60">
        </div>
        <button id="run" type="submit">Render</button>
      </form>

      <section class="table-shell">
        <table>
          <thead>
            <tr>
              <th style="width: 90px;">Source</th>
              <th style="width: 120px;">Chemical</th>
              <th style="width: 120px;">Disease</th>
              <th>Names</th>
              <th style="width: 130px;">Relation</th>
              <th style="width: 110px;">Confidence</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody id="rows">
            <tr><td colspan="7" class="empty-row">No relations.</td></tr>
          </tbody>
        </table>
      </section>
    </section>

    <section id="casesView" class="view-section">
      <section class="case-library">
        <h2>Knowledge Graph Cases</h2>
        <p id="caseSummary" class="abstract-preview"></p>
        <div id="caseList" class="case-list"></div>
      </section>
    </section>
  </main>

  <script>
    const el = {
      tabAnalyze: document.querySelector("#tabAnalyze"),
      tabCases: document.querySelector("#tabCases"),
      analyzeView: document.querySelector("#analyzeView"),
      casesView: document.querySelector("#casesView"),
      form: document.querySelector("#controls"),
      abstractForm: document.querySelector("#abstractControls"),
      abstractText: document.querySelector("#abstractText"),
      examples: document.querySelector("#examples"),
      caseList: document.querySelector("#caseList"),
      caseSummary: document.querySelector("#caseSummary"),
      clearInput: document.querySelector("#clearInput"),
      appendSeparator: document.querySelector("#appendSeparator"),
      loadMixedCase: document.querySelector("#loadMixedCase"),
      mode: document.querySelector("#mode"),
      q: document.querySelector("#q"),
      minConfidence: document.querySelector("#minConfidence"),
      limit: document.querySelector("#limit"),
      run: document.querySelector("#run"),
      runAbstract: document.querySelector("#runAbstract"),
      svg: document.querySelector("#graph"),
      empty: document.querySelector("#empty"),
      rows: document.querySelector("#rows"),
      error: document.querySelector("#error"),
      status: document.querySelector("#status"),
      details: document.querySelector("#details"),
      mAbstracts: document.querySelector("#mAbstracts"),
      mChem: document.querySelector("#mChem"),
      mDisease: document.querySelector("#mDisease"),
      mEdges: document.querySelector("#mEdges"),
    };

    const exampleCases = {{ example_cases_json|safe }};
    const caseLibrary = {{ case_library_json|safe }};
    let simulationTimer = null;
    let graphData = {nodes: [], edges: []};

    function esc(value) {
      return String(value ?? "-").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function showError(message) {
      el.error.textContent = message;
      el.error.style.display = message ? "block" : "none";
    }

    async function api(path) {
      const response = await fetch(path);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    async function postApi(path, body) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    function updateDetails(title, pairs) {
      el.details.innerHTML = pairs.map(([key, value]) => (
        `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`
      )).join("");
      if (title) {
        el.details.insertAdjacentHTML("afterbegin", `<dt>Item</dt><dd><strong>${esc(title)}</strong></dd>`);
      }
    }

    function renderMetrics(meta) {
      el.mAbstracts.textContent = meta.abstract_count ?? 0;
      el.mChem.textContent = meta.chemical_nodes;
      el.mDisease.textContent = meta.disease_nodes;
      el.mEdges.textContent = meta.edge_count;
    }

    function abstractTextForCase(item) {
      return item.abstracts.join("\\n\\n====\\n\\n");
    }

    function showView(name) {
      const showCases = name === "cases";
      el.tabAnalyze.classList.toggle("active", !showCases);
      el.tabCases.classList.toggle("active", showCases);
      el.analyzeView.classList.toggle("active", !showCases);
      el.casesView.classList.toggle("active", showCases);
    }

    function loadCase(item, analyze = false) {
      el.abstractText.value = abstractTextForCase(item);
      showView("analyze");
      el.abstractText.focus();
      if (analyze) el.abstractForm.requestSubmit();
    }

    function renderExampleButtons() {
      el.examples.innerHTML = exampleCases.map((item, index) => `
        <button class="example-button" type="button" data-example="${index}">
          ${esc(item.title)}
          <span>${esc(item.abstracts.length)} abstract${item.abstracts.length > 1 ? "s" : ""}</span>
          <small>${esc((item.abstracts[0] || "").slice(0, 170))}</small>
        </button>
      `).join("");
      el.examples.querySelectorAll("[data-example]").forEach(button => {
        button.addEventListener("click", () => loadCase(exampleCases[Number(button.dataset.example)], false));
      });
    }

    function renderCaseLibrary() {
      el.caseList.innerHTML = exampleCases.map((item, index) => `
        <button class="case-button" type="button" data-case="${index}">
          ${esc(item.title)}
          <span>${esc(item.category)} · ${esc(item.abstracts.length)} abstracts</span>
        </button>
      `).join("");
      el.caseList.querySelectorAll("[data-case]").forEach(button => {
        button.addEventListener("click", () => loadCase(exampleCases[Number(button.dataset.case)], true));
      });
    }

    function renderCaseLibrary() {
      el.caseSummary.textContent = `${caseLibrary.length} cases loaded from trained graph artifacts. Click a card to analyze its abstract evidence.`;
      el.caseList.innerHTML = caseLibrary.map((item, index) => `
        <button class="case-button" type="button" data-case="${index}">
          ${esc(item.title)}
          <span>${esc(item.category)} - ${esc(item.edge_count || 0)} relation${item.edge_count === 1 ? "" : "s"}</span>
          <small>${esc(item.preview || (item.abstracts[0] || "").slice(0, 180))}</small>
        </button>
      `).join("");
      el.caseList.querySelectorAll("[data-case]").forEach(button => {
        button.addEventListener("click", () => loadCase(caseLibrary[Number(button.dataset.case)], true));
      });
    }

    function renderRows(edges) {
      if (!edges.length) {
        el.rows.innerHTML = '<tr><td colspan="7" class="empty-row">No relations.</td></tr>';
        return;
      }
      el.rows.innerHTML = edges.map(edge => `
        <tr>
          <td class="id">${esc(edge.pmid)}</td>
          <td class="id">${esc(edge.chemical_id)}</td>
          <td class="id">${esc(edge.disease_id)}</td>
          <td>${esc(edge.chemical_label)} -> ${esc(edge.disease_label)}</td>
          <td>${esc(edge.predicate || "CID")}</td>
          <td class="confidence">${Number(edge.confidence).toFixed(4)}</td>
          <td>${esc(edge.evidence)}</td>
        </tr>
      `).join("");
    }

    function resizeNode(node) {
      return Math.max(7, Math.min(22, 6 + Math.sqrt(node.degree || 1) * 3));
    }

    function layoutGraph(data) {
      if (simulationTimer) window.clearInterval(simulationTimer);
      const width = el.svg.clientWidth || 900;
      const height = el.svg.clientHeight || 640;
      const nodes = data.nodes.map((node, index) => ({
        ...node,
        x: width / 2 + Math.cos(index) * Math.min(width, height) * 0.25,
        y: height / 2 + Math.sin(index) * Math.min(width, height) * 0.25,
        vx: 0,
        vy: 0
      }));
      const byId = new Map(nodes.map(node => [node.id, node]));
      const edges = data.edges.map(edge => ({...edge, sourceNode: byId.get(edge.source), targetNode: byId.get(edge.target)}))
        .filter(edge => edge.sourceNode && edge.targetNode);

      function tick() {
        for (const node of nodes) {
          node.vx += (width / 2 - node.x) * 0.0009;
          node.vy += (height / 2 - node.y) * 0.0009;
        }
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i], b = nodes[j];
            let dx = a.x - b.x, dy = a.y - b.y;
            let dist2 = dx * dx + dy * dy || 1;
            const force = Math.min(2600 / dist2, 1.6);
            const dist = Math.sqrt(dist2);
            dx /= dist; dy /= dist;
            a.vx += dx * force; a.vy += dy * force;
            b.vx -= dx * force; b.vy -= dy * force;
          }
        }
        for (const edge of edges) {
          const a = edge.sourceNode, b = edge.targetNode;
          const dx = b.x - a.x, dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const target = 115;
          const force = (dist - target) * 0.012;
          const fx = dx / dist * force, fy = dy / dist * force;
          a.vx += fx; a.vy += fy;
          b.vx -= fx; b.vy -= fy;
        }
        for (const node of nodes) {
          node.vx *= 0.82;
          node.vy *= 0.82;
          node.x = Math.max(28, Math.min(width - 28, node.x + node.vx));
          node.y = Math.max(28, Math.min(height - 28, node.y + node.vy));
        }
        draw(nodes, edges);
      }
      for (let i = 0; i < 80; i++) tick();
      simulationTimer = window.setInterval(tick, 40);
    }

    function draw(nodes, edges) {
      const edgeSvg = edges.map(edge => {
        const width = 1 + Number(edge.confidence) * 3;
        const midX = (edge.sourceNode.x + edge.targetNode.x) / 2;
        const midY = (edge.sourceNode.y + edge.targetNode.y) / 2;
        const label = edge.predicate || "CID";
        const labelWidth = Math.min(132, Math.max(42, label.length * 7 + 16));
        return `
          <g data-edge-id="${esc(edge.id)}">
            <line class="edge" x1="${edge.sourceNode.x}" y1="${edge.sourceNode.y}" x2="${edge.targetNode.x}" y2="${edge.targetNode.y}" stroke-width="${width}"></line>
            <rect class="edge-label-bg" x="${midX - labelWidth / 2}" y="${midY - 10}" width="${labelWidth}" height="20"></rect>
            <text class="edge-label" x="${midX}" y="${midY + 4}" text-anchor="middle">${esc(label.length > 18 ? label.slice(0, 17) + "..." : label)}</text>
          </g>
        `;
      }).join("");
      const nodeSvg = nodes.map(node => {
        const r = resizeNode(node);
        const label = node.label.length > 24 ? node.label.slice(0, 23) + "..." : node.label;
        return `
          <g>
            <circle class="node ${esc(node.type)}" cx="${node.x}" cy="${node.y}" r="${r}" data-node-id="${esc(node.id)}"></circle>
            <text class="label" x="${node.x + r + 5}" y="${node.y + 4}">${esc(label)}</text>
          </g>
        `;
      }).join("");
      el.svg.innerHTML = `<g>${edgeSvg}</g><g>${nodeSvg}</g>`;
      el.svg.querySelectorAll("[data-node-id]").forEach(item => {
        item.addEventListener("click", () => {
          const node = graphData.nodes.find(row => row.id === item.dataset.nodeId);
          updateDetails(node.label, [
            ["Type", node.type],
            ["MeSH ID", node.mesh_id],
            ["Degree", node.degree],
            ["Mentions", node.mentions ? node.mentions.join(", ") : "-"],
            ["Examples", node.mention_examples || "-"]
          ]);
        });
      });
      el.svg.querySelectorAll("[data-edge-id]").forEach(item => {
        item.addEventListener("click", () => {
          const edge = graphData.edges.find(row => row.id === item.dataset.edgeId);
          updateDetails("CID relation", [
            ["PMID", edge.pmid],
            ["Chemical", `${edge.chemical_label} (${edge.chemical_id})`],
            ["Disease", `${edge.disease_label} (${edge.disease_id})`],
            ["Predicate", edge.predicate || "CID"],
            ["Cue", edge.relation_cue || "-"],
            ["Confidence", Number(edge.confidence).toFixed(4)],
            ["Source", edge.relation_source || "-"],
            ["Evidence", edge.evidence || "-"]
          ]);
        });
      });
    }

    async function render(event) {
      event?.preventDefault();
      showError("");
      el.run.disabled = true;
      try {
        const params = new URLSearchParams({
          mode: el.mode.value,
          q: el.q.value.trim(),
          min_confidence: el.minConfidence.value,
          limit: el.limit.value
        });
        const data = await api(`/api/graph?${params.toString()}`);
        graphData = data;
        el.analyzeView.classList.add("has-results");
        renderMetrics(data.meta);
        renderRows(data.edges);
        el.empty.style.display = data.nodes.length ? "none" : "grid";
        if (data.nodes.length) layoutGraph(data);
        updateDetails("", [["Item", "Click a node or relation."]]);
        el.status.textContent = `${data.meta.total_edges_available} relations available`;
      } catch (error) {
        showError(error.message);
      } finally {
        el.run.disabled = false;
      }
    }

    async function renderAbstract(event) {
      event.preventDefault();
      showError("");
      el.runAbstract.disabled = true;
      try {
        const data = await postApi("/api/abstract-graph", {abstract: el.abstractText.value});
        graphData = data;
        el.analyzeView.classList.add("has-results");
        renderMetrics(data.meta);
        renderRows(data.edges);
        el.empty.style.display = data.nodes.length ? "none" : "grid";
        if (data.nodes.length) layoutGraph(data);
        updateDetails("Abstract", [
          ["Abstracts", data.meta.abstract_count ?? 1],
          ["Chemicals", data.meta.chemical_mentions],
          ["Diseases", data.meta.disease_mentions],
          ["Relations", data.meta.edge_count],
          ["Text", data.meta.abstract_preview || "-"]
        ]);
        el.status.textContent = `${data.meta.edge_count} relations from ${data.meta.abstract_count ?? 1} abstract(s)`;
      } catch (error) {
        showError(error.message);
      } finally {
        el.runAbstract.disabled = false;
      }
    }

    el.mode.addEventListener("change", () => {
      el.q.disabled = el.mode.value === "all";
      if (el.mode.value === "all") el.q.value = "";
    });
    el.form.addEventListener("submit", render);
    el.abstractForm.addEventListener("submit", renderAbstract);
    el.tabAnalyze.addEventListener("click", () => showView("analyze"));
    el.tabCases.addEventListener("click", () => showView("cases"));
    el.clearInput.addEventListener("click", () => {
      el.abstractText.value = "";
      el.analyzeView.classList.remove("has-results");
      el.abstractText.focus();
    });
    el.appendSeparator.addEventListener("click", () => {
      const current = el.abstractText.value.trim();
      el.abstractText.value = current ? `${current}\\n\\n====\\n\\n` : "====\\n\\n";
      el.abstractText.focus();
    });
    el.loadMixedCase.addEventListener("click", () => loadCase(exampleCases.find(item => item.category === "mixed") || exampleCases[0], false));
    window.addEventListener("resize", () => graphData.nodes.length && layoutGraph(graphData));
    renderExampleButtons();
    renderCaseLibrary();
    el.status.textContent = "Ready for abstract input";
    el.abstractText.focus();
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class GraphData:
    chemicals: dict[str, dict[str, str]]
    diseases: dict[str, dict[str, str]]
    edges: list[dict[str, Any]]


@dataclass(frozen=True)
class MatchedEntity:
    node_id: str
    mesh_id: str
    label: str
    entity_type: str
    start: int
    end: int
    mention: str


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def term_values(row: dict[str, str]) -> list[str]:
    values = []
    for field in ["label", "mention_examples", "mesh_id"]:
        for part in (row.get(field) or "").split("|"):
            term = normalize_space(part)
            if len(term) >= 3 and term.lower() not in {item.lower() for item in values}:
                values.append(term)
    return values


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.IGNORECASE)


def find_entities(text: str, rows: dict[str, dict[str, str]], entity_type: str) -> list[MatchedEntity]:
    matches: list[MatchedEntity] = []
    occupied: list[tuple[int, int, str]] = []
    sorted_rows = sorted(
        rows.values(),
        key=lambda row: max((len(term) for term in term_values(row)), default=0),
        reverse=True,
    )
    for row in sorted_rows:
        mesh_id = row["mesh_id"]
        label = row.get("label") or mesh_id
        for term in sorted(term_values(row), key=len, reverse=True):
            for match in term_pattern(term).finditer(text):
                start, end = match.span()
                overlap = any(
                    start < old_end and end > old_start and entity_type == old_type
                    for old_start, old_end, old_type in occupied
                )
                if overlap:
                    continue
                matches.append(
                    MatchedEntity(
                        node_id=f"{entity_type.lower()}:{mesh_id}",
                        mesh_id=mesh_id,
                        label=label,
                        entity_type=entity_type.lower(),
                        start=start,
                        end=end,
                        mention=text[start:end],
                    )
                )
                occupied.append((start, end, entity_type))
    return sorted(matches, key=lambda item: (item.start, item.end, item.entity_type, item.mesh_id))


def snippet_for_span_pair(text: str, first: MatchedEntity, second: MatchedEntity, radius: int = 180) -> str:
    start = max(0, min(first.start, second.start) - radius)
    end = min(len(text), max(first.end, second.end) + radius)
    return normalize_space(text[start:end])


def sentence_window(text: str, first: MatchedEntity, second: MatchedEntity) -> str:
    start = min(first.start, second.start)
    end = max(first.end, second.end)
    sentence_start = max(text.rfind(".", 0, start), text.rfind("?", 0, start), text.rfind("!", 0, start))
    sentence_end_candidates = [
        position for position in [text.find(".", end), text.find("?", end), text.find("!", end)]
        if position != -1
    ]
    sentence_end = min(sentence_end_candidates) + 1 if sentence_end_candidates else len(text)
    return normalize_space(text[sentence_start + 1 : sentence_end])


def relation_phrase(text: str, chemical: MatchedEntity, disease: MatchedEntity, known: dict[str, Any] | None) -> tuple[str, str]:
    sentence = sentence_window(text, chemical, disease)
    between_start = min(chemical.end, disease.end)
    between_end = max(chemical.start, disease.start)
    between = text[between_start:between_end] if between_start <= between_end else ""
    for region in [between, sentence]:
        region_lower = region.lower()
        center = len(region_lower) / 2
        cue_matches: list[tuple[float, str, str]] = []
        for cue, predicate in RELATION_CUES:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(cue)}(?![A-Za-z0-9])"
            for match in re.finditer(pattern, region_lower):
                cue_center = (match.start() + match.end()) / 2
                cue_matches.append((abs(cue_center - center), predicate, cue))
        if cue_matches:
            _, predicate, cue = min(cue_matches, key=lambda item: item[0])
            return predicate, cue
    if known:
        return "CID", ""
    return "co-occurs with", ""


def best_known_edge(data: GraphData, chemical_id: str, disease_id: str) -> dict[str, Any] | None:
    candidates = [
        edge for edge in data.edges
        if edge["chemical_id"] == chemical_id and edge["disease_id"] == disease_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence") or 0))


def cooccurrence_score(chemical: MatchedEntity, disease: MatchedEntity) -> float:
    distance = max(0, max(chemical.start, disease.start) - min(chemical.end, disease.end))
    return max(0.05, min(0.95, 1.0 / (1.0 + distance / 260.0)))


def split_abstracts(value: str) -> list[str]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?:^|\n)\s*(?:={4,}|---+)\s*(?:\n|$)|\n\s*\n+", normalized)
    abstracts = [normalize_space(part) for part in parts if normalize_space(part)]
    if not abstracts and normalize_space(value):
        abstracts = [normalize_space(value)]
    if len(abstracts) > MAX_ABSTRACTS_PER_REQUEST:
        raise ValueError(f"Too many abstracts. Maximum is {MAX_ABSTRACTS_PER_REQUEST} per request.")
    return abstracts


def abstract_graph(data: GraphData, abstract: str) -> dict[str, Any]:
    text = normalize_space(abstract)
    if not text:
        raise ValueError("Abstract is required.")
    if len(text) > MAX_ABSTRACT_CHARS:
        raise ValueError(f"Abstract is too long. Maximum length is {MAX_ABSTRACT_CHARS} characters.")

    chemicals = find_entities(text, data.chemicals, "Chemical")
    diseases = find_entities(text, data.diseases, "Disease")
    chemicals_by_id: dict[str, list[MatchedEntity]] = {}
    diseases_by_id: dict[str, list[MatchedEntity]] = {}
    for entity in chemicals:
        chemicals_by_id.setdefault(entity.mesh_id, []).append(entity)
    for entity in diseases:
        diseases_by_id.setdefault(entity.mesh_id, []).append(entity)

    edges: list[dict[str, Any]] = []
    edge_index = 0
    for chemical_id, chemical_mentions in sorted(chemicals_by_id.items()):
        for disease_id, disease_mentions in sorted(diseases_by_id.items()):
            chemical = min(chemical_mentions, key=lambda item: item.start)
            disease = min(disease_mentions, key=lambda item: item.start)
            known = best_known_edge(data, chemical_id, disease_id)
            confidence = float(known["confidence"]) if known else cooccurrence_score(chemical, disease)
            source = "known_cid" if known else "abstract_cooccurrence"
            predicate, cue = relation_phrase(text, chemical, disease, known)
            edge_index += 1
            edges.append(
                {
                    "id": f"abstract-e{edge_index}",
                    "source": f"chemical:{chemical_id}",
                    "target": f"disease:{disease_id}",
                    "chemical_id": chemical_id,
                    "disease_id": disease_id,
                    "chemical_label": chemical.label,
                    "disease_label": disease.label,
                    "pmid": "abstract",
                    "confidence": confidence,
                    "predicate": predicate,
                    "relation_cue": cue,
                    "relation_source": source,
                    "evidence": snippet_for_span_pair(text, chemical, disease),
                    "chemical_mentions": ", ".join(sorted({item.mention for item in chemical_mentions})),
                    "disease_mentions": ", ".join(sorted({item.mention for item in disease_mentions})),
                    "known_pmid": known.get("pmid", "") if known else "",
                }
            )

    edges.sort(key=lambda item: (-float(item["confidence"]), item["chemical_label"], item["disease_label"]))
    response = build_response(data, edges)
    mention_map = {
        f"chemical:{mesh_id}": sorted({item.mention for item in mentions})
        for mesh_id, mentions in chemicals_by_id.items()
    }
    mention_map.update(
        {
            f"disease:{mesh_id}": sorted({item.mention for item in mentions})
            for mesh_id, mentions in diseases_by_id.items()
        }
    )
    present_node_ids = {node["id"] for node in response["nodes"]}
    for node_id, mentions in sorted(mention_map.items()):
        if node_id in present_node_ids:
            continue
        kind, mesh_id = node_id.split(":", 1)
        source_row = data.chemicals[mesh_id] if kind == "chemical" else data.diseases[mesh_id]
        response["nodes"].append(
            {
                "id": node_id,
                "type": kind,
                "mesh_id": mesh_id,
                "label": source_row.get("label") or mesh_id,
                "mention_examples": source_row.get("mention_examples", ""),
                "degree": 0,
                "mentions": mentions,
            }
        )
    for node in response["nodes"]:
        node["mentions"] = mention_map.get(node["id"], [])
    response["meta"]["chemical_nodes"] = sum(1 for node in response["nodes"] if node["type"] == "chemical")
    response["meta"]["disease_nodes"] = sum(1 for node in response["nodes"] if node["type"] == "disease")
    response["meta"].update(
        {
            "abstract_preview": text[:500],
            "chemical_mentions": len(chemicals),
            "disease_mentions": len(diseases),
            "source": "abstract",
        }
    )
    return response


def multi_abstract_graph(data: GraphData, raw_text: str) -> dict[str, Any]:
    abstracts = split_abstracts(raw_text)
    if not abstracts:
        raise ValueError("Abstract is required.")

    responses = []
    all_edges: list[dict[str, Any]] = []
    mention_map: dict[str, set[str]] = {}
    isolated_nodes: dict[str, dict[str, Any]] = {}
    chemical_mentions = 0
    disease_mentions = 0

    for index, abstract in enumerate(abstracts, start=1):
        response = abstract_graph(data, abstract)
        source_label = f"A{index}"
        responses.append(response)
        chemical_mentions += int(response["meta"].get("chemical_mentions", 0))
        disease_mentions += int(response["meta"].get("disease_mentions", 0))

        for node in response["nodes"]:
            node_id = node["id"]
            mention_map.setdefault(node_id, set()).update(node.get("mentions") or [])
            isolated_nodes[node_id] = node

        for edge in response["edges"]:
            all_edges.append(
                {
                    **edge,
                    "id": f"{source_label}-{edge['id']}",
                    "pmid": source_label,
                    "source_abstract": source_label,
                    "abstract_preview": abstract[:220],
                }
            )

    combined = build_response(data, all_edges)
    present_node_ids = {node["id"] for node in combined["nodes"]}
    for node_id, node in sorted(isolated_nodes.items()):
        if node_id not in present_node_ids:
            combined["nodes"].append({**node, "degree": 0})

    for node in combined["nodes"]:
        node["mentions"] = sorted(mention_map.get(node["id"], set()))

    combined["meta"].update(
        {
            "abstract_count": len(abstracts),
            "abstract_preview": " | ".join(abstract[:180] for abstract in abstracts[:3]),
            "chemical_mentions": chemical_mentions,
            "disease_mentions": disease_mentions,
            "chemical_nodes": sum(1 for node in combined["nodes"] if node["type"] == "chemical"),
            "disease_nodes": sum(1 for node in combined["nodes"] if node["type"] == "disease"),
            "source": "multi_abstract",
        }
    )
    return combined


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing graph artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_graph_data(graph_dir: Path) -> GraphData:
    chemicals = {row["mesh_id"]: row for row in read_csv(graph_dir / "chemical_nodes.csv")}
    diseases = {row["mesh_id"]: row for row in read_csv(graph_dir / "disease_nodes.csv")}
    edges: list[dict[str, Any]] = []
    for index, row in enumerate(read_csv(graph_dir / "cid_edges.csv")):
        chemical = chemicals.get(row["chemical_id"], {})
        disease = diseases.get(row["disease_id"], {})
        confidence = float(row.get("confidence") or 0)
        edges.append(
            {
                "id": f"e{index}",
                "source": f"chemical:{row['chemical_id']}",
                "target": f"disease:{row['disease_id']}",
                "chemical_id": row["chemical_id"],
                "disease_id": row["disease_id"],
                "chemical_label": chemical.get("label") or row["chemical_id"],
                "disease_label": disease.get("label") or row["disease_id"],
                "pmid": row.get("pmid", ""),
                "confidence": confidence,
                "predicate": "CID",
                "relation_cue": "",
                "relation_source": row.get("source", ""),
                "evidence": row.get("evidence", ""),
            }
        )
    return GraphData(chemicals=chemicals, diseases=diseases, edges=edges)


def shorten(value: str, limit: int = 72) -> str:
    text = normalize_space(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def graph_case_library(data: GraphData) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edge in data.edges:
        pmid = str(edge.get("pmid") or "unknown")
        grouped.setdefault(pmid, []).append(edge)

    cases: list[dict[str, Any]] = []
    for pmid, edges in grouped.items():
        edges = sorted(edges, key=lambda item: (-float(item.get("confidence") or 0), item["chemical_label"], item["disease_label"]))
        chemicals = sorted({edge["chemical_label"] for edge in edges})
        diseases = sorted({edge["disease_label"] for edge in edges})
        abstracts = []
        seen_evidence = set()
        for edge in edges:
            evidence = normalize_space(str(edge.get("evidence") or ""))
            if evidence and evidence not in seen_evidence:
                abstracts.append(evidence)
                seen_evidence.add(evidence)
        if not abstracts:
            abstracts = [
                f"{edge['chemical_label']} CID {edge['disease_label']}"
                for edge in edges[:5]
            ]
        category = "single relation"
        if len(edges) > 1 and (len(chemicals) == 1 or len(diseases) == 1):
            category = "connected"
        elif len(edges) > 1:
            category = "multi-relation"
        title_entity = shorten(chemicals[0] if chemicals else "Chemical", 34)
        title_target = shorten(diseases[0] if diseases else "Disease", 34)
        cases.append(
            {
                "id": f"pmid-{pmid}",
                "title": f"PMID {pmid}: {title_entity} -> {title_target}",
                "category": category,
                "edge_count": len(edges),
                "chemical_count": len(chemicals),
                "disease_count": len(diseases),
                "preview": abstracts[0][:220],
                "abstracts": abstracts[:3],
            }
        )
    cases.sort(key=lambda item: (-int(item["edge_count"]), item["title"]))
    return cases


def matches_text(value: str, *candidates: str) -> bool:
    needle = value.strip().lower()
    return not needle or any(needle in candidate.lower() for candidate in candidates if candidate)


def filter_edges(data: GraphData, mode: str, query: str, min_confidence: float, limit: int) -> list[dict[str, Any]]:
    rows = [edge for edge in data.edges if edge["confidence"] >= min_confidence]
    if mode == "pmid":
        rows = [edge for edge in rows if edge["pmid"] == query.strip()]
    elif mode == "chemical":
        rows = [
            edge for edge in rows
            if matches_text(query, edge["chemical_id"], edge["chemical_label"])
        ]
    elif mode == "disease":
        rows = [
            edge for edge in rows
            if matches_text(query, edge["disease_id"], edge["disease_label"])
        ]
    elif mode != "all":
        raise ValueError("Unsupported filter mode.")
    rows.sort(key=lambda item: (-item["confidence"], item["pmid"], item["chemical_id"], item["disease_id"]))
    return rows[: max(1, min(limit, 250))]


def build_response(data: GraphData, edges: list[dict[str, Any]]) -> dict[str, Any]:
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    nodes = []
    for node_id, count in sorted(degree.items()):
        kind, mesh_id = node_id.split(":", 1)
        source = data.chemicals[mesh_id] if kind == "chemical" else data.diseases[mesh_id]
        nodes.append(
            {
                "id": node_id,
                "type": kind,
                "mesh_id": mesh_id,
                "label": source.get("label") or mesh_id,
                "mention_examples": source.get("mention_examples", ""),
                "degree": count,
            }
        )

    avg_confidence = sum(edge["confidence"] for edge in edges) / len(edges) if edges else 0
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "chemical_nodes": sum(1 for node in nodes if node["type"] == "chemical"),
            "disease_nodes": sum(1 for node in nodes if node["type"] == "disease"),
            "edge_count": len(edges),
            "avg_confidence": avg_confidence,
            "total_edges_available": len(data.edges),
        },
    }


def create_app(graph_dir: Path = DEFAULT_GRAPH_DIR) -> Flask:
    app = Flask(__name__)
    data = load_graph_data(graph_dir)

    @app.get("/")
    def index() -> str:
        return render_template_string(
            INDEX_HTML,
            example_cases_json=json.dumps(EXAMPLE_CASES, ensure_ascii=False),
            case_library_json=json.dumps(graph_case_library(data), ensure_ascii=False),
        )

    @app.get("/api/health")
    def health() -> Any:
        return jsonify(
            {
                "chemical_nodes": len(data.chemicals),
                "disease_nodes": len(data.diseases),
                "cid_relationships": len(data.edges),
                "graph_dir": str(graph_dir),
            }
        )

    @app.get("/api/graph")
    def graph() -> Any:
        try:
            mode = request.args.get("mode", "all")
            query = request.args.get("q", "")
            min_confidence = float(request.args.get("min_confidence", "0.70"))
            limit = int(request.args.get("limit", "60"))
            if mode != "all" and not query.strip():
                return jsonify({"error": "Filter value is required."}), 400
            edges = filter_edges(data, mode, query, min_confidence, limit)
            return jsonify(build_response(data, edges))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/abstract-graph")
    def graph_from_abstract() -> Any:
        try:
            payload = request.get_json(silent=True) or {}
            abstract = str(payload.get("abstract") or "")
            return jsonify(multi_abstract_graph(data, abstract))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ENEXRE Flask knowledge graph viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app(args.graph_dir)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
