# Copyright 2026 The Information Extraction Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Editable visualization for entity evidence spans in grounded JSON outputs."""

from __future__ import annotations

import json
import pathlib
import textwrap

# Fallback if IPython is not present.
try:
    from IPython import get_ipython  # type: ignore[import-not-found]
    from IPython.display import HTML  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover

    def get_ipython():  # type: ignore[no-redef]
        return None

    HTML = None  # pytype: disable=annotation-type-mismatch


def _is_jupyter() -> bool:
    """Check if we're in a Jupyter/IPython environment that can display HTML."""
    try:
        if get_ipython is None:
            return False
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def _default_review_filename(json_path: pathlib.Path) -> str:
    name = json_path.name
    if name.endswith(".json"):
        return name[:-5] + ".review.json"
    return name + ".review.json"


def visualize_entity_review(
    markdown_path: str | pathlib.Path,
    entity_json_path: str | pathlib.Path,
    *,
    output_filename: str | None = None,
    show_file_picker: bool = True,
) -> HTML | str:
    """Visualize grounded entity JSON with editable values + evidence spans."""
    md_path = pathlib.Path(markdown_path)
    json_path = pathlib.Path(entity_json_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Entity JSON not found: {json_path}")

    markdown_text = md_path.read_text(encoding="utf-8")
    record = json.loads(json_path.read_text(encoding="utf-8"))

    output_name = output_filename or _default_review_filename(json_path)
    doc_name = json_path.stem
    md_name = md_path.name
    json_name = json_path.name

    text_json = json.dumps(markdown_text, ensure_ascii=True)
    record_json = json.dumps(record, ensure_ascii=True)
    output_name_json = json.dumps(output_name)
    doc_name_json = json.dumps(doc_name)
    md_name_json = json.dumps(md_name)
    json_name_json = json.dumps(json_name)
    show_picker_json = "true" if show_file_picker else "false"

    html_content = textwrap.dedent(
        f"""
        <div class="ev-app" id="entityEvidenceApp">
          <header class="ev-topbar">
            <div class="ev-brand">
              <div class="ev-title">Evidence Studio</div>
              <div class="ev-subtitle">Edit extracted values and evidence spans</div>
            </div>
            <div class="ev-doc-meta">
              <div class="ev-doc-name" id="docName">Document</div>
              <div class="ev-doc-stats" id="docStats"></div>
            </div>
            <div class="ev-top-actions">
              <div class="ev-file-row" id="fileRow">
                <label class="ev-file-btn">
                  Load JSON
                  <input type="file" id="jsonFile" accept=".json" />
                </label>
                <label class="ev-file-btn">
                  Load Markdown
                  <input type="file" id="mdFile" accept=".md,.txt" />
                </label>
              </div>
              <button class="ev-download-btn" id="downloadBtn">Download review JSON</button>
            </div>
          </header>

          <main class="ev-layout">
            <aside class="ev-panel ev-left">
              <div class="ev-panel-header">
                <span>Fields</span>
                <span class="ev-pill" id="fieldCount">0</span>
              </div>
              <input class="ev-search" id="searchInput" type="text"
                     placeholder="Filter by field or value" />
              <div class="ev-chip-row" id="chipRow"></div>
              <div class="ev-field-list" id="fieldList"></div>
            </aside>

            <section class="ev-panel ev-center">
              <div class="ev-panel-header">Document</div>
              <div class="ev-legend" id="legend"></div>
              <div class="ev-text-window" id="textWindow"></div>
              <div class="ev-selection-row">
                <div class="ev-selection-info" id="selectionInfo">Select text to add evidence.</div>
                <button class="ev-action-btn" id="addEvidenceBtn">Add evidence from selection</button>
              </div>
            </section>

            <aside class="ev-panel ev-right">
              <div class="ev-panel-header">Field detail</div>
              <div class="ev-detail" id="fieldDetail"></div>
              <div class="ev-status" id="statusMsg"></div>
            </aside>
          </main>
        </div>

        <style>
          @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Spectral:wght@400;600&display=swap");
          :root {{
            --bg: #f2efe9;
            --panel: #fffdf8;
            --ink: #1b1a17;
            --muted: #5c5c5c;
            --accent: #1d4a3a;
            --accent-2: #d36b3b;
            --border: #e4ddd3;
            --shadow: 0 18px 40px rgba(25, 20, 10, 0.12);
            --study: #f6d3b2;
            --task: #cde9d5;
            --modality: #cfe2f7;
            --analysis: #f0d2e8;
            --group: #f9e3a6;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            background: radial-gradient(circle at top, #fef7ef 0%, var(--bg) 55%, #efe7dd 100%);
            color: var(--ink);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
          }}
          .ev-app {{
            padding: 20px 22px 30px;
            min-height: 100vh;
          }}
          .ev-topbar {{
            display: grid;
            grid-template-columns: 1.2fr 1fr 1.3fr;
            gap: 18px;
            align-items: center;
            padding: 16px 18px;
            background: var(--panel);
            border-radius: 18px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
          }}
          .ev-title {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: 0.4px;
          }}
          .ev-subtitle {{
            font-size: 12px;
            color: var(--muted);
            margin-top: 4px;
          }}
          .ev-doc-meta {{
            text-align: center;
            font-size: 12px;
          }}
          .ev-doc-name {{
            font-weight: 600;
          }}
          .ev-doc-stats {{
            color: var(--muted);
            margin-top: 4px;
          }}
          .ev-top-actions {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: flex-end;
          }}
          .ev-file-row {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
          }}
          .ev-file-btn {{
            position: relative;
            overflow: hidden;
            border-radius: 999px;
            background: #f0ebe4;
            border: 1px solid var(--border);
            padding: 8px 14px;
            font-size: 12px;
            cursor: pointer;
          }}
          .ev-file-btn input {{
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
          }}
          .ev-download-btn {{
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 999px;
            padding: 10px 16px;
            font-size: 12px;
            cursor: pointer;
            box-shadow: 0 8px 18px rgba(29, 74, 58, 0.2);
          }}
          .ev-layout {{
            margin-top: 18px;
            display: grid;
            grid-template-columns: 280px 1fr 340px;
            gap: 18px;
          }}
          .ev-panel {{
            background: var(--panel);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            min-height: 72vh;
          }}
          .ev-panel-header {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
          }}
          .ev-pill {{
            background: #efe6db;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 11px;
          }}
          .ev-search {{
            margin: 12px 16px 0;
            border-radius: 10px;
            border: 1px solid var(--border);
            padding: 8px 10px;
            font-size: 12px;
          }}
          .ev-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 12px 16px 6px;
          }}
          .ev-chip {{
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 11px;
            border: 1px solid var(--border);
            cursor: pointer;
            background: #fbf7f2;
          }}
          .ev-chip.active {{
            border-color: var(--accent);
            box-shadow: inset 0 0 0 1px var(--accent);
          }}
          .ev-field-list {{
            padding: 8px 10px 16px;
            overflow-y: auto;
            flex: 1;
          }}
          .ev-field-item {{
            border-radius: 12px;
            padding: 10px 12px;
            margin: 8px 6px;
            border: 1px solid transparent;
            cursor: pointer;
            background: #fbf8f3;
          }}
          .ev-field-item.active {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(29, 74, 58, 0.1);
          }}
          .ev-field-title {{
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 4px;
          }}
          .ev-field-meta {{
            font-size: 11px;
            color: var(--muted);
          }}
          .ev-field-preview {{
            margin-top: 6px;
            font-family: "Spectral", serif;
            font-size: 12px;
            color: #2d2a27;
          }}
          .ev-center {{
            padding-bottom: 16px;
          }}
          .ev-legend {{
            padding: 10px 16px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            font-size: 11px;
            color: var(--muted);
          }}
          .ev-legend span {{
            border-radius: 999px;
            padding: 4px 10px;
          }}
          .ev-text-window {{
            margin: 12px 16px 0;
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: #fff;
            max-height: 62vh;
            overflow-y: auto;
            font-family: "Spectral", serif;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
          }}
          .ev-span {{
            border-radius: 6px;
            padding: 1px 2px;
            transition: box-shadow 0.15s ease;
          }}
          .ev-span[data-entity="Study"] {{ background: var(--study); }}
          .ev-span[data-entity="Task"] {{ background: var(--task); }}
          .ev-span[data-entity="Modality"] {{ background: var(--modality); }}
          .ev-span[data-entity="Analysis"] {{ background: var(--analysis); }}
          .ev-span[data-entity="Group"] {{ background: var(--group); }}
          .ev-span.ev-current {{
            box-shadow: 0 0 0 2px var(--accent-2);
          }}
          .ev-selection-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 16px 0;
          }}
          .ev-selection-info {{
            font-size: 11px;
            color: var(--muted);
          }}
          .ev-action-btn {{
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 12px;
            border: 1px solid var(--border);
            background: #f7f0e7;
            cursor: pointer;
          }}
          .ev-detail {{
            padding: 10px 16px 0;
            flex: 1;
            overflow-y: auto;
          }}
          .ev-detail h3 {{
            margin: 12px 0 8px;
            font-size: 13px;
          }}
          .ev-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 999px;
            background: #f1e9df;
            margin-right: 6px;
          }}
          .ev-input {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 8px 10px;
            font-size: 12px;
            margin-top: 6px;
          }}
          .ev-textarea {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 8px 10px;
            font-size: 12px;
            min-height: 80px;
            margin-top: 6px;
            font-family: "Spectral", serif;
          }}
          .ev-btn-row {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 8px;
          }}
          .ev-btn {{
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 11px;
            border: 1px solid var(--border);
            cursor: pointer;
            background: #f7f1e8;
          }}
          .ev-btn.primary {{
            background: var(--accent);
            color: #fff;
            border: none;
          }}
          .ev-btn.danger {{
            background: #fbe2dd;
            border-color: #f2b7ab;
          }}
          .ev-evidence-card {{
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 10px 12px;
            margin-top: 10px;
            background: #fffaf4;
          }}
          .ev-evidence-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 600;
          }}
          .ev-evidence-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-top: 8px;
          }}
          .ev-status {{
            min-height: 22px;
            padding: 10px 16px 14px;
            font-size: 12px;
            color: #9b3b2c;
          }}
          @media (max-width: 1200px) {{
            .ev-layout {{
              grid-template-columns: 1fr;
            }}
            .ev-topbar {{
              grid-template-columns: 1fr;
              text-align: left;
            }}
            .ev-top-actions {{
              align-items: flex-start;
            }}
          }}
        </style>

        <script>
          (function() {{
            const initialText = {text_json};
            const initialRecord = {record_json};
            const initialDocName = {doc_name_json};
            const initialMdName = {md_name_json};
            const initialJsonName = {json_name_json};
            const defaultOutputName = {output_name_json};
            const showFilePicker = {show_picker_json};

            const state = {{
              text: initialText,
              record: initialRecord,
              docName: initialDocName,
              mdName: initialMdName,
              jsonName: initialJsonName,
              outputName: defaultOutputName,
            }};

            const entityTypes = ["Study", "Task", "Modality", "Analysis", "Group"];
            const entityMap = {{
              study: "Study",
              tasks: "Task",
              modalities: "Modality",
              analyses: "Analysis",
              demographics: "Group",
            }};

            const filters = {{}};
            entityTypes.forEach((name) => {{ filters[name] = true; }});

            const fieldListEl = document.getElementById("fieldList");
            const fieldCountEl = document.getElementById("fieldCount");
            const legendEl = document.getElementById("legend");
            const textWindowEl = document.getElementById("textWindow");
            const detailEl = document.getElementById("fieldDetail");
            const statusEl = document.getElementById("statusMsg");
            const selectionInfoEl = document.getElementById("selectionInfo");
            const searchInputEl = document.getElementById("searchInput");
            const chipRowEl = document.getElementById("chipRow");
            const addEvidenceBtn = document.getElementById("addEvidenceBtn");
            const downloadBtn = document.getElementById("downloadBtn");
            const docNameEl = document.getElementById("docName");
            const docStatsEl = document.getElementById("docStats");
            const fileRowEl = document.getElementById("fileRow");
            const jsonFileEl = document.getElementById("jsonFile");
            const mdFileEl = document.getElementById("mdFile");

            let fields = [];
            let currentFieldId = null;
            let sections = [];

            if (!showFilePicker && fileRowEl) {{
              fileRowEl.style.display = "none";
            }}

            function escapeHtml(value) {{
              return String(value)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
            }}

            function formatPath(path) {{
              let out = "";
              path.forEach((part) => {{
                if (typeof part === "number") {{
                  out += "[" + part + "]";
                }} else {{
                  if (out) out += ".";
                  out += part;
                }}
              }});
              return out;
            }}

            function getNodeByPath(path) {{
              let node = state.record;
              for (const part of path) {{
                if (node == null) return null;
                node = node[part];
              }}
              return node;
            }}

            function parseValue(raw) {{
              const trimmed = raw.trim();
              if (!trimmed) return null;
              if (trimmed.startsWith("{{") || trimmed.startsWith("[")) {{
                return JSON.parse(trimmed);
              }}
              if (/^-?\\d+(\\.\\d+)?$/.test(trimmed)) {{
                return Number(trimmed);
              }}
              if (trimmed === "true" || trimmed === "false") {{
                return trimmed === "true";
              }}
              return trimmed;
            }}

            function formatValue(value) {{
              if (value == null) return "";
              if (typeof value === "string") return value;
              return JSON.stringify(value, null, 2);
            }}

            function buildSections(text) {{
              const lines = text.split("\\n");
              const output = [];
              let cursor = 0;
              let current = "Document";
              let start = 0;
              lines.forEach((line) => {{
                const match = line.match(/^(#{{1,6}})\\s+(.*)$/);
                if (match) {{
                  output.push({{ title: current, start: start, end: cursor }});
                  current = match[2].trim() || "Untitled";
                  start = cursor;
                }}
                cursor += line.length + 1;
              }});
              output.push({{ title: current, start: start, end: text.length }});
              return output;
            }}

            function findSectionTitle(offset) {{
              for (const section of sections) {{
                if (offset >= section.start && offset <= section.end) {{
                  return section.title;
                }}
              }}
              return "Document";
            }}

            function buildFields() {{
              const items = [];
              let counter = 0;

              function walk(node, path, entityType, entityId, entityIndex) {{
                if (Array.isArray(node)) {{
                  node.forEach((child, idx) => {{
                    walk(child, path.concat(idx), entityType, entityId, idx);
                  }});
                  return;
                }}
                if (!node || typeof node !== "object") {{
                  return;
                }}

                let nextEntityId = entityId;
                let nextEntityIndex = entityIndex;
                if (entityType === "Task" && path.length === 2 && node.id) {{
                  nextEntityId = node.id;
                  nextEntityIndex = path[1];
                }} else if (entityType === "Modality" && path.length === 2 && node.id) {{
                  nextEntityId = node.id;
                  nextEntityIndex = path[1];
                }} else if (entityType === "Analysis" && path.length === 2 && node.id) {{
                  nextEntityId = node.id;
                  nextEntityIndex = path[1];
                }} else if (
                  entityType === "Group" &&
                  path.length === 3 &&
                  path[1] === "groups" &&
                  node.id
                ) {{
                  nextEntityId = node.id;
                  nextEntityIndex = path[2];
                }}

                if (Object.prototype.hasOwnProperty.call(node, "value") &&
                    Object.prototype.hasOwnProperty.call(node, "evidence")) {{
                  const fieldKey = path[path.length - 1];
                  items.push({{
                    id: counter++,
                    path: path,
                    entityType: entityType,
                    entityId: nextEntityId,
                    entityIndex: nextEntityIndex,
                    fieldKey: fieldKey,
                    label: fieldKey,
                  }});
                }}

                Object.keys(node).forEach((key) => {{
                  if (key === "value" || key === "evidence") return;
                  walk(node[key], path.concat(key), entityType, nextEntityId, nextEntityIndex);
                }});
              }}

              Object.keys(state.record || {{}}).forEach((key) => {{
                if (!(key in entityMap)) return;
                const entityType = entityMap[key];
                walk(state.record[key], [key], entityType, null, null);
              }});

              fields = items;
              if (fields.length && currentFieldId == null) {{
                currentFieldId = fields[0].id;
              }}
              fieldCountEl.textContent = String(fields.length);
            }}

            function buildLegend() {{
              legendEl.innerHTML = entityTypes.map((name) => {{
                return '<span style="background:' + getEntityColor(name) + ';">' +
                  name + '</span>';
              }}).join("");
            }}

            function getEntityColor(name) {{
              const map = {{
                Study: "var(--study)",
                Task: "var(--task)",
                Modality: "var(--modality)",
                Analysis: "var(--analysis)",
                Group: "var(--group)",
              }};
              return map[name] || "#eee";
            }}

            function renderChips() {{
              chipRowEl.innerHTML = "";
              entityTypes.forEach((name) => {{
                const chip = document.createElement("button");
                chip.type = "button";
                chip.className = "ev-chip" + (filters[name] ? " active" : "");
                chip.textContent = name;
                chip.style.background = getEntityColor(name);
                chip.addEventListener("click", () => {{
                  filters[name] = !filters[name];
                  chip.classList.toggle("active", filters[name]);
                  renderFieldList();
                  renderTextWindow();
                }});
                chipRowEl.appendChild(chip);
              }});
            }}

            function renderFieldList() {{
              const query = (searchInputEl.value || "").toLowerCase();
              fieldListEl.innerHTML = "";
              const visibleFields = fields.filter((field) => {{
                if (!filters[field.entityType]) return false;
                const node = getNodeByPath(field.path);
                const valueText = node ? formatValue(node.value) : "";
                const label = buildFieldLabel(field).toLowerCase();
                return !query || label.includes(query) || valueText.toLowerCase().includes(query);
              }});

              visibleFields.forEach((field) => {{
                const node = getNodeByPath(field.path);
                const evidenceCount = Array.isArray(node?.evidence) ? node.evidence.length : 0;
                const wrapper = document.createElement("div");
                wrapper.className = "ev-field-item" + (field.id === currentFieldId ? " active" : "");
                wrapper.style.borderLeft = "4px solid " + getEntityColor(field.entityType);
                wrapper.innerHTML = `
                  <div class="ev-field-title">${{escapeHtml(buildFieldLabel(field))}}</div>
                  <div class="ev-field-meta">${{field.entityType}} - evidence ${{evidenceCount}}</div>
                  <div class="ev-field-preview">${{escapeHtml(truncateValue(node?.value))}}</div>
                `;
                wrapper.addEventListener("click", () => {{
                  currentFieldId = field.id;
                  renderFieldList();
                  renderFieldDetail();
                  renderTextWindow();
                }});
                fieldListEl.appendChild(wrapper);
              }});
            }}

            function buildFieldLabel(field) {{
              if (field.entityType === "Study") {{
                return "Study - " + field.fieldKey;
              }}
              const indexLabel = field.entityIndex != null ? "#" + (field.entityIndex + 1) : "";
              const idLabel = field.entityId ? String(field.entityId) : indexLabel;
              return field.entityType + " " + (idLabel || "") + " - " + field.fieldKey;
            }}

            function truncateValue(value) {{
              if (value == null) return "No value";
              const text = typeof value === "string" ? value : JSON.stringify(value);
              if (text.length <= 80) return text;
              return text.slice(0, 77) + "...";
            }}

            function buildHighlightHtml() {{
              const spans = [];
              const spanLengths = {{}};
              let spanId = 0;
              fields.forEach((field) => {{
                if (!filters[field.entityType]) return;
                const node = getNodeByPath(field.path);
                const evidence = Array.isArray(node?.evidence) ? node.evidence : [];
                evidence.forEach((ev, idx) => {{
                  const interval = ev && ev.char_interval;
                  if (!interval) return;
                  const start = Number(interval.start_pos);
                  const end = Number(interval.end_pos);
                  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) return;
                  const id = spanId++;
                  spans.push({{
                    id: id,
                    start: start,
                    end: end,
                    fieldId: field.id,
                    entityType: field.entityType,
                    evidenceIndex: idx,
                    label: buildFieldLabel(field),
                    text: ev.extraction_text || "",
                  }});
                  spanLengths[id] = end - start;
                }});
              }});

              const points = [];
              spans.forEach((span) => {{
                points.push({{ position: span.start, type: "start", span: span }});
                points.push({{ position: span.end, type: "end", span: span }});
              }});

              points.sort((a, b) => {{
                if (a.position !== b.position) return a.position - b.position;
                if (a.type === b.type) {{
                  const lenA = spanLengths[a.span.id] || 0;
                  const lenB = spanLengths[b.span.id] || 0;
                  if (a.type === "end") return lenA - lenB;
                  return lenB - lenA;
                }}
                return a.type === "end" ? -1 : 1;
              }});

              let cursor = 0;
              const parts = [];
              points.forEach((point) => {{
                if (point.position > cursor) {{
                  parts.push(escapeHtml(state.text.slice(cursor, point.position)));
                }}
                if (point.type === "start") {{
                  const tip = point.span.label + (point.span.text ? ": " + point.span.text : "");
                  parts.push(
                    '<span class="ev-span" data-field-id="' + point.span.fieldId +
                    '" data-evidence-index="' + point.span.evidenceIndex +
                    '" data-entity="' + point.span.entityType +
                    '" title="' + escapeHtml(tip) + '">'
                  );
                }} else {{
                  parts.push("</span>");
                }}
                cursor = point.position;
              }});
              if (cursor < state.text.length) {{
                parts.push(escapeHtml(state.text.slice(cursor)));
              }}
              return parts.join("");
            }}

            function renderTextWindow() {{
              textWindowEl.innerHTML = buildHighlightHtml();
              applyCurrentHighlight();
            }}

            function applyCurrentHighlight() {{
              const spans = textWindowEl.querySelectorAll(".ev-span");
              spans.forEach((span) => span.classList.remove("ev-current"));
              const currentSpans = textWindowEl.querySelectorAll(
                '.ev-span[data-field-id="' + currentFieldId + '"]'
              );
              currentSpans.forEach((span) => span.classList.add("ev-current"));
              if (currentSpans.length) {{
                currentSpans[0].scrollIntoView({{ block: "center", behavior: "smooth" }});
              }}
            }}

            function renderFieldDetail() {{
              const field = fields.find((item) => item.id === currentFieldId);
              if (!field) {{
                detailEl.innerHTML = "<div>No field selected.</div>";
                return;
              }}
              const node = getNodeByPath(field.path) || {{}};
              const evidence = Array.isArray(node.evidence) ? node.evidence : [];
              const valueText = formatValue(node.value);
              const fieldPath = formatPath(field.path);

              detailEl.innerHTML = `
                <div class="ev-badge">${{field.entityType}}</div>
                <div class="ev-badge">${{escapeHtml(field.entityId || "No ID")}}</div>
                <div class="ev-badge">${{escapeHtml(fieldPath)}}</div>
                <h3>Value</h3>
                <textarea class="ev-textarea" id="valueInput">${{escapeHtml(valueText)}}</textarea>
                <div class="ev-btn-row">
                  <button class="ev-btn primary" id="applyValueBtn">Apply value</button>
                </div>
                <h3>Evidence spans (${{evidence.length}})</h3>
                <div class="ev-btn-row">
                  <button class="ev-btn danger" id="clearEvidenceBtn">Clear evidence</button>
                </div>
                <div id="evidenceList"></div>
              `;

              const evidenceList = detailEl.querySelector("#evidenceList");
              evidence.forEach((ev, idx) => {{
                const interval = ev.char_interval || {{}};
                const extraction = ev.extraction_text || "";
                const section = ev.section || "";
                const start = interval.start_pos ?? "";
                const end = interval.end_pos ?? "";
                const card = document.createElement("div");
                card.className = "ev-evidence-card";
                card.innerHTML = `
                  <div class="ev-evidence-header">
                    <span>Evidence ${{idx + 1}}</span>
                    <div class="ev-btn-row">
                      <button class="ev-btn" data-action="jump" data-idx="${{idx}}">Jump</button>
                      <button class="ev-btn danger" data-action="remove" data-idx="${{idx}}">Remove</button>
                    </div>
                  </div>
                  <div class="ev-evidence-grid">
                    <label>Start
                      <input class="ev-input" data-field="start" data-idx="${{idx}}" value="${{start}}" />
                    </label>
                    <label>End
                      <input class="ev-input" data-field="end" data-idx="${{idx}}" value="${{end}}" />
                    </label>
                  </div>
                  <label>Section
                    <input class="ev-input" data-field="section" data-idx="${{idx}}" value="${{escapeHtml(section)}}" />
                  </label>
                  <label>Extraction text
                    <textarea class="ev-textarea" data-field="text" data-idx="${{idx}}">${{escapeHtml(extraction)}}</textarea>
                  </label>
                  <div class="ev-btn-row">
                    <button class="ev-btn" data-action="sync" data-idx="${{idx}}">Sync text</button>
                    <button class="ev-btn primary" data-action="apply" data-idx="${{idx}}">Apply span</button>
                  </div>
                `;
                evidenceList.appendChild(card);
              }});

              const applyValueBtn = detailEl.querySelector("#applyValueBtn");
              const valueInput = detailEl.querySelector("#valueInput");
              applyValueBtn.addEventListener("click", () => {{
                try {{
                  node.value = parseValue(valueInput.value);
                  statusEl.textContent = "Value updated.";
                  renderFieldList();
                }} catch (err) {{
                  statusEl.textContent = "Value parse error: " + err.message;
                }}
              }});

              const clearEvidenceBtn = detailEl.querySelector("#clearEvidenceBtn");
              clearEvidenceBtn.addEventListener("click", () => {{
                node.evidence = [];
                statusEl.textContent = "Evidence cleared.";
                renderFieldDetail();
                renderTextWindow();
              }});

              evidenceList.addEventListener("click", (event) => {{
                const btn = event.target.closest("button");
                if (!btn) return;
                const action = btn.dataset.action;
                const idx = Number(btn.dataset.idx);
                if (Number.isNaN(idx)) return;
                if (action === "remove") {{
                  node.evidence.splice(idx, 1);
                  renderFieldDetail();
                  renderTextWindow();
                }} else if (action === "jump") {{
                  const span = textWindowEl.querySelector(
                    '.ev-span[data-field-id="' + field.id + '"][data-evidence-index="' + idx + '"]'
                  );
                  if (span) {{
                    span.scrollIntoView({{ block: "center", behavior: "smooth" }});
                  }}
                }} else if (action === "sync") {{
                  const ev = node.evidence[idx];
                  if (!ev || !ev.char_interval) return;
                  const startPos = Number(ev.char_interval.start_pos);
                  const endPos = Number(ev.char_interval.end_pos);
                  if (!Number.isFinite(startPos) || !Number.isFinite(endPos)) return;
                  ev.extraction_text = state.text.slice(startPos, endPos);
                  renderFieldDetail();
                }} else if (action === "apply") {{
                  const card = btn.closest(".ev-evidence-card");
                  const startInput = card.querySelector('[data-field="start"]');
                  const endInput = card.querySelector('[data-field="end"]');
                  const sectionInput = card.querySelector('[data-field="section"]');
                  const textInput = card.querySelector('[data-field="text"]');
                  const startPos = Number(startInput.value);
                  const endPos = Number(endInput.value);
                  if (!Number.isFinite(startPos) || !Number.isFinite(endPos) || startPos >= endPos) {{
                    statusEl.textContent = "Invalid start/end for evidence.";
                    return;
                  }}
                  node.evidence[idx] = node.evidence[idx] || {{}};
                  node.evidence[idx].char_interval = {{
                    start_pos: startPos,
                    end_pos: endPos,
                  }};
                  node.evidence[idx].section = sectionInput.value.trim() || findSectionTitle(startPos);
                  node.evidence[idx].extraction_text = textInput.value;
                  node.evidence[idx].alignment_status = node.evidence[idx].alignment_status || "manual";
                  node.evidence[idx].source = node.evidence[idx].source || "manual";
                  statusEl.textContent = "Evidence updated.";
                  renderTextWindow();
                }}
              }});
            }}

            function updateDocMeta() {{
              docNameEl.textContent = state.docName || "Document";
              const counts = entityTypes.map((name) => {{
                const count = fields.filter((f) => f.entityType === name).length;
                return name + ": " + count;
              }});
              docStatsEl.textContent = counts.join(" - ");
            }}

            function renderAll() {{
              sections = buildSections(state.text);
              buildFields();
              updateDocMeta();
              buildLegend();
              renderChips();
              renderFieldList();
              renderFieldDetail();
              renderTextWindow();
            }}

            function getOffset(container, node, offset) {{
              const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
              let total = 0;
              let current;
              while ((current = walker.nextNode())) {{
                if (current === node) {{
                  return total + offset;
                }}
                total += current.nodeValue.length;
              }}
              return null;
            }}

            function getSelectionOffsets(container) {{
              const selection = window.getSelection();
              if (!selection || selection.rangeCount === 0) return null;
              const range = selection.getRangeAt(0);
              if (!container.contains(range.commonAncestorContainer)) return null;
              const start = getOffset(container, range.startContainer, range.startOffset);
              const end = getOffset(container, range.endContainer, range.endOffset);
              if (start == null || end == null) return null;
              return start <= end ? [start, end] : [end, start];
            }}

            function addEvidenceFromSelection() {{
              const field = fields.find((item) => item.id === currentFieldId);
              if (!field) return;
              const offsets = getSelectionOffsets(textWindowEl);
              if (!offsets) {{
                statusEl.textContent = "Select text inside the document to add evidence.";
                return;
              }}
              const [start, end] = offsets;
              if (start === end) {{
                statusEl.textContent = "Selection is empty.";
                return;
              }}
              const node = getNodeByPath(field.path);
              if (!node) return;
              const text = state.text.slice(start, end);
              const evidence = {{
                alignment_status: "manual",
                char_interval: {{
                  start_pos: start,
                  end_pos: end,
                }},
                document_id: state.docName || null,
                extraction_text: text,
                section: findSectionTitle(start),
                source: "manual",
              }};
              if (!Array.isArray(node.evidence)) {{
                node.evidence = [];
              }}
              node.evidence.push(evidence);
              statusEl.textContent = "Evidence added.";
              renderFieldDetail();
              renderTextWindow();
            }}

            function updateSelectionInfo() {{
              const offsets = getSelectionOffsets(textWindowEl);
              if (!offsets) {{
                selectionInfoEl.textContent = "Select text to add evidence.";
                return;
              }}
              selectionInfoEl.textContent = "Selection: " + offsets[0] + " - " + offsets[1];
            }}

            function downloadReviewJson() {{
              const jsonString = JSON.stringify(state.record, null, 2);
              const blob = new Blob([jsonString], {{ type: "application/json" }});
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = state.outputName || "review.json";
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            }}

            textWindowEl.addEventListener("mouseup", updateSelectionInfo);
            textWindowEl.addEventListener("keyup", updateSelectionInfo);
            textWindowEl.addEventListener("click", (event) => {{
              const target = event.target.closest(".ev-span");
              if (!target) return;
              const fieldId = Number(target.dataset.fieldId);
              if (!Number.isNaN(fieldId)) {{
                currentFieldId = fieldId;
                renderFieldList();
                renderFieldDetail();
                renderTextWindow();
              }}
            }});

            addEvidenceBtn.addEventListener("click", addEvidenceFromSelection);
            searchInputEl.addEventListener("input", renderFieldList);
            downloadBtn.addEventListener("click", downloadReviewJson);

            if (jsonFileEl) {{
              jsonFileEl.addEventListener("change", (event) => {{
                const file = event.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = () => {{
                  try {{
                    state.record = JSON.parse(reader.result);
                    state.jsonName = file.name;
                    state.docName = file.name.replace(/\\.json$/i, "");
                    state.outputName = file.name.replace(/\\.json$/i, ".review.json");
                    renderAll();
                    statusEl.textContent = "Loaded JSON file.";
                  }} catch (err) {{
                    statusEl.textContent = "JSON load error: " + err.message;
                  }}
                }};
                reader.readAsText(file);
              }});
            }}

            if (mdFileEl) {{
              mdFileEl.addEventListener("change", (event) => {{
                const file = event.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = () => {{
                  state.text = reader.result;
                  state.mdName = file.name;
                  renderAll();
                  statusEl.textContent = "Loaded markdown file.";
                }};
                reader.readAsText(file);
              }});
            }}

            renderAll();
          }})();
        </script>
        """
    )

    if _is_jupyter() and HTML is not None:
        return HTML(html_content)
    return html_content
