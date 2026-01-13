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

"""Editable LangExtract visualization for review and training export."""

from __future__ import annotations

import json
import pathlib
import textwrap

from langextract import data_lib
from langextract import io
from langextract.core import data

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


_PALETTE: list[str] = [
    "#D2E3FC",
    "#C8E6C9",
    "#FEF0C3",
    "#F9DEDC",
    "#FFDDBE",
    "#EADDFF",
    "#C4E9E4",
    "#FCE4EC",
    "#E8EAED",
    "#DDE8E8",
]

_EDITABLE_CSS = textwrap.dedent(
    """\
    <style>
    .lx-highlight { position: relative; border-radius:3px; padding:1px 2px; }
    .lx-highlight .lx-tooltip {
      visibility: hidden;
      opacity: 0;
      transition: opacity 0.2s ease-in-out;
      background: #333;
      color: #fff;
      text-align: left;
      border-radius: 4px;
      padding: 6px 8px;
      position: absolute;
      z-index: 1000;
      bottom: 125%;
      left: 50%;
      transform: translateX(-50%);
      font-size: 12px;
      max-width: 240px;
      white-space: normal;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .lx-highlight:hover .lx-tooltip { visibility: visible; opacity:1; }
    .lx-animated-wrapper { max-width: 100%; font-family: Arial, sans-serif; }
    .lx-controls {
      background: #fafafa; border: 1px solid #90caf9; border-radius: 8px;
      padding: 12px; margin-bottom: 16px;
    }
    .lx-button-row {
      display: flex; justify-content: center; gap: 8px; margin-bottom: 12px;
    }
    .lx-control-btn {
      background: #4285f4; color: white; border: none; border-radius: 4px;
      padding: 8px 12px; cursor: pointer; font-size: 13px; font-weight: 500;
      transition: background-color 0.2s;
    }
    .lx-control-btn:hover { background: #3367d6; }
    .lx-progress-container { margin-bottom: 8px; }
    .lx-progress-slider {
      width: 100%; margin: 0; appearance: none; height: 6px;
      background: #ddd; border-radius: 3px; outline: none;
    }
    .lx-progress-slider::-webkit-slider-thumb {
      appearance: none; width: 18px; height: 18px; background: #4285f4;
      border-radius: 50%; cursor: pointer;
    }
    .lx-progress-slider::-moz-range-thumb {
      width: 18px; height: 18px; background: #4285f4; border-radius: 50%;
      cursor: pointer; border: none;
    }
    .lx-status-text {
      text-align: center; font-size: 12px; color: #666; margin-top: 4px;
    }
    .lx-text-window {
      font-family: monospace; white-space: pre-wrap; border: 1px solid #90caf9;
      padding: 12px; max-height: 320px; overflow-y: auto; margin-bottom: 12px;
      line-height: 1.6;
    }
    .lx-attributes-panel {
      background: #fafafa; border: 1px solid #90caf9; border-radius: 6px;
      padding: 8px 10px; margin-top: 8px; font-size: 13px;
    }
    .lx-current-highlight {
      border-bottom: 3px solid #ff4444;
      font-weight: bold;
    }
    .lx-legend {
      font-size: 12px; margin-bottom: 8px;
      padding-bottom: 8px; border-bottom: 1px solid #e0e0e0;
    }
    .lx-label {
      display: inline-block;
      padding: 2px 4px;
      border-radius: 3px;
      margin-right: 4px;
      color: #000;
    }
    .lx-attr-key { font-weight: 600; color: #1565c0; letter-spacing: 0.3px; }
    .lx-attr-value { font-weight: 400; opacity: 0.85; letter-spacing: 0.2px; }
    .lx-editor {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
    }
    .lx-editor-panel {
      border: 1px solid #90caf9;
      border-radius: 8px;
      padding: 12px;
      background: #fbfbfb;
    }
    .lx-editor-panel label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      margin: 8px 0 4px;
      color: #333;
    }
    .lx-editor-panel input,
    .lx-editor-panel textarea {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #cfd8dc;
      border-radius: 4px;
      padding: 6px 8px;
      font-size: 12px;
      font-family: Arial, sans-serif;
    }
    .lx-editor-panel textarea { min-height: 80px; font-family: monospace; }
    .lx-editor-title {
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .lx-editor-actions {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .lx-editor-actions button {
      background: #1976d2;
      color: #fff;
      border: none;
      border-radius: 4px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }
    .lx-editor-actions button.secondary { background: #5c6bc0; }
    .lx-editor-actions button.ghost { background: #455a64; }
    .lx-save-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 10px;
      gap: 8px;
      font-size: 12px;
    }
    .lx-status-msg {
      font-size: 12px;
      margin-top: 8px;
      color: #c62828;
      min-height: 16px;
    }
    @media (max-width: 900px) {
      .lx-editor { grid-template-columns: 1fr; }
    }

    .lx-gif-optimized .lx-text-window { font-size: 16px; line-height: 1.8; }
    .lx-gif-optimized .lx-editor-panel { font-size: 14px; }
    </style>"""
)


def visualize_editable(
    data_source: data.AnnotatedDocument | str | pathlib.Path,
    *,
    output_filename: str = "langextract_review.jsonl",
    show_legend: bool = True,
    gif_optimized: bool = True,
) -> HTML | str:
  """Visualizes LangExtract output with inline editing and JSONL export.

  Args:
    data_source: AnnotatedDocument or path to a JSONL file.
    output_filename: Filename for the download button in the UI.
    show_legend: If ``True``, show a color legend for extraction classes.
    gif_optimized: If ``True``, applies larger fonts for screen capture.

  Returns:
    An IPython HTML object if available, otherwise a raw HTML string.
  """
  if isinstance(data_source, (str, pathlib.Path)):
    file_path = pathlib.Path(data_source)
    if not file_path.exists():
      raise FileNotFoundError(f"JSONL file not found: {file_path}")
    documents = list(io.load_annotated_documents_jsonl(file_path))
    if not documents:
      raise ValueError(f"No documents found in JSONL file: {file_path}")
    annotated_doc = documents[0]
  else:
    annotated_doc = data_source

  if not annotated_doc or annotated_doc.text is None:
    raise ValueError("annotated_doc must contain text to visualise.")
  if annotated_doc.extractions is None:
    raise ValueError("annotated_doc must contain extractions to visualise.")

  doc_dict = data_lib.annotated_document_to_dict(annotated_doc)
  doc_json = json.dumps(doc_dict, ensure_ascii=False)
  palette_json = json.dumps(_PALETTE)
  output_filename_json = json.dumps(output_filename)
  show_legend_json = "true" if show_legend else "false"

  html_content = textwrap.dedent(
      f"""
      <div class="lx-animated-wrapper" id="lxEditableRoot">
        <div class="lx-attributes-panel">
          <div id="legendContainer"></div>
          <div id="attributesContainer"></div>
        </div>
        <div class="lx-editor">
          <div>
            <div class="lx-text-window" id="textWindow"></div>
            <div class="lx-controls">
              <div class="lx-button-row">
                <button class="lx-control-btn" onclick="prevExtraction()">Prev</button>
                <button class="lx-control-btn" onclick="nextExtraction()">Next</button>
              </div>
              <div class="lx-progress-container">
                <input type="range" id="progressSlider" class="lx-progress-slider"
                       min="0" max="0" value="0"
                       onchange="jumpToExtraction(this.value)">
              </div>
              <div class="lx-status-text">
                Entity <span id="entityInfo">0/0</span> |
                Pos <span id="posInfo">[]</span>
              </div>
            </div>
          </div>
          <div class="lx-editor-panel">
            <div class="lx-editor-title">Edit Extraction</div>
            <label for="classInput">Class</label>
            <input id="classInput" type="text" />
            <label for="startInput">Start</label>
            <input id="startInput" type="number" min="0" />
            <label for="endInput">End</label>
            <input id="endInput" type="number" min="0" />
            <label for="textInput">Extraction Text</label>
            <textarea id="textInput"></textarea>
            <label for="attributesInput">Attributes (JSON)</label>
            <textarea id="attributesInput"></textarea>
            <div class="lx-editor-actions">
              <button onclick="applyEdits()">Apply</button>
              <button class="secondary" onclick="syncTextFromSpan()">Sync Text</button>
            </div>
            <div class="lx-save-row">
              <label><input type="checkbox" id="syncOnSave" checked /> Sync text on save</label>
              <button class="ghost" onclick="downloadJsonl()">Download JSONL</button>
            </div>
            <div id="statusMsg" class="lx-status-msg"></div>
          </div>
        </div>
      </div>

      <script>
        (function() {{
          const docData = {doc_json};
          const palette = {palette_json};
          const outputFilename = {output_filename_json};
          const showLegend = {show_legend_json};
          const text = docData.text || "";

          const extractions = (docData.extractions || []).map((extraction, idx) => {{
            const charInterval = extraction.char_interval || null;
            return {{
              index: idx,
              extraction_class: extraction.extraction_class || "",
              extraction_text: extraction.extraction_text || "",
              startPos: charInterval ? charInterval.start_pos : null,
              endPos: charInterval ? charInterval.end_pos : null,
              attributes: extraction.attributes || {{}},
              alignment_status: extraction.alignment_status || null,
              extraction_index: extraction.extraction_index ?? null,
              group_index: extraction.group_index ?? null,
              description: extraction.description || null,
            }};
          }});

          let currentIndex = 0;
          let colorMap = {{}};

          const legendContainer = document.getElementById("legendContainer");
          const attributesContainer = document.getElementById("attributesContainer");
          const textWindow = document.getElementById("textWindow");
          const entityInfo = document.getElementById("entityInfo");
          const posInfo = document.getElementById("posInfo");
          const progressSlider = document.getElementById("progressSlider");
          const classInput = document.getElementById("classInput");
          const startInput = document.getElementById("startInput");
          const endInput = document.getElementById("endInput");
          const textInput = document.getElementById("textInput");
          const attributesInput = document.getElementById("attributesInput");
          const statusMsg = document.getElementById("statusMsg");
          const syncOnSave = document.getElementById("syncOnSave");

          function escapeHtml(value) {{
            return String(value)
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#39;");
          }}

          function assignColors() {{
            const classes = Array.from(new Set(
              extractions.map((ex) => ex.extraction_class).filter(Boolean)
            )).sort();
            const map = {{}};
            classes.forEach((cls, idx) => {{
              map[cls] = palette[idx % palette.length];
            }});
            colorMap = map;
          }}

          function buildLegendHtml() {{
            if (!showLegend) return "";
            const entries = Object.entries(colorMap);
            if (!entries.length) return "";
            const labels = entries.map(([cls, color]) => {{
              return '<span class="lx-label" style="background-color:' + color + ';">' +
                escapeHtml(cls) + '</span>';
            }});
            return '<div class="lx-legend">Highlights Legend: ' + labels.join(" ") + '</div>';
          }}

          function buildAttributesHtml(extraction) {{
            const attrs = extraction.attributes || {{}};
            const parts = ['<div><strong>class:</strong> ' + escapeHtml(extraction.extraction_class) + '</div>'];
            const keys = Object.keys(attrs);
            if (!keys.length) {{
              parts.push('<div><strong>attributes:</strong> {{}}</div>');
            }} else {{
              const formatted = keys.map((key) => {{
                const value = attrs[key];
                const textValue = Array.isArray(value) ? value.join(", ") : String(value);
                return '<span class="lx-attr-key">' + escapeHtml(key) + '</span>: ' +
                  '<span class="lx-attr-value">' + escapeHtml(textValue) + '</span>';
              }});
              parts.push('<div><strong>attributes:</strong> {{' + formatted.join(", ") + '}}</div>');
            }}
            return parts.join("");
          }}

          function buildHighlightedHtml() {{
            const points = [];
            const spanLengths = {{}};
            extractions.forEach((ex) => {{
              if (ex.startPos == null || ex.endPos == null) return;
              if (ex.startPos >= ex.endPos) return;
              points.push({{ position: ex.startPos, tagType: "start", spanIdx: ex.index }});
              points.push({{ position: ex.endPos, tagType: "end", spanIdx: ex.index }});
              spanLengths[ex.index] = ex.endPos - ex.startPos;
            }});

            points.sort((a, b) => {{
              if (a.position !== b.position) return a.position - b.position;
              if (a.tagType === b.tagType) {{
                const lenA = spanLengths[a.spanIdx] || 0;
                const lenB = spanLengths[b.spanIdx] || 0;
                if (a.tagType === "end") return lenA - lenB;
                return lenB - lenA;
              }}
              return a.tagType === "end" ? -1 : 1;
            }});

            let cursor = 0;
            const parts = [];
            points.forEach((point) => {{
              if (point.position > cursor) {{
                parts.push(escapeHtml(text.slice(cursor, point.position)));
              }}
              if (point.tagType === "start") {{
                const ex = extractions.find((item) => item.index === point.spanIdx);
                const color = colorMap[ex.extraction_class] || "#ffff8d";
                parts.push('<span class="lx-highlight" data-idx="' + point.spanIdx + '" style="background-color:' + color + ';">');
              }} else {{
                parts.push("</span>");
              }}
              cursor = point.position;
            }});

            if (cursor < text.length) {{
              parts.push(escapeHtml(text.slice(cursor)));
            }}
            return parts.join("");
          }}

          function renderTextWindow() {{
            textWindow.innerHTML = buildHighlightedHtml();
            applyCurrentHighlight();
          }}

          function applyCurrentHighlight() {{
            const prev = textWindow.querySelector(".lx-current-highlight");
            if (prev) prev.classList.remove("lx-current-highlight");
            const currentSpan = textWindow.querySelector('span[data-idx="' + currentIndex + '"]');
            if (currentSpan) {{
              currentSpan.classList.add("lx-current-highlight");
              currentSpan.scrollIntoView({{block: "center", behavior: "smooth"}});
            }}
          }}

          function updateDisplay() {{
            const extraction = extractions[currentIndex];
            if (!extraction) return;
            attributesContainer.innerHTML = buildAttributesHtml(extraction);
            entityInfo.textContent = (currentIndex + 1) + "/" + extractions.length;
            if (extraction.startPos == null || extraction.endPos == null) {{
              posInfo.textContent = "[]";
            }} else {{
              posInfo.textContent = "[" + extraction.startPos + "-" + extraction.endPos + "]";
            }}
            progressSlider.max = Math.max(0, extractions.length - 1);
            progressSlider.value = currentIndex;
            classInput.value = extraction.extraction_class || "";
            startInput.value = extraction.startPos == null ? "" : extraction.startPos;
            endInput.value = extraction.endPos == null ? "" : extraction.endPos;
            textInput.value = extraction.extraction_text || "";
            attributesInput.value = JSON.stringify(extraction.attributes || {{}}, null, 2);
            statusMsg.textContent = "";
            applyCurrentHighlight();
          }}

          function nextExtraction() {{
            if (!extractions.length) return;
            currentIndex = (currentIndex + 1) % extractions.length;
            updateDisplay();
          }}

          function prevExtraction() {{
            if (!extractions.length) return;
            currentIndex = (currentIndex - 1 + extractions.length) % extractions.length;
            updateDisplay();
          }}

          function jumpToExtraction(index) {{
            if (!extractions.length) return;
            currentIndex = parseInt(index, 10);
            updateDisplay();
          }}

          function parseNumber(value) {{
            if (value === "") return null;
            const parsed = Number(value);
            if (!Number.isFinite(parsed) || parsed < 0) return NaN;
            return Math.floor(parsed);
          }}

          function applyEdits() {{
            const extraction = extractions[currentIndex];
            if (!extraction) return;
            const startValue = parseNumber(startInput.value.trim());
            const endValue = parseNumber(endInput.value.trim());
            if (Number.isNaN(startValue) || Number.isNaN(endValue)) {{
              statusMsg.textContent = "Start/end must be non-negative integers or blank.";
              return;
            }}
            if (startValue != null && endValue != null && startValue > endValue) {{
              statusMsg.textContent = "Start must be <= end.";
              return;
            }}
            let attrs = {{}};
            const rawAttrs = attributesInput.value.trim();
            if (rawAttrs) {{
              try {{
                attrs = JSON.parse(rawAttrs);
              }} catch (err) {{
                statusMsg.textContent = "Attributes JSON is invalid: " + err.message;
                return;
              }}
            }}

            extraction.extraction_class = classInput.value.trim();
            extraction.startPos = startValue;
            extraction.endPos = endValue;
            extraction.extraction_text = textInput.value;
            extraction.attributes = attrs;

            assignColors();
            legendContainer.innerHTML = buildLegendHtml();
            renderTextWindow();
            updateDisplay();
          }}

          function syncTextFromSpan() {{
            const extraction = extractions[currentIndex];
            if (!extraction) return;
            if (extraction.startPos == null || extraction.endPos == null) {{
              statusMsg.textContent = "Set start/end before syncing text.";
              return;
            }}
            extraction.extraction_text = text.slice(extraction.startPos, extraction.endPos);
            textInput.value = extraction.extraction_text;
            statusMsg.textContent = "";
          }}

          function buildOutputDoc() {{
            const output = {{
              document_id: docData.document_id || null,
              text: text,
              extractions: extractions.map((ex) => {{
                const charInterval = (ex.startPos == null || ex.endPos == null)
                  ? null
                  : {{ start_pos: ex.startPos, end_pos: ex.endPos }};
                return {{
                  extraction_class: ex.extraction_class,
                  extraction_text: ex.extraction_text,
                  char_interval: charInterval,
                  alignment_status: ex.alignment_status,
                  extraction_index: ex.extraction_index,
                  group_index: ex.group_index,
                  description: ex.description,
                  attributes: ex.attributes,
                }};
              }}),
            }};
            return output;
          }}

          function downloadJsonl() {{
            if (syncOnSave && syncOnSave.checked) {{
              extractions.forEach((ex) => {{
                if (ex.startPos == null || ex.endPos == null) return;
                ex.extraction_text = text.slice(ex.startPos, ex.endPos);
              }});
            }}
            const outputDoc = buildOutputDoc();
            const jsonl = JSON.stringify(outputDoc) + "\\n";
            const blob = new Blob([jsonl], {{ type: "application/jsonl" }});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = outputFilename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          }}

          textWindow.addEventListener("click", (event) => {{
            const target = event.target.closest("span[data-idx]");
            if (!target) return;
            const idx = Number(target.getAttribute("data-idx"));
            if (!Number.isNaN(idx)) {{
              currentIndex = idx;
              updateDisplay();
            }}
          }});

          window.nextExtraction = nextExtraction;
          window.prevExtraction = prevExtraction;
          window.jumpToExtraction = jumpToExtraction;
          window.applyEdits = applyEdits;
          window.syncTextFromSpan = syncTextFromSpan;
          window.downloadJsonl = downloadJsonl;

          assignColors();
          legendContainer.innerHTML = buildLegendHtml();
          renderTextWindow();
          updateDisplay();
        }})();
      </script>"""
  )

  full_html = _EDITABLE_CSS + html_content
  if gif_optimized:
    full_html = full_html.replace(
        'class="lx-animated-wrapper"',
        'class="lx-animated-wrapper lx-gif-optimized"',
    )

  if HTML is not None and _is_jupyter():
    return HTML(full_html)
  return full_html
