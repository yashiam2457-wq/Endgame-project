/**
 * Enterprise GraphRAG — Reasoning Chat & Scoped Query Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  const sendBtn = document.getElementById('chat-send-btn');
  const inputField = document.getElementById('chat-input-field');

  if (sendBtn && inputField) {
    sendBtn.onclick = () => submitQuery(inputField.value);
    inputField.onkeydown = (e) => {
      if (e.key === 'Enter') submitQuery(inputField.value);
    };
  }

  const wbQueryBtn = document.getElementById('workbench-query-btn');
  const wbQueryInput = document.getElementById('workbench-query-input');

  if (wbQueryBtn && wbQueryInput) {
    wbQueryBtn.onclick = () => {
      const q = wbQueryInput.value.trim();
      if (!q) return;
      const targetSec = document.getElementById('sec-chat');
      if (targetSec) targetSec.scrollIntoView({ behavior: 'smooth' });
      submitQuery(q);
      wbQueryInput.value = '';
    };
  }
});

async function submitQuery(queryText) {
  queryText = (queryText || '').trim();
  if (!queryText) return;

  const historyContainer = document.getElementById('chat-history-container');
  const inputField = document.getElementById('chat-input-field');
  if (inputField) inputField.value = '';

  // Append user bubble
  appendBubble(historyContainer, queryText, 'user');

  // Append bot loading bubble
  const botBubble = appendBubble(historyContainer, 'Synthesizing GraphRAG answer...', 'bot');

  try {
    const activeDocs = window.getActiveSelectedDocIds ? window.getActiveSelectedDocIds() : null;
    const docIds = Array.isArray(activeDocs) ? activeDocs : (window.WorkbenchState && window.WorkbenchState.selectedDocIds ? Array.from(window.WorkbenchState.selectedDocIds) : []);

    const requestPayload = {
      query: queryText,
      mode: 'hybrid',
      top_k: 10,
      graph_hops: 2,
      include_graph: true,
      include_citations: true
    };

    if (docIds && docIds.length > 0) {
      requestPayload.filters = { document_id: docIds };
    }

    const res = await fetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestPayload)
    });

    if (!res.ok) {
      botBubble.textContent = 'Error: Failed to generate GraphRAG response.';
      return;
    }

    const json = await res.json();
    if (!json.success) {
      botBubble.textContent = json.message || 'Query processing returned no results.';
      return;
    }

    const responseData = json.data || json;
    botBubble.innerHTML = formatBotResponse(responseData);
  } catch (err) {
    botBubble.textContent = 'Error communicating with GraphRAG backend.';
  }
}

function appendBubble(container, text, sender) {
  if (!container) return null;

  const bubble = document.createElement('div');
  bubble.className = `msg-bubble ${sender === 'user' ? 'msg-user' : 'msg-bot'}`;
  bubble.textContent = text;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

function formatBotResponse(data) {
  if (typeof data === 'string') {
    return `<p style="margin-bottom:0.6rem;">${data}</p>`;
  }

  let answer = data.answer || data.response || data.output || 'No grounded answer generated.';

  // Format line breaks into clean HTML paragraphs
  let answerHtml = answer.split('\n\n').map(p => `<p style="margin-bottom:0.6rem;">${p.replace(/\n/g, '<br/>')}</p>`).join('');

  // Include reasoning explanation box if present
  if (data.reasoning) {
    answerHtml += `<div style="font-size:0.8rem;color:var(--text-secondary);background:rgba(255,255,255,0.03);padding:0.6rem 0.85rem;border-radius:6px;border:1px solid var(--border-color);margin-top:0.5rem;line-height:1.5;"><strong>GraphRAG Reasoning:</strong> ${data.reasoning}</div>`;
  }

  // Performance telemetry bar
  let metaHtml = `<div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid rgba(255,255,255,0.08);display:flex;gap:0.75rem;flex-wrap:wrap;">`;
  if (data.intent) metaHtml += `<span>Intent: <strong>${data.intent}</strong></span>`;
  if (data.confidence_score !== undefined) metaHtml += `<span>Confidence: <strong>${(data.confidence_score * 100).toFixed(0)}%</strong></span>`;
  if (data.total_time_ms) metaHtml += `<span>Total Latency: <strong>${data.total_time_ms.toFixed(0)} ms</strong></span>`;
  if (data.graph_nodes_visited !== undefined) metaHtml += `<span>Graph Nodes Visited: <strong>${data.graph_nodes_visited}</strong></span>`;
  metaHtml += `</div>`;

  // Add detected entities badges if present
  let entitiesHtml = '';
  if (data.entities_detected && data.entities_detected.length > 0) {
    entitiesHtml += '<div style="margin-top:0.5rem;font-size:0.75rem;color:var(--text-muted);">';
    entitiesHtml += '<strong>Entities Detected:</strong> ';
    data.entities_detected.forEach(ent => {
      entitiesHtml += `<span style="background:rgba(99,102,241,0.15);color:#a5b4fc;padding:1px 6px;border-radius:4px;margin-right:0.25rem;font-size:0.7rem;">${ent}</span>`;
    });
    entitiesHtml += '</div>';
  }

  return answerHtml + metaHtml + entitiesHtml;
}
