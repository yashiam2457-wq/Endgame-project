/**
 * Enterprise GraphRAG — Core Application Controller & Full Backend Integration
 */

document.addEventListener('DOMContentLoaded', () => {
  initScrollNavigation();
  initCommandPalette();
  initUploadHandlers();

  // Load all backend modules simultaneously
  fetchSystemStats();
  fetchHealthAndMetrics();
  if (window.initWorkbench) window.initWorkbench();
  else if (window.loadDocumentLibrary) window.loadDocumentLibrary();
  if (window.initGraph) window.initGraph();
  if (window.loadLogs) window.loadLogs();
  loadSystemConfig();

  // Poll system stats & health every 10 seconds
  setInterval(() => {
    fetchSystemStats();
    fetchHealthAndMetrics();
  }, 10000);
});

// Single-Page Smooth Scroll & Active ScrollSpy Navigation
function initScrollNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const mainCanvas = document.getElementById('main-canvas');
  const sections = document.querySelectorAll('.section');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetId = item.getAttribute('data-target');
      if (!targetId) return;

      const targetSec = document.getElementById(targetId);
      if (targetSec) {
        targetSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  if (mainCanvas && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const secId = entry.target.id;
          navItems.forEach(item => {
            if (item.getAttribute('data-target') === secId) {
              item.classList.add('active');
            } else {
              item.classList.remove('active');
            }
          });
        }
      });
    }, {
      root: mainCanvas,
      threshold: 0.2
    });

    sections.forEach(sec => observer.observe(sec));
  }
}

// 1. Fetch System Telemetries (GET /api/v1/stats)
async function fetchSystemStats() {
  try {
    const res = await fetch('/api/v1/stats');
    if (!res.ok) return;
    const json = await res.json();
    // API wraps data in { success, data: {...} }
    const d = (json.data) || json;

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    // Dashboard meter bar
    set('dash-docs-val',    d.document_count ?? 0);
    set('dash-chunks-val',  d.chunk_count    ?? 0);
    set('dash-vectors-val', d.faiss_index_size ?? 0);               // faiss vectors
    set('dash-ram-val',     Math.round(d.ram_used_mb ?? 0) + ' MB');
    set('dash-queries-val', d.query_count    ?? 0);
    set('dash-latency-val', Math.round(d.avg_query_latency_ms ?? 0) + ' ms');
    set('dash-cache-val',   Math.round((d.cache_hit_rate ?? 0) * 100) + '%');

    // Top-bar mini chips
    set('top-docs',   (d.document_count ?? 0) + ' docs');
    set('top-chunks', (d.chunk_count ?? 0) + ' chunks');

    // Knowledge Graph section: entity/relationship counts (from stats)
    set('graph-stat-nodes', d.graph_node_count ?? 0);
    set('graph-stat-edges', d.graph_edge_count ?? 0);

    // Health status cards in Settings section
    set('hsc-docs',    d.document_count ?? 0);
    set('hsc-vectors', d.faiss_index_size ?? 0);
    if (d.ram_used_mb != null && d.ram_total_mb != null) {
      set('hsc-ram', Math.round(d.ram_used_mb) + ' / ' + Math.round(d.ram_total_mb) + ' MB');
    }

    // Animate value changes with a quick flash
    document.querySelectorAll('.meter-item-val, .gs-val').forEach(el => {
      el.classList.remove('val-flash');
      void el.offsetWidth;  // reflow
      el.classList.add('val-flash');
    });

  } catch (err) {
    console.error('[GraphRAG] Failed to fetch stats:', err);
  }
}

// 2. Fetch System Health & Metrics (GET /api/v1/health + GET /api/v1/metrics)
async function fetchHealthAndMetrics() {
  try {
    // --- Health check ---
    const hRes = await fetch('/api/v1/health');
    const topHealth  = document.getElementById('top-health');
    const sideStatus = document.getElementById('sidebar-status');
    const sideModel  = document.getElementById('sidebar-model');

    if (hRes.ok) {
      const hJson = await hRes.json();
      const h = hJson.data || hJson;
      const isHealthy = (h.status === 'healthy');

      if (topHealth) {
        topHealth.textContent = isHealthy ? 'Core Online' : 'Degraded';
        topHealth.style.color = isHealthy ? 'var(--success)' : 'var(--warning)';
      }
      if (sideStatus) {
        sideStatus.textContent = isHealthy ? 'Healthy' : 'Degraded';
        sideStatus.style.color = isHealthy ? 'var(--success)' : 'var(--warning, #f59e0b)';
      }
      if (sideModel && h.llm_model) {
        // Shorten long model names
        const shortName = h.llm_model.split('/').pop();
        sideModel.textContent = 'LLM: ' + shortName;
      }

      // Sub-system status indicators — Neo4j is optional
      const setStatus = (id, status) => {
        const el = document.getElementById(id);
        if (!el) return;
        const s = String(status || '').toLowerCase();
        if (s.includes('connected') || s.includes('healthy')) {
          el.textContent = '● Online';
          el.style.color = 'var(--success)';
        } else if (s.includes('empty')) {
          el.textContent = '○ Empty';
          el.style.color = 'var(--text-muted)';
        } else {
          el.textContent = '◎ Offline';
          el.style.color = 'var(--text-muted)';
        }
      };
      setStatus('status-neo4j',  h.neo4j);
      setStatus('status-faiss',  h.faiss);
      setStatus('status-sqlite', h.sqlite);

      // Uptime
      if (h.uptime_seconds != null) {
        const el = document.getElementById('status-uptime');
        if (el) {
          const s = Math.round(h.uptime_seconds);
          const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
          el.textContent = `${hh}h ${mm}m ${ss}s`;
        }
      }
    } else {
      if (topHealth)  { topHealth.textContent  = 'Offline'; topHealth.style.color  = 'var(--danger)'; }
      if (sideStatus) { sideStatus.textContent = 'Offline'; sideStatus.style.color = 'var(--danger)'; }
    }

    // --- Metrics (faiss vectors, query counts, latency) ---
    const mRes = await fetch('/api/v1/metrics');
    if (mRes.ok) {
      const mJson = await mRes.json();
      const m = mJson.data || mJson;

      // Only update if stats fetch didn't already set them
      const qEl = document.getElementById('dash-queries-val');
      if (qEl && qEl.textContent === '0') qEl.textContent = m.total_queries ?? 0;

      const lEl = document.getElementById('dash-latency-val');
      if (lEl && lEl.textContent === '0 ms') lEl.textContent = Math.round(m.avg_latency_ms ?? 0) + ' ms';

      const vEl = document.getElementById('dash-vectors-val');
      if (vEl && vEl.textContent === '0') vEl.textContent = m.faiss_total_vectors ?? 0;
    }
  } catch (e) {
    console.error('[GraphRAG] Failed to fetch health/metrics:', e);
  }
}

// 3. Document Upload & Auto Indexing (POST /api/v1/upload + POST /api/v1/index)
function initUploadHandlers() {
  const fileInput = document.getElementById('file-input-element');
  const dropZone = document.getElementById('upload-drop-zone');

  if (fileInput) {
    fileInput.onchange = () => {
      if (fileInput.files.length > 0) uploadFiles(fileInput.files);
    };
  }

  if (dropZone) {
    dropZone.ondragover = (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    };
    dropZone.ondragleave = () => dropZone.classList.remove('drag-over');
    dropZone.ondrop = (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length > 0) uploadFiles(e.dataTransfer.files);
    };
  }
}

function updateProgressBar(percent, statusMsg, detailMsg) {
  const container = document.getElementById('upload-progress-container');
  const fill = document.getElementById('upload-progress-fill');
  const percentText = document.getElementById('upload-progress-percent');
  const statusText = document.getElementById('upload-status-text');
  const detailText = document.getElementById('upload-detail-text');

  if (container) container.style.display = 'block';
  if (fill) fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  if (percentText) percentText.textContent = `${Math.round(percent)}%`;
  if (statusText && statusMsg) statusText.textContent = statusMsg;
  if (detailText && detailMsg) detailText.textContent = detailMsg;
}

async function pollIndexingJobProgress(jobId, filePrefix, fileName) {
  return new Promise((resolve) => {
    let attempts = 0;
    const pollInterval = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch(`/api/v1/index/progress/${jobId}`);
        const data = await res.json();
        if (data.success && data.data) {
          const info = data.data;
          const pct = info.progress_percent || 0;
          const stage = info.stage || 'Processing';
          const detail = info.detail || '';

          updateProgressBar(pct, `${filePrefix}${stage}: ${fileName}`, detail);

          if (info.status === 'completed' || pct >= 100 || info.status === 'failed' || attempts > 200) {
            clearInterval(pollInterval);
            resolve(info);
          }
        }
      } catch (e) {
        console.error('Progress polling error:', e);
      }
    }, 500);
  });
}

async function uploadFiles(files) {
  const progressContainer = document.getElementById('upload-progress-container');
  if (progressContainer) progressContainer.style.display = 'block';

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const filePrefix = `[${i + 1}/${files.length}] `;

    updateProgressBar(
      5,
      `${filePrefix}Uploading ${file.name}...`,
      'Transferring document stream...'
    );

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Step A: POST /api/v1/upload
      const res = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData
      });
      const uploadResult = await res.json();

      if (uploadResult.success && uploadResult.data) {
        const docId = uploadResult.data.document_id;

        updateProgressBar(
          15,
          `${filePrefix}Triggering Indexing: ${file.name}`,
          'Queuing pipeline tasks...'
        );

        // Step B: POST /api/v1/index
        const indexRes = await fetch('/api/v1/index', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_id: docId, force: true })
        });
        const indexResult = await indexRes.json();

        if (indexResult.success && indexResult.data && indexResult.data.job_id) {
          const jobId = indexResult.data.job_id;
          // Poll real-time backend progress
          await pollIndexingJobProgress(jobId, filePrefix, file.name);
        } else {
          updateProgressBar(100, `${filePrefix}Indexing finished: ${file.name}`, indexResult.message || 'Complete');
        }
      }
    } catch (err) {
      console.error('Upload failed for', file.name, err);
      updateProgressBar(0, `Failed for ${file.name}`, err.message || 'Network error');
    }
  }

  updateProgressBar(100, 'Ingestion & indexing complete!', 'All document vectors & Knowledge Graph updated');
  setTimeout(() => {
    if (progressContainer) progressContainer.style.display = 'none';
  }, 3500);

  // Refresh document library, graph, and telemetries
  fetchSystemStats();
  if (window.loadDocumentLibrary) window.loadDocumentLibrary();
  if (window.initGraph) window.initGraph();
}

// 4. Batch Trigger Indexing (POST /api/v1/index)
window.triggerBatchIndex = async function() {
  const btn = document.getElementById('btn-sync-index');
  if (btn) btn.textContent = 'Syncing...';

  updateProgressBar(10, 'Syncing & Indexing Pending Documents...', 'Scanning pending files...');

  let batchPct = 15;
  const batchTimer = setInterval(() => {
    if (batchPct < 90) {
      batchPct += 10;
      updateProgressBar(batchPct, 'Batch Indexing in progress...', 'Extracting entities, embeddings & Neo4j graph...');
    }
  }, 1000);

  try {
    const res = await fetch('/api/v1/index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_id: null, force: false })
    });
    clearInterval(batchTimer);
    const result = await res.json();
    updateProgressBar(100, 'Batch Indexing Complete!', result.message || 'All pending documents indexed.');
    setTimeout(() => {
      const container = document.getElementById('upload-progress-container');
      if (container) container.style.display = 'none';
    }, 3500);

    fetchSystemStats();
    if (window.loadDocumentLibrary) window.loadDocumentLibrary();
    if (window.initGraph) window.initGraph();
  } catch (e) {
    clearInterval(batchTimer);
    alert('Failed to trigger batch indexing.');
  } finally {
    if (btn) btn.textContent = 'Sync & Index All Pending';
  }
};

// 5. System Configuration Reader (GET /api/v1/config)
window.loadSystemConfig = async function() {
  const container = document.getElementById('sys-config-container');
  if (!container) return;

  try {
    const res = await fetch('/api/v1/config');
    if (!res.ok) return;
    const data = await res.json();
    const cfg = data.data || {};

    container.innerHTML = `
      <div><strong>Application Name:</strong> ${cfg.app_name || 'Enterprise GraphRAG'} v${cfg.version || '1.0.0'}</div>
      <div><strong>Embedding Model:</strong> ${cfg.model ? cfg.model.embedding_model_name : 'all-MiniLM-L6-v2'}</div>
      <div><strong>LLM Generation Model:</strong> ${cfg.model ? cfg.model.generation_model_name : 'google/gemma-4-1b-it'}</div>
      <div><strong>Chunk Strategy:</strong> ${cfg.chunking ? cfg.chunking.strategy : 'recursive'} (Size: ${cfg.chunking ? cfg.chunking.chunk_size : 512})</div>
      <div><strong>FAISS Top-K:</strong> ${cfg.faiss ? cfg.faiss.top_k : 10}</div>
      <div><strong>RAG Max Context Tokens:</strong> ${cfg.retrieval ? cfg.retrieval.max_context_tokens : 3072}</div>
    `;
  } catch (e) {
    container.textContent = 'Failed to load system config.';
  }
};

// 6. Logs Loader (GET /api/v1/logs)
window.loadLogs = async function() {
  const pre = document.getElementById('sys-logs-pre');
  if (!pre) return;
  try {
    const res = await fetch('/api/v1/logs');
    if (res.ok) {
      const data = await res.json();
      pre.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    } else {
      pre.textContent = 'Server logs active.';
    }
  } catch (e) {
    pre.textContent = 'Failed to retrieve server logs.';
  }
};

// Command Palette (Ctrl + K) Modal Controller
function initCommandPalette() {
  const triggerBtn = document.getElementById('global-cmd-trigger');
  const modal = document.getElementById('cmd-modal');
  const listContainer = document.getElementById('cmd-options-list');

  if (!triggerBtn || !modal || !listContainer) return;

  const routes = [
    { name: 'Scroll to Dashboard', target: 'sec-dashboard' },
    { name: 'Scroll to Ingestion & Upload', target: 'sec-upload' },
    { name: 'Scroll to Document Repository', target: 'sec-documents' },
    { name: 'Scroll to Multi-File Workbench', target: 'sec-workbench' },
    { name: 'Scroll to Knowledge Graph', target: 'sec-graph' },
    { name: 'Scroll to Reasoning Chat', target: 'sec-chat' },
    { name: 'Scroll to System Configuration & Logs', target: 'sec-settings' },
  ];

  triggerBtn.addEventListener('click', () => {
    renderCmdOptions();
    modal.classList.add('open');
  });

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      renderCmdOptions();
      modal.classList.toggle('open');
    }
  });

  function renderCmdOptions() {
    listContainer.innerHTML = '';
    routes.forEach(route => {
      const btn = document.createElement('button');
      btn.className = 'btn btn-secondary';
      btn.style.justifyContent = 'flex-start';
      btn.style.width = '100%';
      btn.textContent = route.name;
      btn.onclick = () => {
        modal.classList.remove('open');
        const targetSec = document.getElementById(route.target);
        if (targetSec) targetSec.scrollIntoView({ behavior: 'smooth' });
      };
      listContainer.appendChild(btn);
    });
  }
}
