/**
 * Enterprise GraphRAG — Multi-File Workbench & Document Repository Controller
 */

const WORKBENCH_STORAGE_KEY = 'graphrag_workbench_state_v1';

window.WorkbenchState = {
  documents: [],
  selectedDocIds: new Set(),
  workspaces: [{ id: 'all', name: 'All Files', docIds: [] }],
  activeWorkspaceId: 'all',
  searchQuery: ''
};

function saveWorkbenchStateToStorage() {
  try {
    if (!window.WorkbenchState) return;
    const data = {
      workspaces: window.WorkbenchState.workspaces,
      activeWorkspaceId: window.WorkbenchState.activeWorkspaceId,
      selectedDocIds: Array.from(window.WorkbenchState.selectedDocIds || [])
    };
    localStorage.setItem(WORKBENCH_STORAGE_KEY, JSON.stringify(data));
  } catch (err) {
    console.error('Failed to save WorkbenchState to localStorage:', err);
  }
}

function loadWorkbenchStateFromStorage() {
  try {
    const raw = localStorage.getItem(WORKBENCH_STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.workspaces && Array.isArray(data.workspaces) && data.workspaces.length > 0) {
      window.WorkbenchState.workspaces = data.workspaces;
    }
    if (data.activeWorkspaceId) {
      window.WorkbenchState.activeWorkspaceId = data.activeWorkspaceId;
    }
    if (data.selectedDocIds && Array.isArray(data.selectedDocIds)) {
      window.WorkbenchState.selectedDocIds = new Set(data.selectedDocIds);
    }
  } catch (err) {
    console.error('Failed to load WorkbenchState from localStorage:', err);
  }
}

window.initWorkbench = function() {
  loadWorkbenchStateFromStorage();
  loadDocumentLibrary();
  initSelectAllAndClear();
  initWorkspaceControls();
  initFileSearch();
};

window.getActiveSelectedDocIds = function() {
  if (!window.WorkbenchState) return null;

  // Priority 1: Checkbox selections in file list
  if (window.WorkbenchState.selectedDocIds && window.WorkbenchState.selectedDocIds.size > 0) {
    return Array.from(window.WorkbenchState.selectedDocIds);
  }

  // Priority 2: Active Workspace selection (if not 'all')
  const activeWsId = window.WorkbenchState.activeWorkspaceId;
  if (activeWsId && activeWsId !== 'all') {
    const ws = window.WorkbenchState.workspaces.find(w => w.id === activeWsId);
    if (ws && ws.docIds && ws.docIds.length > 0) {
      return ws.docIds;
    }
  }

  return null; // All Files / Global Corpus
};

window.loadDocumentLibrary = async function() {
  try {
    const res = await fetch('/api/v1/documents');
    if (!res.ok) return;
    const json = await res.json();

    let docs = [];
    if (Array.isArray(json)) {
      docs = json;
    } else if (json.data && Array.isArray(json.data.items)) {
      docs = json.data.items;
    } else if (json.data && Array.isArray(json.data)) {
      docs = json.data;
    } else if (json.items && Array.isArray(json.items)) {
      docs = json.items;
    } else if (json.documents && Array.isArray(json.documents)) {
      docs = json.documents;
    }

    window.WorkbenchState.documents = docs;

    renderDocumentTable();
    renderWorkspaces();
    renderFileList();
    updateScopeIndicators();
  } catch (err) {
    console.error('Failed to load document library:', err);
  }
};

function getFilteredDocuments() {
  let docs = window.WorkbenchState.documents || [];
  const q = (window.WorkbenchState.searchQuery || '').toLowerCase().trim();
  if (q) {
    docs = docs.filter(d => (d.filename || '').toLowerCase().includes(q));
  }
  return docs;
}

function initSelectAllAndClear() {
  const selectAllBtn = document.getElementById('wb-select-all-btn');
  const clearBtn = document.getElementById('wb-clear-select-btn');

  if (selectAllBtn) {
    selectAllBtn.onclick = () => {
      const visibleDocs = getFilteredDocuments();
      visibleDocs.forEach(d => window.WorkbenchState.selectedDocIds.add(d.id));

      const activeWs = window.WorkbenchState.workspaces.find(w => w.id === window.WorkbenchState.activeWorkspaceId);
      if (activeWs && activeWs.id !== 'all') {
        activeWs.docIds = Array.from(window.WorkbenchState.selectedDocIds);
      }

      renderWorkspaces();
      renderFileList();
      updateScopeIndicators();
    };
  }

  if (clearBtn) {
    clearBtn.onclick = () => {
      window.WorkbenchState.selectedDocIds.clear();

      const activeWs = window.WorkbenchState.workspaces.find(w => w.id === window.WorkbenchState.activeWorkspaceId);
      if (activeWs && activeWs.id !== 'all') {
        activeWs.docIds = [];
      }

      renderWorkspaces();
      renderFileList();
      updateScopeIndicators();
    };
  }
}

function initWorkspaceControls() {
  const newWsBtn = document.getElementById('wb-assign-ws-btn');
  const editWsBtn = document.getElementById('wb-edit-ws-btn');
  const saveWsBtn = document.getElementById('ws-modal-save-btn');
  const modalInput = document.getElementById('ws-modal-input');

  if (newWsBtn) {
    newWsBtn.onclick = () => {
      openWorkspaceModal();
    };
  }

  if (saveWsBtn) {
    saveWsBtn.onclick = () => {
      saveNewWorkspace();
    };
  }

  if (modalInput) {
    modalInput.onkeydown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        saveNewWorkspace();
      }
    };
  }

  if (editWsBtn) {
    editWsBtn.onclick = () => {
      deleteActiveWorkspace();
    };
  }
}

function openWorkspaceModal() {
  const modal = document.getElementById('ws-modal');
  const input = document.getElementById('ws-modal-input');
  const info = document.getElementById('ws-modal-info');
  if (!modal) return;

  const selCount = window.WorkbenchState.selectedDocIds.size;
  if (info) {
    info.textContent = selCount > 0
      ? `Assigning ${selCount} currently selected document(s) to this workspace.`
      : 'No documents currently selected. Workspace will start empty.';
  }
  if (input) input.value = '';
  modal.classList.add('open');
  if (input) setTimeout(() => input.focus(), 100);
}

function saveNewWorkspace() {
  const input = document.getElementById('ws-modal-input');
  const modal = document.getElementById('ws-modal');
  if (!input) return;

  const wsName = input.value.trim();
  if (!wsName) {
    alert('Please enter a workspace name.');
    return;
  }

  const newWs = {
    id: 'ws_' + Date.now(),
    name: wsName,
    docIds: Array.from(window.WorkbenchState.selectedDocIds)
  };

  window.WorkbenchState.workspaces.push(newWs);
  window.WorkbenchState.activeWorkspaceId = newWs.id;

  if (modal) modal.classList.remove('open');

  renderWorkspaces();
  renderFileList();
  updateScopeIndicators();
}

function deleteActiveWorkspace() {
  const activeWsId = window.WorkbenchState.activeWorkspaceId;
  if (activeWsId === 'all') return;

  const activeWs = window.WorkbenchState.workspaces.find(w => w.id === activeWsId);
  if (!activeWs) return;

  if (confirm(`Delete workspace "${activeWs.name}"? (Files will remain in repository)`)) {
    window.WorkbenchState.workspaces = window.WorkbenchState.workspaces.filter(w => w.id !== activeWsId);
    window.WorkbenchState.activeWorkspaceId = 'all';
    window.WorkbenchState.selectedDocIds.clear();

    renderWorkspaces();
    renderFileList();
    updateScopeIndicators();
  }
}

function initFileSearch() {
  const searchInput = document.getElementById('workbench-file-search');
  if (searchInput) {
    searchInput.oninput = (e) => {
      window.WorkbenchState.searchQuery = e.target.value;
      renderFileList();
    };
  }
}

// Render Document Library Table
function renderDocumentTable() {
  const tbody = document.getElementById('doc-library-tbody');
  if (!tbody) return;

  const docs = window.WorkbenchState.documents;
  if (!docs || docs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:2rem;color:var(--text-muted);">No documents uploaded yet. Drag &amp; drop files in Ingestion &amp; Upload.</td></tr>`;
    return;
  }

  tbody.innerHTML = '';
  docs.forEach(doc => {
    const tr = document.createElement('tr');
    const typeExt = (doc.filename || doc.file_type || 'PDF').split('.').pop().toUpperCase();
    const created = doc.created_at ? new Date(doc.created_at).toLocaleString() : 'N/A';

    tr.innerHTML = `
      <td style="font-weight:600;color:var(--text-primary);">${doc.filename}</td>
      <td><span class="scope-pill" style="font-size:0.7rem;">${typeExt}</span></td>
      <td style="font-family:var(--font-mono);">${doc.chunk_count || 0}</td>
      <td style="color:var(--text-secondary);font-size:0.8rem;">${created}</td>
      <td style="text-align:right;">
        <button class="btn btn-secondary btn-sm" onclick="openChunkViewerModal('${doc.id}', '${doc.filename}')">View Chunks</button>
        <button class="btn btn-danger btn-sm" onclick="deleteDocument('${doc.id}')">Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Chunk Inspector Modal Popup Trigger (GET /api/v1/document/{id})
window.openChunkViewerModal = async function(docId, filename) {
  const modal = document.getElementById('chunk-modal');
  const title = document.getElementById('chunk-modal-title');
  const body = document.getElementById('chunk-modal-body');

  if (!modal || !title || !body) return;

  title.textContent = `Chunks — ${filename}`;
  body.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-muted);">Fetching text chunks...</div>';
  modal.classList.add('open');

  try {
    const res = await fetch(`/api/v1/document/${docId}`);
    if (!res.ok) {
      body.innerHTML = '<div style="color:var(--danger);padding:1rem;">Failed to load document chunks.</div>';
      return;
    }
    const json = await res.json();
    const data = json.data || json;
    const chunks = Array.isArray(data.chunks) ? data.chunks : (Array.isArray(data) ? data : []);

    if (chunks.length === 0) {
      body.innerHTML = '<div style="padding:1rem;color:var(--text-muted);">No chunks found for this document.</div>';
      return;
    }

    body.innerHTML = '';
    chunks.forEach((c, idx) => {
      const card = document.createElement('div');
      card.className = 'glass-card';
      card.style.marginBottom = '0.75rem';
      card.style.padding = '1rem';
      card.innerHTML = `
        <div style="font-size:0.75rem;font-weight:700;color:var(--accent-primary);margin-bottom:0.35rem;">Chunk #${idx + 1} &bull; Tokens: ${c.token_count || c.word_count || 'N/A'}</div>
        <div style="font-family:var(--font-mono);font-size:0.82rem;color:#e2e8f0;white-space:pre-wrap;line-height:1.5;">${c.text || c.content || c.chunk_text || ''}</div>
      `;
      body.appendChild(card);
    });
  } catch (e) {
    body.innerHTML = '<div style="color:var(--danger);padding:1rem;">Error fetching chunk data.</div>';
  }
};

// Delete Document (DELETE /api/v1/delete/{id})
window.deleteDocument = async function(docId) {
  if (!confirm('Are you sure you want to delete this document?')) return;
  try {
    const res = await fetch(`/api/v1/delete/${docId}`, { method: 'DELETE' });
    if (res.ok) {
      loadDocumentLibrary();
      if (window.fetchSystemStats) window.fetchSystemStats();
      if (window.fetchGraphData) window.fetchGraphData();
    } else {
      alert('Failed to delete document.');
    }
  } catch (e) {
    alert('Error deleting document.');
  }
};

// Workspace Pills & File List
function renderWorkspaces() {
  const container = document.getElementById('workspace-pills-container');
  const editWsBtn = document.getElementById('wb-edit-ws-btn');
  if (!container) return;

  container.innerHTML = '';
  window.WorkbenchState.workspaces.forEach(ws => {
    const pill = document.createElement('div');
    const isActive = ws.id === window.WorkbenchState.activeWorkspaceId;
    pill.className = `workspace-pill ${isActive ? 'active' : ''}`;
    pill.textContent = `${ws.name} (${ws.id === 'all' ? window.WorkbenchState.documents.length : ws.docIds.length})`;
    pill.onclick = () => {
      window.WorkbenchState.activeWorkspaceId = ws.id;

      if (ws.id === 'all') {
        window.WorkbenchState.selectedDocIds.clear();
      } else {
        window.WorkbenchState.selectedDocIds = new Set(ws.docIds || []);
      }

      renderWorkspaces();
      renderFileList();
      updateScopeIndicators();
    };
    container.appendChild(pill);
  });

  if (editWsBtn) {
    if (window.WorkbenchState.activeWorkspaceId !== 'all') {
      editWsBtn.style.display = 'inline-flex';
      editWsBtn.textContent = 'Delete Workspace';
    } else {
      editWsBtn.style.display = 'none';
    }
  }
}

function renderFileList() {
  const listContainer = document.getElementById('workbench-file-list');
  if (!listContainer) return;

  const docs = getFilteredDocuments();
  if (!docs || docs.length === 0) {
    listContainer.innerHTML = '<div style="padding:1.5rem;text-align:center;color:var(--text-muted);font-size:0.8rem;">No files matching criteria.</div>';
    return;
  }

  listContainer.innerHTML = '';
  docs.forEach(doc => {
    const isSelected = window.WorkbenchState.selectedDocIds.has(doc.id);
    const row = document.createElement('div');
    row.className = `file-item-row ${isSelected ? 'selected' : ''}`;

    row.innerHTML = `
      <input type="checkbox" class="file-checkbox" ${isSelected ? 'checked' : ''} />
      <div class="file-info" style="flex:1;min-width:0;">
        <div class="file-name" title="${doc.filename}">${doc.filename}</div>
        <div class="file-sub">
          <span>${(doc.file_type || 'PDF').toUpperCase()}</span> &bull;
          <span>${doc.chunk_count || 0} chunks</span>
        </div>
      </div>
      <button class="btn btn-secondary btn-sm chunk-view-btn" style="font-size:0.72rem;padding:2px 8px;margin-left:0.5rem;" title="View text chunks">Chunks</button>
    `;

    const toggleSelection = () => {
      if (window.WorkbenchState.selectedDocIds.has(doc.id)) {
        window.WorkbenchState.selectedDocIds.delete(doc.id);
      } else {
        window.WorkbenchState.selectedDocIds.add(doc.id);
      }

      const activeWs = window.WorkbenchState.workspaces.find(w => w.id === window.WorkbenchState.activeWorkspaceId);
      if (activeWs && activeWs.id !== 'all') {
        activeWs.docIds = Array.from(window.WorkbenchState.selectedDocIds);
      }

      renderWorkspaces();
      renderFileList();
      updateScopeIndicators();
    };

    const checkbox = row.querySelector('.file-checkbox');
    if (checkbox) {
      checkbox.onclick = (e) => {
        e.stopPropagation();
        toggleSelection();
      };
    }

    const chunkBtn = row.querySelector('.chunk-view-btn');
    if (chunkBtn) {
      chunkBtn.onclick = (e) => {
        e.stopPropagation();
        openChunkViewerModal(doc.id, doc.filename);
      };
    }

    row.onclick = () => {
      toggleSelection();
    };

    listContainer.appendChild(row);
  });
}

function updateScopeIndicators() {
  const topPill = document.getElementById('top-scope-pill');
  const activeDocIds = window.getActiveSelectedDocIds();

  if (topPill) {
    if (activeDocIds && activeDocIds.length > 0) {
      topPill.textContent = `Scope: ${activeDocIds.length} File(s) Isolated`;
    } else {
      topPill.textContent = 'Scope: Global Corpus';
    }
  }

  // Persist current WorkbenchState (workspaces, active workspace ID, selectedDocIds) to localStorage
  saveWorkbenchStateToStorage();

  // Automatically refresh Knowledge Graph for the active isolated scope!
  if (window.fetchGraphData) {
    window.fetchGraphData();
  }
}
