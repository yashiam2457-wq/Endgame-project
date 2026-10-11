/**
 * Enterprise GraphRAG — Premium Knowledge Graph Visualizer
 * Palantir / Neo4j Bloom / Microsoft Fabric aesthetic
 * Pure frontend redesign — zero backend modifications
 */

'use strict';

// ─── Module State ──────────────────────────────────────────────────────────
let cyInstance = null;
let currentStats = null;
let activeFilterType = 'ALL';
let isLoading = false;

// ─── Enterprise Color System ────────────────────────────────────────────────
// Muted, professional, information-dense palette
const ENTITY_PALETTE = {
  PERSON:       { accent: '#4ade80', bg: '#052e16', border: '#166534', text: '#bbf7d0', icon: '👤' },
  ORGANIZATION: { accent: '#a78bfa', bg: '#1e1b4b', border: '#4338ca', text: '#ddd6fe', icon: '🏢' },
  COMPANY:      { accent: '#a78bfa', bg: '#1e1b4b', border: '#4338ca', text: '#ddd6fe', icon: '🏢' },
  CONCEPT:      { accent: '#fb923c', bg: '#2c1501', border: '#9a3412', text: '#fed7aa', icon: '💡' },
  POLICY:       { accent: '#fb923c', bg: '#2c1501', border: '#9a3412', text: '#fed7aa', icon: '📋' },
  TECHNOLOGY:   { accent: '#22d3ee', bg: '#012a2e', border: '#0e7490', text: '#a5f3fc', icon: '⚙️' },
  PRODUCT:      { accent: '#22d3ee', bg: '#012a2e', border: '#0e7490', text: '#a5f3fc', icon: '📦' },
  LOCATION:     { accent: '#f87171', bg: '#2c0a0a', border: '#991b1b', text: '#fecaca', icon: '📍' },
  EVENT:        { accent: '#f87171', bg: '#2c0a0a', border: '#991b1b', text: '#fecaca', icon: '📅' },
  DOCUMENT:     { accent: '#60a5fa', bg: '#0c1a2e', border: '#1d4ed8', text: '#bfdbfe', icon: '📄' },
  FILE:         { accent: '#60a5fa', bg: '#0c1a2e', border: '#1d4ed8', text: '#bfdbfe', icon: '📄' },
  PROCESS:      { accent: '#fb7185', bg: '#2c0d14', border: '#9f1239', text: '#fecdd3', icon: '⚡' },
  UNKNOWN:      { accent: '#94a3b8', bg: '#0f172a', border: '#334155', text: '#cbd5e1', icon: '◉' },
};

function getPalette(type) {
  const key = String(type || 'UNKNOWN').toUpperCase().trim();
  return ENTITY_PALETTE[key] || ENTITY_PALETTE.UNKNOWN;
}

// ─── Init ────────────────────────────────────────────────────────────────────
window.initGraph = function () {
  initToolbar();
  initFilterPills();
  fetchGraphData();
};

// ─── Toolbar Buttons ──────────────────────────────────────────────────────────
function initToolbar() {
  const searchBtn = document.getElementById('btn-graph-search');
  const fitBtn    = document.getElementById('btn-graph-fit');
  const resetBtn  = document.getElementById('btn-graph-reset');

  if (searchBtn) searchBtn.onclick = runSearch;
  if (fitBtn)    fitBtn.onclick    = () => cyInstance && cyInstance.fit(undefined, 60);
  if (resetBtn)  resetBtn.onclick  = relayout;

  const input = document.getElementById('graph-entity-input');
  if (input) {
    input.onkeydown = e => { if (e.key === 'Enter') runSearch(); };
    input.oninput = () => {
      const q = input.value.trim().toLowerCase();
      if (!cyInstance || !q) { cyInstance && cyInstance.elements().removeClass('gh-dim gh-match'); return; }
      cyInstance.batch(() => {
        cyInstance.elements().addClass('gh-dim').removeClass('gh-match');
        cyInstance.nodes().forEach(n => {
          if ((n.data('label') || '').toLowerCase().includes(q)) {
            n.removeClass('gh-dim').addClass('gh-match');
            n.connectedEdges().removeClass('gh-dim');
          }
        });
      });
    };
  }
}

function runSearch() {
  const input = document.getElementById('graph-entity-input');
  const q = input ? input.value.trim().toLowerCase() : '';
  if (!cyInstance || !q) return;

  let first = null;
  cyInstance.batch(() => {
    cyInstance.nodes().forEach(n => {
      if ((n.data('label') || '').toLowerCase().includes(q)) {
        n.select();
        if (!first) first = n;
      } else { n.unselect(); }
    });
  });
  if (first) {
    cyInstance.animate({ center: { eles: first }, zoom: 1.6 }, { duration: 350, easing: 'ease-in-out-cubic' });
    inspectNode(first);
  }
}

/**
 * Component-Separated Relationship Subgraph Cluster Layout Engine
 * Form a visual cluster box for each connected graph with 70px (>50px) margin spacing.
 */
function runComponentSeparatedLayout(cy) {
  if (!cy) return;
  const visibleNodes = cy.nodes(':visible').not(':parent');
  if (visibleNodes.length === 0) return;

  // 1. Find Connected Components via BFS
  const visited = new Set();
  const components = [];

  visibleNodes.forEach(node => {
    const id = node.id();
    if (visited.has(id)) return;

    const component = [];
    const queue = [node];
    visited.add(id);

    while (queue.length > 0) {
      const curr = queue.shift();
      component.push(curr);

      const neighbors = curr.neighborhood('node').not(':parent');
      neighbors.forEach(nb => {
        if (nb.visible() && !visited.has(nb.id())) {
          visited.add(nb.id());
          queue.push(nb);
        }
      });
    }

    components.push(component);
  });

  // Sort components by node count (connected relationship clusters first)
  components.sort((a, b) => b.length - a.length);

  const connectedClusters = components.filter(c => c.length > 1);
  const isolatedNodes = components.filter(c => c.length === 1).map(c => c[0]);

  const cols = Math.min(3, Math.max(2, Math.ceil(Math.sqrt(connectedClusters.length))));
  const clusterMarginX = 75; // 75px margin (>50px) side-by-side between clusters
  const clusterMarginY = 75; // 75px margin (>50px) between cluster rows

  const verticalEntityMargin = 55; // 55px (>50px) vertical margin between sibling entities
  const relationshipEdgeSpace = 240; // 240px (>200px) horizontal relationship edge space

  cy.batch(() => {
    let currentY = 0;

    // 2. Tile Connected Component Clusters Side-by-Side in Multi-Column Grid Rows
    for (let i = 0; i < connectedClusters.length; i += cols) {
      const rowClusters = connectedClusters.slice(i, i + cols);
      let rowMaxHeight = 0;
      let currentX = 0;

      rowClusters.forEach(cluster => {
        cluster.sort((a, b) => b.degree() - a.degree());
        const root = cluster[0];

        // Build BFS levels from root
        const levels = [];
        const levelVisited = new Set([root.id()]);
        let currentLevel = [root];

        while (currentLevel.length > 0) {
          levels.push(currentLevel);
          const nextLevel = [];

          currentLevel.forEach(parent => {
            const neighbors = parent.neighborhood('node').not(':parent');
            neighbors.forEach(nb => {
              if (nb.visible() && !levelVisited.has(nb.id())) {
                levelVisited.add(nb.id());
                nextLevel.push(nb);
              }
            });
          });

          currentLevel = nextLevel;
        }

        // Level X step = average card width (160px) + 240px relationship edge space (>200px) = 400px
        const levelGapX = 160 + relationshipEdgeSpace;
        let clusterWidth = Math.max(320, (levels.length - 1) * levelGapX + 240);
        let maxLvlY = 0;

        levels.forEach(lvlNodes => {
          let lvlY = 0;
          lvlNodes.forEach(n => {
            const label = String(n.data('label') || '');
            const h = label.length > 18 ? 44 : 36;
            lvlY += h + verticalEntityMargin; // 55px vertical gap (>50px)
          });
          if (lvlY > maxLvlY) maxLvlY = lvlY;
        });

        let clusterHeight = Math.max(140, maxLvlY + 50);

        // Position nodes inside this cluster with 240px (>200px) relationship edge space
        levels.forEach((lvlNodes, lvlIdx) => {
          const x = currentX + lvlIdx * levelGapX + 35;
          let lvlY = 0;

          lvlNodes.forEach(n => {
            const label = String(n.data('label') || '');
            const h = label.length > 18 ? 44 : 36;
            n.position({ x: x, y: currentY + lvlY + 40 });
            lvlY += h + verticalEntityMargin; // 55px vertical gap (>50px) between entities
          });
        });

        if (clusterHeight > rowMaxHeight) rowMaxHeight = clusterHeight;

        // Advance currentX side-by-side past cluster width + 75px margin (> 50px requirement)
        currentX += clusterWidth + clusterMarginX;
      });

      // Advance currentY to next row + 75px margin (> 50px requirement)
      currentY += rowMaxHeight + clusterMarginY;
    }

    // 3. Position Isolated Single Nodes in a clean multi-column grid at bottom
    if (isolatedNodes.length > 0) {
      currentY += 30;
      const isoCols = 6;
      let currentIsoX = 0;
      let rowMaxH = 0;
      let colIdx = 0;

      isolatedNodes.forEach(n => {
        const label = String(n.data('label') || '');
        const w = Math.max(130, Math.min(label.length * 8.5 + 32, 230));
        const h = label.length > 18 ? 44 : 36;

        n.position({ x: currentIsoX + w / 2, y: currentY + h / 2 });

        if (h > rowMaxH) rowMaxH = h;

        colIdx++;
        if (colIdx >= isoCols) {
          colIdx = 0;
          currentIsoX = 0;
          currentY += rowMaxH + verticalEntityMargin; // 55px gap (>50px)
          rowMaxH = 0;
        } else {
          currentIsoX += w + 55; // 55px gap (>50px)
        }
      });
    }
  });

  cy.zoom(1.0);
  cy.center();
}

function relayout() {
  if (!cyInstance) return;
  runComponentSeparatedLayout(cyInstance);
}



// ─── Filter Pills ──────────────────────────────────────────────────────────
function initFilterPills() {
  document.querySelectorAll('#graph-filter-bar .graph-filter-pill').forEach(pill => {
    pill.onclick = () => {
      document.querySelectorAll('#graph-filter-bar .graph-filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      activeFilterType = pill.getAttribute('data-type') || 'ALL';
      applyFilter(activeFilterType);
    };
  });
}

function applyFilter(type) {
  if (!cyInstance) return;
  cyInstance.batch(() => {
    if (type === 'ALL') {
      cyInstance.elements().style('display', 'element');
    } else {
      cyInstance.nodes().forEach(n => {
        const t = (n.data('type') || 'UNKNOWN').toUpperCase();
        let match = (t === type);
        if (type === 'ORGANIZATION') match = (t === 'ORGANIZATION' || t === 'COMPANY');
        if (type === 'CONCEPT')      match = (t === 'CONCEPT' || t === 'POLICY');
        if (type === 'TECHNOLOGY')   match = (t === 'TECHNOLOGY' || t === 'PRODUCT');
        if (type === 'LOCATION')     match = (t === 'LOCATION' || t === 'EVENT');
        n.style('display', match ? 'element' : 'none');
        n.connectedEdges().style('display', match ? 'element' : 'none');
      });
    }
  });
  updateStatsBar();
  runComponentSeparatedLayout(cyInstance);
}

// ─── API Call ───────────────────────────────────────────────────────────────
window.fetchGraphData = async function (entityName = '', docIds = null) {
  const container = document.getElementById('cy-canvas');
  if (!container || isLoading) return;
  isLoading = true;
  showLoadingState(true);

  try {
    const activeDocIds = docIds || (window.getActiveSelectedDocIds ? window.getActiveSelectedDocIds() : null);
    const res = await fetch('/api/v1/graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_name: entityName,
        document_ids: activeDocIds,
        hop_depth: 2,
        limit: 150,
      }),
    });

    if (!res.ok) { showEmptyState(); return; }
    const json = await res.json();
    const data = json.data || json;
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    currentStats = data.stats || null;

    updateStatsBar(nodes, edges);
    if (!nodes.length) { showEmptyState(); return; }

    showLoadingState(false);
    renderGraph(nodes, edges);
  } catch (err) {
    console.error('[GraphRAG] Graph fetch error:', err);
    showEmptyState();
  } finally {
    isLoading = false;
  }
};

// ─── Stats Bar ──────────────────────────────────────────────────────────────
function updateStatsBar(nodes, edges) {
  const n = nodes ? nodes.length : (cyInstance ? cyInstance.nodes(':visible').length : 0);
  const e = edges ? edges.length : (cyInstance ? cyInstance.edges(':visible').length : 0);
  const deg = n > 0 ? ((e * 2) / n).toFixed(2) : '0.00';
  const density = n > 1 ? ((2 * e) / (n * (n - 1))).toFixed(4) : '0.0000';

  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setEl('graph-stat-nodes', n);
  setEl('graph-stat-edges', e);
  setEl('graph-stat-degree', deg);
  setEl('graph-stat-density', density);
  setEl('graph-stat-components', currentStats?.connected_components ?? '—');
  setEl('graph-stat-tophub',
    currentStats?.top_entities?.length
      ? `${currentStats.top_entities[0].name || currentStats.top_entities[0].id} (${currentStats.top_entities[0].degree})`
      : '—'
  );

  // Live zoom stat
  if (cyInstance) {
    setEl('graph-stat-zoom', Math.round(cyInstance.zoom() * 100) + '%');
  }
}

// ─── Cytoscape Render ────────────────────────────────────────────────────────
function renderGraph(nodes, edges) {
  const container = document.getElementById('cy-canvas');
  if (!container) return;

  if (cyInstance) { try { cyInstance.destroy(); } catch (_) {} }

  // Build node lookup and adjacency map to identify connected clusters
  const nodeMap = new Map();
  nodes.forEach(n => nodeMap.set(n.data.id, n));

  const adj = new Map();
  nodes.forEach(n => adj.set(n.data.id, []));
  edges.forEach(e => {
    if (adj.has(e.data.source) && adj.has(e.data.target)) {
      adj.get(e.data.source).push(e.data.target);
      adj.get(e.data.target).push(e.data.source);
    }
  });

  const visited = new Set();
  const clusters = [];

  nodes.forEach(n => {
    const id = n.data.id;
    if (visited.has(id)) return;

    const clusterNodes = [];
    const queue = [id];
    visited.add(id);

    while (queue.length > 0) {
      const curr = queue.shift();
      clusterNodes.push(curr);
      (adj.get(curr) || []).forEach(neighbor => {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      });
    }

    clusters.push(clusterNodes);
  });

  clusters.sort((a, b) => b.length - a.length);

  // Generate parent compound node elements for each connected cluster
  const parentClusterElements = [];
  let clusterIdx = 1;

  clusters.forEach(clusterNodeIds => {
    if (clusterNodeIds.length > 1) {
      const parentId = `cluster_box_${clusterIdx}`;
      const hubId = clusterNodeIds.reduce((maxId, currId) => {
        const degree = (adj.get(currId) || []).length;
        const maxDegree = (adj.get(maxId) || []).length;
        return degree > maxDegree ? currId : maxId;
      }, clusterNodeIds[0]);

      const hubNode = nodeMap.get(hubId);
      const hubLabel = hubNode ? (hubNode.data.label || hubNode.data.id) : `Group ${clusterIdx}`;

      parentClusterElements.push({
        data: {
          id: parentId,
          label: `Cluster ${clusterIdx}: ${hubLabel} (${clusterNodeIds.length} entities)`
        },
        classes: 'graph-cluster-box'
      });

      clusterNodeIds.forEach(id => {
        const n = nodeMap.get(id);
        if (n) {
          n.data.parent = parentId;
        }
      });

      clusterIdx++;
    }
  });

  cyInstance = cytoscape({
    container,
    elements: [...parentClusterElements, ...nodes, ...edges],
    boxSelectionEnabled: true,
    autounselectify: false,
    userZoomingEnabled: true,
    userPanningEnabled: true,
    style: buildCyStyle(),
  });

  runComponentSeparatedLayout(cyInstance);

  bindCyEvents();
  buildLegend();
}

// ─── Cytoscape Style ─────────────────────────────────────────────────────────
function buildCyStyle() {
  return [
    // ── Compound Parent Cluster Box ─────────────────────────────────────────
    {
      selector: 'node:parent',
      style: {
        'shape': 'round-rectangle',
        'background-color': '#0f172a',
        'background-opacity': 0.35,
        'border-color': '#818cf8',
        'border-width': 1.5,
        'border-style': 'dashed',
        'border-opacity': 0.5,
        'padding': '35px',
        'label': 'data(label)',
        'color': '#a5b4fc',
        'font-family': '"Inter", system-ui, sans-serif',
        'font-size': '12px',
        'font-weight': '700',
        'text-valign': 'top',
        'text-halign': 'left',
        'text-margin-y': -10,
        'text-opacity': 0, // Hidden by default!
        'text-background-opacity': 0,
        'transition-property': 'text-opacity, text-background-opacity, border-opacity, border-width',
        'transition-duration': '180ms',
      },
    },
    {
      selector: 'node:parent.gh-hover',
      style: {
        'text-opacity': 1, // Shown on hover!
        'text-background-color': '#070d1a',
        'text-background-opacity': 0.95,
        'text-background-padding': '4px',
        'text-background-shape': 'round-rectangle',
        'border-opacity': 0.9,
        'border-width': 2,
      },
    },
    // ── Base Node ──────────────────────────────────────────────────────────
    {
      selector: 'node',
      style: {
        'shape': 'round-rectangle',
        'background-color': ele => getPalette(ele.data('type')).bg,
        'border-color':     ele => getPalette(ele.data('type')).accent,
        'border-width': 1.5,
        'border-opacity': 0.85,

        'label': 'data(label)',
        'color': ele => getPalette(ele.data('type')).text,
        'font-family': '"Inter", "IBM Plex Sans", system-ui, sans-serif',
        'font-size': '13px',
        'font-weight': '600',
        'text-valign': 'center',
        'text-halign': 'center',
        'text-wrap': 'wrap',
        'text-max-width': ele => {
          const l = String(ele.data('label') || '');
          return Math.max(100, Math.min(l.length * 8.5 + 28, 200)) + 'px';
        },
        'width': ele => {
          const l = String(ele.data('label') || '');
          return Math.max(120, Math.min(l.length * 8.5 + 28, 220)) + 'px';
        },
        'height': ele => {
          const l = String(ele.data('label') || '');
          return l.length > 18 ? '44px' : '36px';
        },
        'padding': '10px',
        'shadow-blur': 8,
        'shadow-color': 'rgba(0,0,0,0.45)',
        'shadow-offset-x': 0,
        'shadow-offset-y': 2,
        'transition-property': 'background-color, border-color, border-width, opacity, width, height, shadow-blur',
        'transition-duration': '180ms',
      },
    },
    // ── Node Hover (via class) ─────────────────────────────────────────────
    {
      selector: 'node.gh-hover',
      style: {
        'border-width': 2.5,
        'border-color': ele => getPalette(ele.data('type')).accent,
        'shadow-blur': 22,
        'shadow-color': ele => getPalette(ele.data('type')).accent + '66',
        'z-index': 999,
      },
    },
    // ── Node Selected ──────────────────────────────────────────────────────
    {
      selector: 'node:selected',
      style: {
        'border-color': '#f8fafc',
        'border-width': 2.5,
        'shadow-blur': 20,
        'shadow-color': 'rgba(248,250,252,0.25)',
      },
    },
    // ── Dim State ──────────────────────────────────────────────────────────
    {
      selector: 'node.gh-dim',
      style: { 'opacity': 0.12 },
    },
    {
      selector: 'edge.gh-dim',
      style: { 'opacity': 0.04 },
    },
    // ── Search Match ──────────────────────────────────────────────────────
    {
      selector: 'node.gh-match',
      style: {
        'border-color': '#fbbf24',
        'border-width': 2.5,
        'shadow-blur': 14,
        'shadow-color': 'rgba(251,191,36,0.4)',
      },
    },
    // ── Neighbor Highlight ────────────────────────────────────────────────
    {
      selector: 'node.gh-neighbor',
      style: {
        'border-width': 2,
        'opacity': 1,
      },
    },
    // ── Base Edge ──────────────────────────────────────────────────────────
    {
      selector: 'edge',
      style: {
        'width': ele => Math.min(1.5 + (ele.data('weight') || 1) * 0.3, 3.5),
        'line-color': 'rgba(148,163,184,0.22)',
        'target-arrow-color': 'rgba(148,163,184,0.35)',
        'target-arrow-shape': 'vee',
        'arrow-scale': 0.85,
        'curve-style': 'bezier',

        'label': 'data(label)',
        'font-family': '"Inter", system-ui, sans-serif',
        'font-size': '10px',
        'font-weight': '600',
        'color': '#94a3b8',
        'text-rotation': 'none',
        'text-background-color': '#070d1a',
        'text-background-opacity': 0.92,
        'text-background-padding': '3px',
        'text-background-shape': 'round-rectangle',
        'text-border-color': 'rgba(148,163,184,0.2)',
        'text-border-width': 1,
        'text-border-opacity': 0.8,
        'transition-property': 'line-color, target-arrow-color, width, opacity',
        'transition-duration': '180ms',
      },
    },
    // ── Edge Hover Active ─────────────────────────────────────────────────
    {
      selector: 'edge.gh-active',
      style: {
        'line-color': 'rgba(148,163,184,0.6)',
        'target-arrow-color': 'rgba(148,163,184,0.8)',
        'width': 2.5,
        'color': '#e2e8f0',
        'z-index': 99,
      },
    },
    // ── Edge Selected ─────────────────────────────────────────────────────
    {
      selector: 'edge:selected',
      style: {
        'line-color': '#818cf8',
        'target-arrow-color': '#818cf8',
        'width': 3,
        'color': '#c7d2fe',
        'text-border-color': '#818cf8',
        'z-index': 999,
      },
    },
  ];
}

// ─── Events ──────────────────────────────────────────────────────────────────
function bindCyEvents() {
  // Hover enter — highlight neighbors, dim others, reveal cluster title if parent
  cyInstance.on('mouseover', 'node', evt => {
    const n = evt.target;
    const neighborhood = n.closedNeighborhood();
    const parentNode = n.isParent() ? n : (n.parent ? n.parent() : null);

    cyInstance.batch(() => {
      cyInstance.elements().addClass('gh-dim').removeClass('gh-hover gh-neighbor');
      neighborhood.removeClass('gh-dim');
      n.addClass('gh-hover').removeClass('gh-dim');
      n.connectedEdges().addClass('gh-active').removeClass('gh-dim');
      n.neighborhood('node').addClass('gh-neighbor').removeClass('gh-dim');
      if (parentNode && parentNode.length) {
        parentNode.addClass('gh-hover').removeClass('gh-dim');
      }
    });
  });

  // Hover leave
  cyInstance.on('mouseout', 'node', () => {
    cyInstance.batch(() => {
      cyInstance.elements().removeClass('gh-dim gh-hover gh-neighbor gh-active');
    });
  });

  // Tap node — open inspector
  cyInstance.on('tap', 'node', evt => inspectNode(evt.target));

  // Tap edge — open inspector
  cyInstance.on('tap', 'edge', evt => inspectEdge(evt.target));

  // Tap background — close inspector / clear selection
  cyInstance.on('tap', evt => {
    if (evt.target === cyInstance) {
      cyInstance.elements().removeClass('gh-dim gh-hover gh-neighbor gh-active gh-match');
      cyInstance.elements().unselect();
      hideInspector();
    }
  });

  // Double-tap node — expand neighborhood
  cyInstance.on('dbltap', 'node', evt => {
    const n = evt.target;
    cyInstance.animate({ fit: { eles: n.closedNeighborhood(), padding: 80 } }, { duration: 400 });
  });

  // Zoom — update stat + adjust label visibility
  cyInstance.on('zoom', () => updateStatsBar());
}

// ─── Inspector Panel ─────────────────────────────────────────────────────────
function showInspector() {
  const panel = document.getElementById('graph-inspector-panel');
  if (panel) { panel.style.display = 'flex'; panel.classList.add('visible'); }
}
function hideInspector() {
  const panel = document.getElementById('graph-inspector-panel');
  if (panel) { panel.style.display = 'none'; panel.classList.remove('visible'); }
}

function inspectNode(node) {
  const data   = node.data();
  const type   = (data.type || 'UNKNOWN').toUpperCase();
  const p      = getPalette(type);
  const degree = node.degree();
  const neighbors = node.neighborhood('node');

  const titleEl   = document.getElementById('graph-inspector-title');
  const contentEl = document.getElementById('graph-inspector-content');
  if (!titleEl || !contentEl) return;

  titleEl.innerHTML = `
    <span style="font-size:1rem;font-weight:700;color:${p.text};">${data.label || data.id}</span>
    <span class="gi-type-badge" style="background:${p.accent}22;color:${p.accent};border-color:${p.accent}44;">${type}</span>
  `;

  const neighborChips = Array.from(neighbors).slice(0, 8).map(nb => `
    <button class="gi-chip" onclick="window.graphFocusNode('${nb.id()}')">${nb.data('label') || nb.id()}</button>
  `).join('');

  contentEl.innerHTML = `
    <div class="gi-section">
      <div class="gi-row">
        <span class="gi-key">Degree</span>
        <span class="gi-val">${degree} connections</span>
      </div>
      <div class="gi-row">
        <span class="gi-key">Entity ID</span>
        <span class="gi-val gi-mono">${String(data.id).substring(0, 22)}…</span>
      </div>
      ${data.canonical_name ? `<div class="gi-row"><span class="gi-key">Canonical</span><span class="gi-val">${data.canonical_name}</span></div>` : ''}
      ${data.document_id ? `<div class="gi-row"><span class="gi-key">Source Doc</span><span class="gi-val gi-mono" style="font-size:0.72rem;">${String(data.document_id).substring(0, 20)}…</span></div>` : ''}
      ${data.confidence != null ? `<div class="gi-row"><span class="gi-key">Confidence</span><span class="gi-val">${(data.confidence * 100).toFixed(0)}%</span></div>` : ''}
    </div>

    ${neighbors.length ? `
    <div class="gi-section">
      <div class="gi-section-title">Connected (${neighbors.length})</div>
      <div class="gi-chip-list">${neighborChips}</div>
    </div>` : ''}

    <div class="gi-actions">
      <button class="btn btn-secondary btn-sm" onclick="window.graphFocusNode('${data.id}')">Center</button>
      <button class="btn btn-secondary btn-sm" onclick="navigator.clipboard.writeText('${data.id}')">Copy ID</button>
      <button class="btn btn-primary btn-sm" onclick="window.fetchGraphData('${(data.label || '').replace(/'/g, '')}')">Explore</button>
    </div>
  `;

  showInspector();
}

function inspectEdge(edge) {
  const data    = edge.data();
  const titleEl = document.getElementById('graph-inspector-title');
  const bodyEl  = document.getElementById('graph-inspector-content');
  if (!titleEl || !bodyEl) return;

  titleEl.innerHTML = `<span style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">Relationship</span>
    <span class="gi-type-badge" style="background:rgba(129,140,248,0.15);color:#818cf8;border-color:rgba(129,140,248,0.3);">
      ${data.label || 'RELATED_TO'}
    </span>`;

  bodyEl.innerHTML = `
    <div class="gi-section">
      <div class="gi-row"><span class="gi-key">Type</span><span class="gi-val">${data.label || 'RELATED_TO'}</span></div>
      <div class="gi-row"><span class="gi-key">Source</span><span class="gi-val gi-mono" style="font-size:0.72rem;">${String(data.source).substring(0, 24)}</span></div>
      <div class="gi-row"><span class="gi-key">Target</span><span class="gi-val gi-mono" style="font-size:0.72rem;">${String(data.target).substring(0, 24)}</span></div>
      ${data.weight != null ? `<div class="gi-row"><span class="gi-key">Weight</span><span class="gi-val">${data.weight}</span></div>` : ''}
      ${data.confidence != null ? `<div class="gi-row"><span class="gi-key">Confidence</span><span class="gi-val">${(data.confidence * 100).toFixed(0)}%</span></div>` : ''}
    </div>
    ${data.evidence_sentence ? `
    <div class="gi-section">
      <div class="gi-section-title">Evidence</div>
      <blockquote class="gi-evidence">"${data.evidence_sentence}"</blockquote>
    </div>` : ''}
    <div class="gi-actions">
      <button class="btn btn-secondary btn-sm" onclick="window.graphFocusNode('${data.source}')">Source</button>
      <button class="btn btn-secondary btn-sm" onclick="window.graphFocusNode('${data.target}')">Target</button>
    </div>
  `;

  showInspector();
}

// ─── Legend Builder ─────────────────────────────────────────────────────────
function buildLegend() {
  const container = document.getElementById('graph-legend');
  if (!container || !cyInstance) return;

  const counts = {};
  cyInstance.nodes().forEach(n => {
    const t = (n.data('type') || 'UNKNOWN').toUpperCase();
    const canonical = ENTITY_PALETTE[t] ? t : 'UNKNOWN';
    counts[canonical] = (counts[canonical] || 0) + 1;
  });

  container.innerHTML = Object.entries(counts).map(([type, count]) => {
    const p = getPalette(type);
    return `
      <div class="gl-item" data-legend-type="${type}" title="Toggle ${type} nodes">
        <span class="gl-dot" style="background:${p.accent};"></span>
        <span class="gl-label">${type}</span>
        <span class="gl-count">${count}</span>
      </div>
    `;
  }).join('');

  container.querySelectorAll('.gl-item').forEach(item => {
    item.onclick = () => {
      item.classList.toggle('hidden');
      const type = item.getAttribute('data-legend-type');
      const hiding = item.classList.contains('hidden');
      cyInstance.nodes().forEach(n => {
        const t = (n.data('type') || 'UNKNOWN').toUpperCase();
        if (t === type) {
          n.style('display', hiding ? 'none' : 'element');
          n.connectedEdges().style('display', hiding ? 'none' : 'element');
        }
      });
    };
  });
}

// ─── State Overlays ──────────────────────────────────────────────────────────
function showLoadingState(show) {
  const canvas = document.getElementById('cy-canvas');
  const loading = document.getElementById('graph-loading-overlay');
  const empty   = document.getElementById('graph-empty-overlay');

  if (loading) loading.style.display = show ? 'flex' : 'none';
  if (empty)   empty.style.display = 'none';
  if (canvas)  canvas.style.opacity = show ? '0.3' : '1';
}

function showEmptyState() {
  showLoadingState(false);
  const empty = document.getElementById('graph-empty-overlay');
  if (empty) empty.style.display = 'flex';
}

// ─── Public Helpers ─────────────────────────────────────────────────────────
window.graphFocusNode = function (nodeId) {
  if (!cyInstance) return;
  const node = cyInstance.getElementById(String(nodeId));
  if (node && node.length) {
    cyInstance.animate({ center: { eles: node }, zoom: 1.5 }, { duration: 350, easing: 'ease-in-out-cubic' });
    node.select();
    inspectNode(node);
  }
};

// Legacy compat alias kept — used by workbench scope changes
window.centerOnNodeId = window.graphFocusNode;
