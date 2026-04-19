import React, { useEffect, useRef, useState, useCallback } from 'react';
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import api from '../services/api';

// Register the layout extension
cytoscape.use(coseBilkent);

/* ─────────────────────────────────────────────
   Color palette inspired by Obsidian graph view
   ───────────────────────────────────────────── */
const PALETTE = {
  document: { bg: '#6366f1', border: '#818cf8', glow: 'rgba(99,102,241,0.45)' },
  image:    { bg: '#f472b6', border: '#f9a8d4', glow: 'rgba(244,114,182,0.45)' },
  audio:    { bg: '#34d399', border: '#6ee7b7', glow: 'rgba(52,211,153,0.45)' },
  concept:  { bg: '#fbbf24', border: '#fcd34d', glow: 'rgba(251,191,36,0.40)' },
  entity:   { bg: '#f97316', border: '#fb923c', glow: 'rgba(249,115,22,0.45)' },
  hub:      { bg: '#2563eb', border: '#60a5fa', glow: 'rgba(37,99,235,0.50)' },
};

const TYPE_LABELS = {
  document: '📄',
  image: '🖼️',
  audio: '🎵',
  concept: '💡',
  entity: '🔗',
  hub: '🌐',
};

/* ──────────────────────────────
   Main Knowledge Graph Component
   ────────────────────────────── */
const KnowledgeGraph = ({ uploadedFiles = [], isVisible = true, isIndexing = false }) => {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const layoutRef = useRef(null);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [stats, setStats] = useState({ nodes: 0, edges: 0, clusters: 0 });
  const [searchQuery, setSearchQuery] = useState('');
  const [layoutName, setLayoutName] = useState('cose-bilkent');
  const [hoveredNode, setHoveredNode] = useState(null);

  /* ─────────────────────────
     Fetch graph data from API
     ───────────────────────── */
  const fetchGraphData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const data = await api.getKnowledgeGraph();
      // Only update if data has nodes (avoid empty flash)
      if (data && data.nodes && data.nodes.length > 0) {
        setGraphData(data);
      } else if (!silent) {
        setGraphData(null);
      }
    } catch (err) {
      console.error('Knowledge graph fetch error:', err);
      if (!silent) setError('Could not load knowledge graph from backend.');
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  /* No more fake data — graph comes from backend LLM extraction */

  /* ────────────────────────
     Initialize Cytoscape
     ──────────────────────── */
  const initCytoscape = useCallback(() => {
    if (!containerRef.current || !graphData) return;

    // Stop any running layout and destroy previous instance
    if (layoutRef.current) {
      try { layoutRef.current.stop(); } catch (e) { /* already stopped */ }
      layoutRef.current = null;
    }
    if (cyRef.current) {
      try { cyRef.current.destroy(); } catch (e) { /* already destroyed */ }
      cyRef.current = null;
    }

    const elements = [];

    // Add nodes
    graphData.nodes.forEach(node => {
      const palette = PALETTE[node.type] || PALETTE.chunk;
      elements.push({
        group: 'nodes',
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          size: node.size || 30,
          description: node.description || '',
          bgColor: palette.bg,
          borderColor: palette.border,
          glowColor: palette.glow,
          status: node.status || '',
          fileType: node.fileType || '',
        },
      });
    });

    // Add edges
    graphData.edges.forEach((edge, idx) => {
      elements.push({
        group: 'edges',
        data: {
          id: `edge-${idx}`,
          source: edge.source,
          target: edge.target,
          weight: edge.weight || 0.5,
          edgeType: edge.type || 'default',
        },
      });
    });

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      minZoom: 0.15,
      maxZoom: 4,
      wheelSensitivity: 0.3,
      pixelRatio: 'auto',

      style: [
        /* ── Default node style ── */
        {
          selector: 'node',
          style: {
            'background-color': 'data(bgColor)',
            'border-color': 'data(borderColor)',
            'border-width': 2,
            'label': 'data(label)',
            'width': 'data(size)',
            'height': 'data(size)',
            'font-size': '10px',
            'font-family': "'Inter', system-ui, sans-serif",
            'font-weight': '600',
            'color': '#e2e8f0',
            'text-outline-color': 'rgba(7,16,40,0.85)',
            'text-outline-width': 2,
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 8,
            'overlay-opacity': 0,
            'transition-property': 'background-color, border-color, width, height, opacity, border-width',
            'transition-duration': '0.3s',
            'transition-timing-function': 'ease-out',
            'shadow-blur': 15,
            'shadow-color': 'data(glowColor)',
            'shadow-offset-x': 0,
            'shadow-offset-y': 0,
            'shadow-opacity': 0.7,
          },
        },
        /* ── Hub node ── */
        {
          selector: 'node[type="hub"]',
          style: {
            'font-size': '13px',
            'font-weight': '700',
            'text-outline-width': 3,
            'border-width': 3,
            'shadow-blur': 30,
            'shadow-opacity': 0.9,
          },
        },
        /* ── Concept node ── */
        {
          selector: 'node[type="concept"]',
          style: {
            'shape': 'diamond',
            'font-size': '9px',
            'text-outline-width': 1.5,
          },
        },
        /* ── Entity node ── */
        {
          selector: 'node[type="entity"]',
          style: {
            'shape': 'round-hexagon',
            'font-size': '9px',
            'text-outline-width': 1.5,
            'border-width': 2,
          },
        },
        /* ── Highlighted node ── */
        {
          selector: 'node.highlighted',
          style: {
            'border-width': 4,
            'border-color': '#ffffff',
            'shadow-blur': 40,
            'shadow-color': '#ffffff',
            'shadow-opacity': 1,
            'z-index': 999,
          },
        },
        /* ── Selected node ── */
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#ffffff',
            'shadow-blur': 40,
            'shadow-opacity': 1,
          },
        },
        /* ── Faded node ── */
        {
          selector: 'node.faded',
          style: {
            'opacity': 0.05,
            'text-opacity': 0,
            'shadow-opacity': 0,
          },
        },
        /* ── Search matched ── */
        {
          selector: 'node.search-match',
          style: {
            'border-color': '#fbbf24',
            'border-width': 3,
            'shadow-color': 'rgba(251,191,36,0.6)',
            'shadow-blur': 25,
            'shadow-opacity': 1,
          },
        },
        /* ── Default edge style ── */
        {
          selector: 'edge',
          style: {
            'width': function(el) {
              return Math.max(0.5, el.data('weight') * 2.5);
            },
            'line-color': 'rgba(148,163,184,0.15)',
            'curve-style': 'bezier',
            'opacity': 0.5,
            'transition-property': 'opacity, line-color, width',
            'transition-duration': '0.3s',
            'target-arrow-shape': 'none',
          },
        },
        /* ── "Contains" edges ── */
        {
          selector: 'edge[edgeType="contains"]',
          style: {
            'line-color': 'rgba(99,102,241,0.25)',
            'width': 2,
            'line-style': 'solid',
          },
        },
        /* ── "Mentions" edges ── */
        {
          selector: 'edge[edgeType="mentions"]',
          style: {
            'line-color': 'rgba(251,191,36,0.2)',
            'width': 1,
            'line-style': 'dashed',
          },
        },
        /* ── "Related" edges ── */
        {
          selector: 'edge[edgeType="related"]',
          style: {
            'line-color': 'rgba(52,211,153,0.2)',
            'width': 1.5,
          },
        },
        /* ── "has_entity" edges ── */
        {
          selector: 'edge[edgeType="has_entity"]',
          style: {
            'line-color': 'rgba(249,115,22,0.25)',
            'width': 1.5,
            'line-style': 'solid',
          },
        },
        /* ── "related_to" edges ── */
        {
          selector: 'edge[edgeType="related_to"]',
          style: {
            'line-color': 'rgba(168,85,247,0.3)',
            'width': 2,
            'line-style': 'solid',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'rgba(168,85,247,0.3)',
            'arrow-scale': 0.8,
          },
        },
        /* ── Highlighted edge ── */
        {
          selector: 'edge.highlighted',
          style: {
            'line-color': '#3b82f6', /* Bright primary blue */
            'opacity': 1,
            'width': function(el) {
              return Math.max(3, el.data('weight') * 5);
            },
            'z-index': 999,
          },
        },
        /* ── Faded edge ── */
        {
          selector: 'edge.faded',
          style: {
            'opacity': 0.02,
          },
        },
      ],

      layout: { name: 'preset' },
    });

    /* ── Interactive: hover ── */
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      setHoveredNode({
        id: node.id(),
        label: node.data('label'),
        type: node.data('type'),
        description: node.data('description'),
      });

      // Highlight neighborhood
      const neighborhood = node.neighborhood().add(node);
      cy.elements().not(neighborhood).addClass('faded');
      neighborhood.addClass('highlighted');
      neighborhood.connectedEdges().addClass('highlighted');
    });

    cy.on('mouseout', 'node', () => {
      setHoveredNode(null);
      cy.elements().removeClass('faded highlighted');
    });

    /* ── Interactive: click ── */
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNode({
        id: node.id(),
        label: node.data('label'),
        type: node.data('type'),
        description: node.data('description'),
        status: node.data('status'),
        fileType: node.data('fileType'),
        connections: node.neighborhood('node').length,
        degree: node.degree(),
      });
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null);
      }
    });

    // Store reference
    cyRef.current = cy;

    // Run layout after cytoscape is fully initialized
    const layout = cy.layout(getLayoutConfig(layoutName));
    layoutRef.current = layout;
    layout.run();

    // Update stats
    setStats({
      nodes: graphData.nodes.length,
      edges: graphData.edges.length,
      clusters: graphData.clusters || new Set(graphData.nodes.map(n => n.type)).size,
    });

  }, [graphData, layoutName]);

  /* ───────────────────────
     Layout configurations
     ─────────────────────── */
  function getLayoutConfig(name) {
    const configs = {
      'cose-bilkent': {
        name: 'cose-bilkent',
        quality: 'default',
        nodeDimensionsIncludeLabels: true,
        animate: 'end',
        animationDuration: 800,
        animationEasing: 'ease-out',
        fit: true,
        padding: 60,
        randomize: true,
        nodeRepulsion: 8500,
        idealEdgeLength: 120,
        edgeElasticity: 0.45,
        nestingFactor: 0.1,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
        tilingPaddingVertical: 20,
        tilingPaddingHorizontal: 20,
        gravityRangeCompound: 1.5,
        gravityCompound: 1.0,
        gravityRange: 3.8,
      },
      'concentric': {
        name: 'concentric',
        animate: true,
        animationDuration: 800,
        fit: true,
        padding: 50,
        minNodeSpacing: 50,
        concentric: (node) => {
          if (node.data('type') === 'hub') return 10;
          if (node.data('type') === 'document' || node.data('type') === 'image' || node.data('type') === 'audio') return 7;
          if (node.data('type') === 'concept') return 4;
          return 1;
        },
        levelWidth: () => 2,
      },
      'circle': {
        name: 'circle',
        animate: true,
        animationDuration: 800,
        fit: true,
        padding: 50,
        avoidOverlap: true,
        spacingFactor: 1.5,
      },
      'grid': {
        name: 'grid',
        animate: true,
        animationDuration: 600,
        fit: true,
        padding: 40,
        avoidOverlap: true,
        condense: true,
        rows: undefined,
      },
    };
    return configs[name] || configs['cose-bilkent'];
  }

  /* ────────────────
     Search handler
     ──────────────── */
  const handleSearch = useCallback((query) => {
    setSearchQuery(query);
    if (!cyRef.current) return;
    const cy = cyRef.current;

    cy.elements().removeClass('search-match faded');

    if (!query.trim()) return;

    const lower = query.toLowerCase();
    const matched = cy.nodes().filter(n =>
      n.data('label').toLowerCase().includes(lower) ||
      (n.data('description') || '').toLowerCase().includes(lower)
    );

    if (matched.length > 0) {
      cy.elements().not(matched).not(matched.neighborhood()).addClass('faded');
      matched.addClass('search-match');
      cy.animate({ fit: { eles: matched, padding: 80 }, duration: 600 });
    }
  }, []);

  /* ────────────────
     Re-layout
     ──────────────── */
  const relayout = useCallback((name) => {
    setLayoutName(name);
    if (!cyRef.current) return;
    // Stop previous layout before starting new one
    if (layoutRef.current) {
      try { layoutRef.current.stop(); } catch (e) { /* already stopped */ }
    }
    const layout = cyRef.current.layout(getLayoutConfig(name));
    layoutRef.current = layout;
    layout.run();
  }, []);

  /* ────────────────
     Fit to screen
     ──────────────── */
  const fitToScreen = useCallback(() => {
    if (!cyRef.current) return;
    cyRef.current.animate({ fit: { padding: 50 }, duration: 500 });
  }, []);

  /* ────────────────
     Zoom controls
     ──────────────── */
  const zoomIn = useCallback(() => {
    if (!cyRef.current) return;
    cyRef.current.animate({
      zoom: cyRef.current.zoom() * 1.3,
      center: cyRef.current.extent(),
      duration: 300,
    });
  }, []);

  const zoomOut = useCallback(() => {
    if (!cyRef.current) return;
    cyRef.current.animate({
      zoom: cyRef.current.zoom() / 1.3,
      center: cyRef.current.extent(),
      duration: 300,
    });
  }, []);

  /* ──────────────────────
     Effects
     ────────────────────── */
  useEffect(() => {
    if (isVisible) fetchGraphData();
  }, [isVisible, fetchGraphData]);

  // Auto-refresh during indexing
  useEffect(() => {
    if (!isVisible || !isIndexing) return;
    const interval = setInterval(() => {
      fetchGraphData(true); // silent refresh
    }, 4000);
    return () => clearInterval(interval);
  }, [isVisible, isIndexing, fetchGraphData]);

  useEffect(() => {
    if (isVisible && graphData) {
      const timer = setTimeout(initCytoscape, 100);
      return () => clearTimeout(timer);
    }
  }, [isVisible, graphData, initCytoscape]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (layoutRef.current) {
        try { layoutRef.current.stop(); } catch (e) { /* already stopped */ }
        layoutRef.current = null;
      }
      if (cyRef.current) {
        try { cyRef.current.destroy(); } catch (e) { /* already destroyed */ }
        cyRef.current = null;
      }
    };
  }, []);

  if (!isVisible) return null;

  /* ──────────────────────
     RENDER
     ────────────────────── */
  return (
    <div className="kg-container" id="knowledge-graph-container">
      {/* ── Toolbar ── */}
      <div className="kg-toolbar">
        <div className="kg-toolbar-left">
          <div className="kg-toolbar-title">
            <span className="kg-toolbar-icon">🕸️</span>
            <span>Knowledge Graph</span>
          </div>
          <div className="kg-search-box">
            <svg className="kg-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35"/>
            </svg>
            <input
              type="text"
              className="kg-search-input"
              placeholder="Search nodes..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              id="kg-search-input"
            />
          </div>
        </div>

        <div className="kg-toolbar-right">
          {/* Layout selector */}
          <div className="kg-layout-select-wrapper">
            <select
              className="kg-layout-select"
              value={layoutName}
              onChange={(e) => relayout(e.target.value)}
              id="kg-layout-select"
            >
              <option value="cose-bilkent">Force Directed</option>
              <option value="concentric">Concentric</option>
              <option value="circle">Circle</option>
              <option value="grid">Grid</option>
            </select>
          </div>

          {/* Zoom buttons */}
          <div className="kg-zoom-controls">
            <button className="kg-btn kg-btn-icon" onClick={zoomIn} title="Zoom In" id="kg-zoom-in">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
                <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
              </svg>
            </button>
            <button className="kg-btn kg-btn-icon" onClick={zoomOut} title="Zoom Out" id="kg-zoom-out">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
                <line x1="8" y1="11" x2="14" y2="11"/>
              </svg>
            </button>
            <button className="kg-btn kg-btn-icon" onClick={fitToScreen} title="Fit" id="kg-fit-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/>
              </svg>
            </button>
          </div>

          <button
            className="kg-btn kg-btn-refresh"
            onClick={fetchGraphData}
            disabled={loading}
            id="kg-refresh-btn"
          >
            {loading ? (
              <span className="kg-spinner" />
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="15" height="15">
                <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
                <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* ── Cytoscape Canvas ── */}
      <div className="kg-canvas-wrapper">
        {loading && (
          <div className="kg-loading-overlay">
            <div className="kg-loading-spinner" />
            <p>Mapping knowledge connections...</p>
          </div>
        )}

        {!loading && !graphData && !error && (
          <div className="kg-loading-overlay">
            <span style={{ fontSize: '48px', marginBottom: '16px' }}>🕸️</span>
            <p style={{ fontSize: '15px', fontWeight: 600 }}>No knowledge graph yet</p>
            <p style={{ fontSize: '13px', opacity: 0.6 }}>Upload and index files to generate a meaningful knowledge graph with LLM-extracted entities and relationships.</p>
          </div>
        )}

        {isIndexing && graphData && (
          <div className="kg-indexing-indicator">
            <span className="loading-spinner"></span>
            <span>Graph updating — extracting entities...</span>
          </div>
        )}

        {error && (
          <div className="kg-error-overlay">
            <span className="kg-error-icon">⚠️</span>
            <p>{error}</p>
            <button className="kg-btn" onClick={fetchGraphData}>Retry</button>
          </div>
        )}

        <div ref={containerRef} className="kg-cytoscape" id="kg-cytoscape-canvas" />

        {/* ── Particle background ── */}
        <div className="kg-particles">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="kg-particle"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${3 + Math.random() * 4}s`,
              }}
            />
          ))}
        </div>
      </div>

      {/* ── Legend ── */}
      <div className="kg-legend">
        {Object.entries(PALETTE).map(([type, colors]) => (
          <div className="kg-legend-item" key={type}>
            <span
              className="kg-legend-dot"
              style={{
                background: colors.bg,
                boxShadow: `0 0 8px ${colors.glow}`,
              }}
            />
            <span className="kg-legend-label">
              {TYPE_LABELS[type]} {type.charAt(0).toUpperCase() + type.slice(1)}
            </span>
          </div>
        ))}
      </div>

      {/* ── Stats bar ── */}
      <div className="kg-stats-bar">
        <div className="kg-stat">
          <span className="kg-stat-value">{stats.nodes}</span>
          <span className="kg-stat-label">Nodes</span>
        </div>
        <div className="kg-stat-divider" />
        <div className="kg-stat">
          <span className="kg-stat-value">{stats.edges}</span>
          <span className="kg-stat-label">Edges</span>
        </div>
        <div className="kg-stat-divider" />
        <div className="kg-stat">
          <span className="kg-stat-value">{stats.clusters}</span>
          <span className="kg-stat-label">Clusters</span>
        </div>
      </div>

      {/* ── Hover tooltip ── */}
      {hoveredNode && (
        <div className="kg-tooltip">
          <div className="kg-tooltip-header">
            <span className="kg-tooltip-type-badge" style={{
              background: PALETTE[hoveredNode.type]?.bg || '#94a3b8',
            }}>
              {TYPE_LABELS[hoveredNode.type] || '◆'} {hoveredNode.type}
            </span>
          </div>
          <div className="kg-tooltip-label">{hoveredNode.label}</div>
          {hoveredNode.description && (
            <div className="kg-tooltip-desc">{hoveredNode.description}</div>
          )}
        </div>
      )}

      {/* ── Detail panel for selected node ── */}
      {selectedNode && (
        <div className="kg-detail-panel" id="kg-detail-panel">
          <div className="kg-detail-header">
            <div className="kg-detail-type" style={{
              background: PALETTE[selectedNode.type]?.bg || '#94a3b8',
            }}>
              {TYPE_LABELS[selectedNode.type] || '◆'}
            </div>
            <div className="kg-detail-title">
              <h3>{selectedNode.label}</h3>
              <span className="kg-detail-type-label">{selectedNode.type}</span>
            </div>
            <button
              className="kg-detail-close"
              onClick={() => setSelectedNode(null)}
              id="kg-detail-close"
            >
              ×
            </button>
          </div>
          <div className="kg-detail-body">
            {selectedNode.description && (
              <div className="kg-detail-row">
                <span className="kg-detail-key">Description</span>
                <span className="kg-detail-val">{selectedNode.description}</span>
              </div>
            )}
            {selectedNode.fileType && (
              <div className="kg-detail-row">
                <span className="kg-detail-key">File Type</span>
                <span className="kg-detail-val kg-detail-badge">{selectedNode.fileType}</span>
              </div>
            )}
            {selectedNode.status && (
              <div className="kg-detail-row">
                <span className="kg-detail-key">Status</span>
                <span className={`kg-detail-val kg-detail-status kg-detail-status-${selectedNode.status}`}>
                  {selectedNode.status === 'indexed' ? '✅ Indexed' : '⏳ Pending'}
                </span>
              </div>
            )}
            <div className="kg-detail-row">
              <span className="kg-detail-key">Connections</span>
              <span className="kg-detail-val">{selectedNode.connections} nodes</span>
            </div>
            <div className="kg-detail-row">
              <span className="kg-detail-key">Degree</span>
              <span className="kg-detail-val">{selectedNode.degree}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeGraph;
