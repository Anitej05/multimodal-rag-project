import React, { useEffect, useRef, useState, useCallback } from 'react';
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';

// Register the layout extension
cytoscape.use(coseBilkent);

/* ─────────────────────────────────────────────
   Color palette inspired by Obsidian graph view
   ───────────────────────────────────────────── */
const PALETTE = {
  document: { bg: '#6366f1', border: '#818cf8', glow: 'rgba(99,102,241,0.45)' },
  image:    { bg: '#f472b6', border: '#f9a8d4', glow: 'rgba(244,114,182,0.45)' },
  audio:    { bg: '#34d399', border: '#6ee7b7', glow: 'rgba(52,211,153,0.45)' },
  chunk:    { bg: '#94a3b8', border: '#cbd5e1', glow: 'rgba(148,163,184,0.25)' },
  concept:  { bg: '#fbbf24', border: '#fcd34d', glow: 'rgba(251,191,36,0.40)' },
  hub:      { bg: '#2563eb', border: '#60a5fa', glow: 'rgba(37,99,235,0.50)' },
};

const TYPE_LABELS = {
  document: '📄',
  image: '🖼️',
  audio: '🎵',
  chunk: '◆',
  concept: '💡',
  hub: '🌐',
};

/* ──────────────────────────────
   Main Knowledge Graph Component
   ────────────────────────────── */
const KnowledgeGraph = ({ uploadedFiles = [], isVisible = true }) => {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
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
  const fetchGraphData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://127.0.0.1:8000/knowledge-graph');
      if (!res.ok) throw new Error(`Failed to fetch graph: ${res.statusText}`);
      const data = await res.json();
      setGraphData(data);
    } catch (err) {
      console.error('Knowledge graph fetch error:', err);
      // Generate demo / sample data if backend is unreachable
      setGraphData(generateSampleData(uploadedFiles));
    } finally {
      setLoading(false);
    }
  }, [uploadedFiles]);

  /* ─────────────────────────────────
     Generate sample data from uploads
     ───────────────────────────────── */
  const generateSampleData = useCallback((files) => {
    const nodes = [];
    const edges = [];

    // Hub node
    nodes.push({
      id: 'hub-knowledge-base',
      label: 'Knowledge Base',
      type: 'hub',
      size: 55,
      description: 'Central knowledge repository',
    });

    // Concept nodes extracted from file names
    const conceptSet = new Set();

    const fileInfos = files.length > 0
      ? files.map(f => ({
          name: f.name || f.file?.name || 'unnamed',
          type: f.file?.type || '',
          size: f.file?.size || 0,
          status: f.status || 'pending',
        }))
      : [
          { name: 'research_paper.pdf', type: 'application/pdf', size: 2048000, status: 'indexed' },
          { name: 'architecture_diagram.png', type: 'image/png', size: 512000, status: 'indexed' },
          { name: 'meeting_notes.docx', type: 'application/docx', size: 128000, status: 'indexed' },
          { name: 'quarterly_data.csv', type: 'text/csv', size: 64000, status: 'indexed' },
          { name: 'interview_recording.mp3', type: 'audio/mpeg', size: 4096000, status: 'indexed' },
          { name: 'project_plan.pdf', type: 'application/pdf', size: 1024000, status: 'indexed' },
          { name: 'system_design.png', type: 'image/png', size: 768000, status: 'indexed' },
          { name: 'user_feedback.txt', type: 'text/plain', size: 32000, status: 'indexed' },
        ];

    fileInfos.forEach((file, idx) => {
      const ext = file.name.split('.').pop().toLowerCase();
      let nodeType = 'document';
      if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext)) nodeType = 'image';
      else if (['mp3', 'wav', 'm4a', 'ogg', 'flac'].includes(ext)) nodeType = 'audio';

      const nodeId = `file-${idx}`;
      const baseName = file.name.replace(/\.[^/.]+$/, '');

      nodes.push({
        id: nodeId,
        label: baseName,
        type: nodeType,
        size: Math.max(25, Math.min(45, 20 + Math.log2(file.size + 1) * 2)),
        description: `${file.name} (${(file.size / 1024).toFixed(1)} KB)`,
        status: file.status,
        fileType: ext.toUpperCase(),
      });

      // Connect to hub
      edges.push({
        source: 'hub-knowledge-base',
        target: nodeId,
        weight: 0.9,
        type: 'contains',
      });

      // Extract concepts from file name
      const words = baseName
        .replace(/[_\-\.]/g, ' ')
        .split(' ')
        .filter(w => w.length > 3)
        .map(w => w.toLowerCase());

      words.forEach(word => {
        const conceptId = `concept-${word}`;
        if (!conceptSet.has(word)) {
          conceptSet.add(word);
          nodes.push({
            id: conceptId,
            label: word.charAt(0).toUpperCase() + word.slice(1),
            type: 'concept',
            size: 18,
            description: `Concept: ${word}`,
          });
        }
        edges.push({
          source: nodeId,
          target: conceptId,
          weight: 0.5,
          type: 'mentions',
        });
      });

      // Generate synthetic chunk nodes
      const chunkCount = Math.max(1, Math.floor(Math.random() * 3) + 1);
      for (let c = 0; c < chunkCount; c++) {
        const chunkId = `chunk-${idx}-${c}`;
        nodes.push({
          id: chunkId,
          label: `Chunk ${c + 1}`,
          type: 'chunk',
          size: 12,
          description: `Text chunk ${c + 1} from ${file.name}`,
        });
        edges.push({
          source: nodeId,
          target: chunkId,
          weight: 0.7,
          type: 'chunk_of',
        });
      }
    });

    // Add inter-concept edges for concepts that share files
    const conceptFiles = {};
    edges.forEach(e => {
      if (e.type === 'mentions') {
        if (!conceptFiles[e.target]) conceptFiles[e.target] = new Set();
        conceptFiles[e.target].add(e.source);
      }
    });

    const conceptArr = Object.keys(conceptFiles);
    for (let i = 0; i < conceptArr.length; i++) {
      for (let j = i + 1; j < conceptArr.length; j++) {
        const shared = [...conceptFiles[conceptArr[i]]]
          .filter(f => conceptFiles[conceptArr[j]].has(f));
        if (shared.length > 0) {
          edges.push({
            source: conceptArr[i],
            target: conceptArr[j],
            weight: 0.3 * shared.length,
            type: 'related',
          });
        }
      }
    }

    // Count clusters (unique types)
    const types = new Set(nodes.map(n => n.type));

    return { nodes, edges, clusters: types.size };
  }, []);

  /* ────────────────────────
     Initialize Cytoscape
     ──────────────────────── */
  const initCytoscape = useCallback(() => {
    if (!containerRef.current || !graphData) return;

    // Destroy previous instance
    if (cyRef.current) {
      cyRef.current.destroy();
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
        /* ── Chunk node ── */
        {
          selector: 'node[type="chunk"]',
          style: {
            'shape': 'round-rectangle',
            'font-size': '7px',
            'opacity': 0.7,
            'text-opacity': 0.6,
            'label': '',
            'border-width': 1,
            'shadow-opacity': 0.3,
          },
        },
        /* ── Highlighted node ── */
        {
          selector: 'node.highlighted',
          style: {
            'border-width': 4,
            'border-color': '#ffffff',
            'shadow-blur': 35,
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
            'opacity': 0.15,
            'text-opacity': 0.1,
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
        /* ── Highlighted edge ── */
        {
          selector: 'edge.highlighted',
          style: {
            'line-color': 'rgba(255,255,255,0.6)',
            'opacity': 1,
            'width': function(el) {
              return Math.max(1.5, el.data('weight') * 3.5);
            },
            'z-index': 999,
          },
        },
        /* ── Faded edge ── */
        {
          selector: 'edge.faded',
          style: {
            'opacity': 0.05,
          },
        },
      ],

      layout: getLayoutConfig(layoutName),
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
        animate: 'during',
        animationDuration: 1200,
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
    const layout = cyRef.current.layout(getLayoutConfig(name));
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

  useEffect(() => {
    if (isVisible && graphData) {
      // Small delay to let the DOM update
      const timer = setTimeout(initCytoscape, 100);
      return () => clearTimeout(timer);
    }
  }, [isVisible, graphData, initCytoscape]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
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
