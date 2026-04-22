---
name: graph-investigation
description: >
  Build interactive connected graph / network visualizations for investigation dashboards,
  entity relationship maps, fraud detection UIs, money laundering detection, and link analysis tools.
  Use this skill whenever the user wants to: visualize relationships between entities (people, accounts,
  companies, transactions), build node-link diagrams, create force-directed graphs, render investigation
  boards, connect nodes with edges, show entity networks, or display graph data with hover tooltips.
  Also triggers for: "connect entities", "relationship map", "network graph", "link analysis",
  "node graph", "entity graph", "investigation board", "transaction network", "follow the money".
  ALWAYS use this skill before writing any graph/network visualization code — it contains critical
  library selection guidance, layout tuning, and tooltip/interaction patterns that prevent common pitfalls.
---

# Graph Investigation Skill

Builds production-grade interactive node-link graphs for investigation and entity relationship UIs —
think money laundering dashboards, fraud detection, corporate ownership trees, and link analysis boards.

## Library Selection

Pick based on graph size and required interactivity:

| Library | Best For | Graph Size | Notes |
|---|---|---|---|
| **Cytoscape.js** | Investigation dashboards, rich styling, layouts | Up to ~5k nodes | ✅ Preferred for investigation UIs. Best layout variety. |
| **D3-force** | Custom, highly tailored force graphs | Up to ~2k nodes | Full control, more code. Use `d3-force` directly. |
| **Sigma.js + Graphology** | Very large graphs, WebGL rendering | 10k+ nodes | Use when performance is the constraint |
| **React Flow** | Node-based editors / flowchart-style | Small-medium | Best when nodes are complex UI components, not data points |
| **vis.js Network** | Quick prototypes | Small-medium | Feature-rich but heavier; less customizable styling |

**Recommended default**: **Cytoscape.js** for investigation dashboards. It has the best built-in layouts, clean API, and rich event model for hover/click interactions.

---

## Architecture Pattern

```
GraphContainer
├── graph engine (Cytoscape / D3 / Sigma)
├── TooltipOverlay       ← absolutely positioned, portal or sibling div
├── SidePanel            ← entity detail panel (expands on click)
├── ControlBar           ← layout switcher, zoom, filter, search
└── LegendPanel          ← node types, edge types, risk levels
```

---

## Node & Edge Design for Investigation UIs

### Node Types (money laundering / fraud context)
```js
const nodeTypes = {
  person:      { shape: 'ellipse',   color: '#3B82F6', icon: '👤' },
  company:     { shape: 'rectangle', color: '#8B5CF6', icon: '🏢' },
  account:     { shape: 'diamond',   color: '#10B981', icon: '💳' },
  transaction: { shape: 'triangle',  color: '#F59E0B', icon: '💸' },
  flagged:     { shape: 'star',      color: '#EF4444', icon: '🚨' },
}
```

### Edge Types
```js
const edgeTypes = {
  transfer:    { lineStyle: 'solid',  color: '#6B7280', label: 'Transfer' },
  owns:        { lineStyle: 'solid',  color: '#8B5CF6', label: 'Owns'     },
  controls:    { lineStyle: 'dashed', color: '#F59E0B', label: 'Controls' },
  suspicious:  { lineStyle: 'solid',  color: '#EF4444', width: 3          },
}
```

### Risk Scoring Visual Encoding
- **Color**: Green (clean) → Yellow (watch) → Orange (elevated) → Red (high risk)
- **Size**: Scale node radius by transaction volume or risk score: `radius = 8 + (riskScore / 100) * 24`
- **Opacity**: Dim unrelated nodes during hover (`opacity: 0.15`)
- **Border**: Pulsing red border for flagged nodes via CSS animation

---

## Cytoscape.js Implementation

### Setup (CDN or npm)
```bash
npm install cytoscape
# Layout add-ons:
npm install cytoscape-cola      # Best for investigation graphs — constraint-based
npm install cytoscape-dagre     # Hierarchical / org-chart style
npm install cytoscape-fcose     # Fast, compound-aware force layout
```

```js
import cytoscape from 'cytoscape';
import cola from 'cytoscape-cola';
cytoscape.use(cola);
```

### Core Initialization
```js
const cy = cytoscape({
  container: document.getElementById('graph'),
  elements: {
    nodes: graphData.nodes.map(n => ({
      data: { id: n.id, label: n.name, type: n.type, risk: n.riskScore, ...n }
    })),
    edges: graphData.edges.map(e => ({
      data: { id: e.id, source: e.source, target: e.target, type: e.type, amount: e.amount }
    }))
  },
  style: cytoscapeStyles,   // see Styling section below
  layout: { name: 'cola', nodeSpacing: 80, edgeLength: 160, animate: true }
});
```

### Recommended Layout Config (Cola — best for investigation)
```js
const colaLayout = {
  name: 'cola',
  nodeSpacing: 80,          // ← key for even spacing
  edgeLengthVal: 160,
  animate: true,
  randomize: false,
  maxSimulationTime: 3000,
  ungrabifyWhileSimulating: false,
  fit: true,
  padding: 40,
  nodeDimensionsIncludeLabels: true,
  handleDisconnected: true,  // ← important: keeps isolated nodes in frame
  convergenceThreshold: 0.01,
};
```

### Stylesheet
```js
const cytoscapeStyles = [
  {
    selector: 'node',
    style: {
      'background-color': (ele) => riskColor(ele.data('risk')),
      'width': (ele) => nodeSize(ele.data('risk')),
      'height': (ele) => nodeSize(ele.data('risk')),
      'label': 'data(label)',
      'color': '#fff',
      'font-size': '11px',
      'text-valign': 'bottom',
      'text-margin-y': 6,
      'text-outline-width': 2,
      'text-outline-color': '#111',
      'border-width': 2,
      'border-color': '#ffffff22',
    }
  },
  {
    selector: 'node:selected',
    style: {
      'border-width': 3,
      'border-color': '#fff',
      'overlay-color': '#fff',
      'overlay-opacity': 0.1,
    }
  },
  {
    selector: 'edge',
    style: {
      'width': 1.5,
      'line-color': '#334155',
      'target-arrow-color': '#334155',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'label': 'data(label)',
      'font-size': '9px',
      'color': '#94a3b8',
      'text-rotation': 'autorotate',
    }
  },
  {
    selector: 'edge[type="suspicious"]',
    style: {
      'line-color': '#EF4444',
      'target-arrow-color': '#EF4444',
      'width': 2.5,
      'line-style': 'solid',
    }
  },
  {
    selector: '.dimmed',   // applied to unrelated nodes on hover
    style: { 'opacity': 0.15 }
  }
];
```

---

## Hover Tooltip Pattern

**Critical**: Never use Cytoscape's built-in qtip for complex data. Use a floating HTML div instead — gives full control over styling.

### HTML Structure
```html
<div id="graph-container" style="position:relative">
  <div id="cy" style="width:100%;height:100%"></div>
  <div id="tooltip" class="node-tooltip" style="display:none; position:absolute; pointer-events:none; z-index:10;">
    <!-- content injected by JS -->
  </div>
</div>
```

### Tooltip Logic
```js
const tooltip = document.getElementById('tooltip');
const container = document.getElementById('graph-container');

cy.on('mouseover', 'node', (evt) => {
  const node = evt.target;
  const data = node.data();
  const pos = evt.renderedPosition;       // pixel position within container
  const containerRect = container.getBoundingClientRect();

  // Highlight neighborhood, dim others
  cy.elements().addClass('dimmed');
  node.neighborhood().add(node).removeClass('dimmed');

  // Render tooltip content
  tooltip.innerHTML = buildTooltipHTML(data);
  tooltip.style.display = 'block';

  // Smart positioning: flip if near edge
  const tx = pos.x + 16;
  const ty = pos.y - 10;
  const flipX = tx + 260 > container.offsetWidth;
  tooltip.style.left  = flipX ? `${pos.x - 270}px` : `${tx}px`;
  tooltip.style.top   = `${Math.max(0, ty)}px`;
});

cy.on('mouseout', 'node', () => {
  tooltip.style.display = 'none';
  cy.elements().removeClass('dimmed');
});

cy.on('tap', 'node', (evt) => {   // click → open side panel
  openSidePanel(evt.target.data());
});

function buildTooltipHTML(data) {
  return `
    <div class="tt-header">
      <span class="tt-type">${data.type?.toUpperCase()}</span>
      <span class="tt-risk risk-${riskLevel(data.risk)}">${data.risk ?? '—'} risk</span>
    </div>
    <div class="tt-name">${data.label}</div>
    <div class="tt-fields">
      ${data.jurisdiction ? `<div class="tt-row"><span>Jurisdiction</span><span>${data.jurisdiction}</span></div>` : ''}
      ${data.totalVolume  ? `<div class="tt-row"><span>Volume</span><span>$${fmt(data.totalVolume)}</span></div>` : ''}
      ${data.flagged      ? `<div class="tt-flag">⚠ Flagged for review</div>` : ''}
    </div>
  `;
}
```

### Tooltip CSS
```css
.node-tooltip {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 12px 14px;
  min-width: 220px;
  max-width: 260px;
  font-size: 12px;
  color: #e2e8f0;
  box-shadow: 0 8px 24px rgba(0,0,0,0.6);
  pointer-events: none;
}
.tt-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.tt-type   { font-size: 10px; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase; }
.tt-name   { font-size: 14px; font-weight: 600; color: #f8fafc; margin-bottom: 8px; }
.tt-row    { display: flex; justify-content: space-between; padding: 2px 0; color: #94a3b8; }
.tt-flag   { margin-top: 8px; padding: 4px 8px; background: #450a0a; border-radius: 4px; color: #fca5a5; font-size: 11px; }
.risk-high   { color: #ef4444; }
.risk-medium { color: #f59e0b; }
.risk-low    { color: #10b981; }
```

---

## Even Spacing — Common Issues & Fixes

| Problem | Fix |
|---|---|
| Nodes pile up | Increase `nodeSpacing` (cola) or `charge` (d3-force) |
| Long chains go offscreen | Enable `fit: true`, use `padding: 60` |
| Hub nodes overlap spokes | Use `fcose` layout with `idealEdgeLength: 100` |
| Disconnected nodes float away | Set `handleDisconnected: true` (cola) |
| Labels overlap | `nodeDimensionsIncludeLabels: true` |
| Graph too compressed | Set `edgeLengthVal: 180` or higher |

### D3-Force equivalent for even spacing
```js
const simulation = d3.forceSimulation(nodes)
  .force('link',    d3.forceLink(links).id(d => d.id).distance(140).strength(0.7))
  .force('charge',  d3.forceManyBody().strength(-400))       // repulsion
  .force('center',  d3.forceCenter(width / 2, height / 2))
  .force('collide', d3.forceCollide().radius(d => d.r + 20)) // prevents overlap
  .force('x',       d3.forceX(width / 2).strength(0.05))    // soft centering
  .force('y',       d3.forceY(height / 2).strength(0.05));
```

---

## Investigation-Specific Patterns

### Path Highlighting (follow the money)
```js
// Highlight shortest path between two selected nodes
const path = cy.elements().aStar({
  root: cy.$('#nodeA'),
  goal: cy.$('#nodeB'),
  weight: (edge) => edge.data('amount') ? 1 / edge.data('amount') : 1,
});
path.path.addClass('highlighted-path');
```

### Cluster by Risk Level
```js
cy.layout({
  name: 'fcose',
  idealEdgeLength: () => 100,
  nodeRepulsion: () => 6000,
  nodeSeparation: 100,
}).run();
```

### Filter / Dim by Entity Type
```js
function filterByType(type) {
  cy.nodes().forEach(n => {
    if (n.data('type') !== type) n.addClass('dimmed');
    else n.removeClass('dimmed');
  });
}
```

### Transaction Amount on Edges
```js
{
  selector: 'edge[amount]',
  style: {
    'width': ele => Math.max(1, Math.log(ele.data('amount') / 1000) * 2),
    'label': ele => `$${(ele.data('amount')/1000).toFixed(0)}k`,
  }
}
```

---

## Reference Files

- `references/cytoscape-api.md` — Full Cytoscape.js API patterns, layout options, event reference
- `references/d3-force-patterns.md` — D3 force graph patterns, SVG patterns, zoom/pan

Read these when:
- Building from scratch with D3 (read `d3-force-patterns.md`)
- Need advanced Cytoscape layout tuning, compound nodes, or animation (read `cytoscape-api.md`)

---

## Quality Checklist

Before shipping a graph component:
- [ ] Nodes are evenly spaced (no pileups, no isolated floaters)
- [ ] Hover tooltip appears within 100ms, positioned correctly, never clips edge
- [ ] Tooltip content is entity-specific (not generic)
- [ ] Dimming effect isolates hovered node neighborhood
- [ ] Click opens side panel or detail view
- [ ] Graph fits container on load (no scroll required)
- [ ] Edge labels don't overlap node labels
- [ ] Risk levels are visually distinct (color + size)
- [ ] Layout reruns gracefully on data update
- [ ] Zoom and pan work (mouse wheel + drag)