'use client';

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import styles from './globe.module.css';
import { ML_HUBS, ML_CORRIDORS, getHubById, riskColor, formatVolume, riskScoreToLevel } from '@/lib/geoData';
import type { MLHub, MLCorridor } from '@/lib/types';

// Dynamic import to avoid SSR issues with Leaflet
let L: typeof import('leaflet') | null = null;

interface Props {
  onCorridorSelect: (corridorId: string) => void;
  onHubSelect: (hubId: string) => void;
}

type ViewMode = '3d' | 'flat';

export default function GlobalThreatMap({ onCorridorSelect, onHubSelect }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<L.Map | null>(null);
  const globeContainerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<any>(null);
  const svgOverlay = useRef<SVGSVGElement | null>(null);
  const [hoveredHub, setHoveredHub] = useState<MLHub | null>(null);
  const [hoveredCorridor, setHoveredCorridor] = useState<MLCorridor | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [mapReady, setMapReady] = useState(false);
  const [countries, setCountries] = useState<any>({ features: [] });
  const [viewMode, setViewMode] = useState<ViewMode>('3d');

  useEffect(() => {
    fetch('https://unpkg.com/globe.gl/example/datasets/ne_110m_admin_0_countries.geojson')
      .then(res => res.json())
      .then(data => setCountries(data));
  }, []);
  const [GlobeComponent, setGlobeComponent] = useState<any>(null);
  const rafId = useRef<number | null>(null);

  // Load Globe component dynamically
  useEffect(() => {
    import('react-globe.gl').then(mod => {
      setGlobeComponent(() => mod.default);
    });
  }, []);

  // Globe data
  const globePointsData = useMemo(() =>
    ML_HUBS.map(hub => ({
      lat: hub.lat,
      lng: hub.lng,
      size: hub.riskLevel === 'critical' ? 0.6 : hub.riskLevel === 'high' ? 0.4 : 0.25,
      color: riskColor(hub.riskLevel),
      name: hub.name,
      id: hub.id,
      riskLevel: hub.riskLevel,
      entityCount: hub.entityCount,
      region: hub.region,
    })),
    []);

  const globeArcsData = useMemo(() =>
    ML_CORRIDORS.map(corridor => {
      const src = getHubById(corridor.source);
      const tgt = getHubById(corridor.target);
      if (!src || !tgt) return null;
      const level = riskScoreToLevel(corridor.riskScore);
      return {
        startLat: src.lat,
        startLng: src.lng,
        endLat: tgt.lat,
        endLng: tgt.lng,
        color: riskColor(level),
        stroke: Math.max(0.5, corridor.volume / 2e9),
        id: corridor.id,
        label: corridor.label,
        riskScore: corridor.riskScore,
        volume: corridor.volume,
        typology: corridor.typology,
      };
    }).filter(Boolean),
    []);

  const globeLabelsData = useMemo(() =>
    ML_HUBS.map(hub => ({
      lat: hub.lat,
      lng: hub.lng,
      text: hub.name,
      size: 0.6,
      color: '#808088',
    })),
    []);

  // Handle 3D globe interactions
  const handleGlobePointClick = useCallback((point: any) => {
    if (point?.id) onHubSelect(point.id);
  }, [onHubSelect]);

  const handleGlobeArcClick = useCallback((arc: any) => {
    if (arc?.id) onCorridorSelect(arc.id);
  }, [onCorridorSelect]);

  // Initialize Leaflet flat map
  useEffect(() => {
    if (viewMode !== 'flat' || !mapRef.current) return;

    let mounted = true;

    const initMap = async () => {
      L = await import('leaflet');
      await import('leaflet/dist/leaflet.css');

      if (!mounted || !mapRef.current || mapInstance.current) return;

      const container = mapRef.current;
      if ((container as any)._leaflet_id) {
        (container as any)._leaflet_id = null;
      }

      const map = L.map(container, {
        center: [20, 10],
        zoom: 2.5,
        minZoom: 2,
        maxZoom: 8,
        zoomControl: true,
        attributionControl: false,
        worldCopyJump: true,
        preferCanvas: true,
      });

      mapInstance.current = map;

      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);

      // Create SVG overlay for arcs
      const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svgEl.setAttribute('class', styles.arcOverlay);
      svgEl.style.position = 'absolute';
      svgEl.style.overflow = 'visible';
      svgEl.style.pointerEvents = 'none';
      svgEl.style.zIndex = '400';
      map.getPanes().overlayPane.appendChild(svgEl);
      svgOverlay.current = svgEl;

      // Pre-create glow filter defs
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
      const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
      filter.id = 'arc-glow';
      const blur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
      blur.setAttribute('stdDeviation', '3');
      blur.setAttribute('result', 'glow');
      const merge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
      const m1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
      m1.setAttribute('in', 'glow');
      const m2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
      m2.setAttribute('in', 'SourceGraphic');
      merge.appendChild(m1);
      merge.appendChild(m2);
      filter.appendChild(blur);
      filter.appendChild(merge);
      defs.appendChild(filter);
      svgEl.appendChild(defs);

      // Add hub markers
      ML_HUBS.forEach(hub => {
        const color = riskColor(hub.riskLevel);
        const visualSize = hub.riskLevel === 'critical' ? 12 : hub.riskLevel === 'high' ? 10 : 8;
        const hitAreaSize = 48;

        const icon = L!.divIcon({
          className: styles.hubMarker,
          html: `
            <div style="width: ${hitAreaSize}px; height: ${hitAreaSize}px; display: flex; align-items: center; justify-content: center; background: transparent;">
              <div class="${styles.hubDot} ${hub.riskLevel === 'critical' ? styles.hubCritical : ''}" style="width:${visualSize}px;height:${visualSize}px;background:${color};box-shadow:0 0 ${visualSize}px ${color}40;"></div>
            </div>
            <div class="${styles.hubLabel}" style="margin-top: -12px;">${hub.name}</div>
          `,
          iconSize: [hitAreaSize, hitAreaSize],
          iconAnchor: [hitAreaSize / 2, hitAreaSize / 2],
        });

        const marker = L!.marker([hub.lat, hub.lng], { icon }).addTo(map);
        marker.on('click', () => onHubSelect(hub.id));
        marker.on('mouseover', (e: L.LeafletMouseEvent) => {
          setHoveredHub(hub);
          setTooltipPos({ x: e.containerPoint.x, y: e.containerPoint.y });
        });
        marker.on('mousemove', (e: L.LeafletMouseEvent) => {
          setTooltipPos({ x: e.containerPoint.x, y: e.containerPoint.y });
        });
        marker.on('mouseout', () => setHoveredHub(null));
      });

      // Draw arcs
      const drawArcs = () => {
        if (!svgOverlay.current || !mapInstance.current) return;
        const svg = svgOverlay.current;
        // Don't need viewBox because the svg is simply a container with overflow:visible in the overlayPane

        const defsEl = svg.querySelector('defs');
        while (svg.lastChild && svg.lastChild !== defsEl) {
          svg.removeChild(svg.lastChild);
        }

        ML_CORRIDORS.forEach(corridor => {
          const src = getHubById(corridor.source);
          const tgt = getHubById(corridor.target);
          if (!src || !tgt) return;

          const p1 = mapInstance.current!.latLngToLayerPoint([src.lat, src.lng]);
          const p2 = mapInstance.current!.latLngToLayerPoint([tgt.lat, tgt.lng]);

          const midX = (p1.x + p2.x) / 2;
          const midY = (p1.y + p2.y) / 2;
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const curvature = Math.min(dist * 0.3, 80);
          const cx = midX - (dy / dist) * curvature;
          const cy = midY + (dx / dist) * curvature;

          const level = riskScoreToLevel(corridor.riskScore);
          const color = riskColor(level);
          const width = Math.max(1, Math.min(3, corridor.volume / 1e9));

          const dPath = `M${p1.x},${p1.y} Q${cx},${cy} ${p2.x},${p2.y}`;

          const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path.setAttribute('d', dPath);
          path.setAttribute('fill', 'none');
          path.setAttribute('stroke', color);
          path.setAttribute('stroke-width', String(width));
          path.setAttribute('opacity', corridor.riskScore >= 90 ? '0.9' : '0.7');
          path.setAttribute('stroke-dasharray', '6,4');
          path.style.animation = 'nx-dash-flow 2s linear infinite';
          path.style.pointerEvents = 'none';
          if (corridor.riskScore >= 90) {
            path.setAttribute('filter', 'url(#arc-glow)');
          }
          svg.appendChild(path);

          const hitbox = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          hitbox.setAttribute('d', dPath);
          hitbox.setAttribute('fill', 'none');
          hitbox.setAttribute('stroke', 'transparent');
          hitbox.setAttribute('stroke-width', '20');
          hitbox.style.pointerEvents = 'stroke';
          hitbox.style.cursor = 'pointer';
          hitbox.addEventListener('click', (e) => { e.stopPropagation(); onCorridorSelect(corridor.id); });
          hitbox.addEventListener('mouseenter', () => {
            setHoveredCorridor(corridor);
            path.setAttribute('stroke-width', String(width + 2));
            path.setAttribute('opacity', '1');
          });
          hitbox.addEventListener('mousemove', (e) => {
            if (mapRef.current) {
              const rect = mapRef.current.getBoundingClientRect();
              setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
            }
          });
          hitbox.addEventListener('mouseleave', () => {
            setHoveredCorridor(null);
            path.setAttribute('stroke-width', String(width));
            path.setAttribute('opacity', corridor.riskScore >= 90 ? '0.9' : '0.7');
          });
          svg.appendChild(hitbox);
        });
      };

      const scheduleRedraw = () => {
        if (rafId.current) cancelAnimationFrame(rafId.current);
        rafId.current = requestAnimationFrame(drawArcs);
      };

      // Since the SVG is in the overlay pane, it will naturally pan and zoom!
      // We only need to redraw when the scale changes significantly, or layout changes
      map.on('zoomend', scheduleRedraw);
      map.on('moveend', scheduleRedraw);
      setTimeout(drawArcs, 200);
      setMapReady(true);
    };

    initMap();

    return () => {
      mounted = false;
      if (rafId.current) cancelAnimationFrame(rafId.current);
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
      if (mapRef.current) {
        (mapRef.current as any)._leaflet_id = null;
      }
    };
  }, [viewMode, onCorridorSelect, onHubSelect]);

  // Cleanup leaflet when switching to 3D
  useEffect(() => {
    if (viewMode === '3d' && mapInstance.current) {
      mapInstance.current.remove();
      mapInstance.current = null;
      if (mapRef.current) (mapRef.current as any)._leaflet_id = null;
    }
  }, [viewMode]);

  return (
    <div className={styles.mapContainer}>
      {/* 3D Globe View */}
      {viewMode === '3d' && GlobeComponent && (
        <div ref={globeContainerRef} style={{ width: '100%', height: '100%' }}>
          <GlobeComponent
            ref={globeRef}
            globeImageUrl=""
            backgroundColor="#131316"
            atmosphereColor="#ea580c"
            atmosphereAltitude={0.15}
            showAtmosphere={true}
            showGraticules={false}
            polygonsData={countries?.features || []}
            polygonCapColor={() => '#1c1c1f'}
            polygonSideColor={() => '#1c1c1f'}
            polygonStrokeColor={() => '#2A2A2D'}
            globeMaterial={undefined}
            customGlobeImage={undefined}

            pointsData={globePointsData}
            pointLat="lat"
            pointLng="lng"
            pointAltitude={0.01}
            pointRadius="size"
            pointColor="color"
            pointLabel={(d: any) => `<div style="font-family:'JetBrains Mono',monospace;font-size:10px;background:rgba(28,28,31,0.95);padding:8px 12px;border:1px solid #2A2A2D;border-radius:2px;color:#D4D4D4;backdrop-filter:blur(8px)"><b style="color:${d.color}">${d.name}</b><br/>Risk: ${d.riskLevel.toUpperCase()}<br/>Entities: ${d.entityCount}</div>`}
            onPointClick={handleGlobePointClick}

            arcsData={globeArcsData}
            arcStartLat="startLat"
            arcStartLng="startLng"
            arcEndLat="endLat"
            arcEndLng="endLng"
            arcColor="color"
            arcStroke="stroke"
            arcDashLength={0.4}
            arcDashGap={0.2}
            arcDashAnimateTime={2000}
            arcAltitudeAutoScale={0.4}
            arcLabel={(d: any) => `<div style="font-family:'JetBrains Mono',monospace;font-size:10px;background:rgba(28,28,31,0.95);padding:8px 12px;border:1px solid #2A2A2D;border-radius:2px;color:#D4D4D4;backdrop-filter:blur(8px)"><b style="color:${d.color}">${d.label || 'Corridor'}</b><br/>Risk: ${d.riskScore}<br/>Volume: ${formatVolume(d.volume)}<br/>Type: ${(d.typology || '').replace('_', ' ')}</div>`}
            onArcClick={handleGlobeArcClick}

            labelsData={globeLabelsData}
            labelLat="lat"
            labelLng="lng"
            labelText="text"
            labelSize="size"
            labelColor="color"
            labelResolution={2}
            labelAltitude={0.015}
            labelDotRadius={0}

            hexPolygonsData={[]}
            width={typeof window !== 'undefined' ? window.innerWidth : 1200}
            height={typeof window !== 'undefined' ? window.innerHeight - 40 : 800}

            onGlobeReady={() => {
              if (globeRef.current) {
                const controls = globeRef.current.controls();
                if (controls) {
                  controls.autoRotate = true;
                  controls.autoRotateSpeed = 0.3;
                  controls.enableDamping = true;
                  controls.dampingFactor = 0.1;
                }
                // Set initial POV
                globeRef.current.pointOfView({ lat: 20, lng: 10, altitude: 2.5 }, 0);

                // Style the globe material for dark vector look
                const scene = globeRef.current.scene();
                if (scene) {
                  scene.traverse((obj: any) => {
                    if (obj.type === 'Mesh' && obj.material) {
                      // Dark globe surface
                      if (obj.material.color) {
                        obj.material.color.setHex(0x181818);
                      }
                    }
                  });
                }
              }
            }}
          />
        </div>
      )}

      {/* Flat Map View */}
      {viewMode === 'flat' && (
        <div ref={mapRef} className={styles.map} />
      )}

      {/* Stats Overlay */}
      <div className={styles.statsOverlay}>
        <div className={styles.statsTitle}>GLOBAL THREAT MONITOR</div>
        <div className={styles.statsGrid}>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>CORRIDORS</span>
            <span className={styles.statValue}>{ML_CORRIDORS.length}</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>HUBS</span>
            <span className={styles.statValue}>{ML_HUBS.length}</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>TOTAL VOL</span>
            <span className={styles.statValue}>{formatVolume(ML_CORRIDORS.reduce((s, c) => s + c.volume, 0))}</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>CRITICAL</span>
            <span className={`${styles.statValue} ${styles.statDanger}`}>
              {ML_CORRIDORS.filter(c => c.riskScore >= 90).length}
            </span>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className={styles.viewToggle}>
          <button
            className={`${styles.viewToggleBtn} ${viewMode === '3d' ? styles.viewToggleBtnActive : ''}`}
            onClick={() => setViewMode('3d')}
          >
            3D GLOBE
          </button>
          <button
            className={`${styles.viewToggleBtn} ${viewMode === 'flat' ? styles.viewToggleBtnActive : ''}`}
            onClick={() => setViewMode('flat')}
          >
            FLAT MAP
          </button>
        </div>

        <div className={styles.statsHint}>Click a corridor or hub to investigate</div>
      </div>

      {/* Hub Tooltip (flat map) */}
      {hoveredHub && viewMode === 'flat' && (
        <div className={styles.tooltip} style={{ left: tooltipPos.x + 16, top: tooltipPos.y - 10 }}>
          <div className={styles.tooltipHeader}>
            <span className={styles.tooltipType}>HUB</span>
            <span className={styles.tooltipRisk} style={{ color: riskColor(hoveredHub.riskLevel) }}>
              {hoveredHub.riskLevel.toUpperCase()}
            </span>
          </div>
          <div className={styles.tooltipName}>{hoveredHub.name}</div>
          <div className={styles.tooltipRow}>
            <span>Region</span><span>{hoveredHub.region}</span>
          </div>
          <div className={styles.tooltipRow}>
            <span>Entities</span><span>{hoveredHub.entityCount}</span>
          </div>
        </div>
      )}

      {/* Corridor Tooltip (flat map) */}
      {hoveredCorridor && viewMode === 'flat' && (
        <div className={styles.tooltip} style={{ left: tooltipPos.x + 16, top: tooltipPos.y - 10 }}>
          <div className={styles.tooltipHeader}>
            <span className={styles.tooltipType}>CORRIDOR</span>
            <span className={styles.tooltipRisk} style={{ color: riskColor(riskScoreToLevel(hoveredCorridor.riskScore)) }}>
              RISK {hoveredCorridor.riskScore}
            </span>
          </div>
          <div className={styles.tooltipName}>{hoveredCorridor.label || `${hoveredCorridor.source} → ${hoveredCorridor.target}`}</div>
          <div className={styles.tooltipRow}>
            <span>Volume</span><span>{formatVolume(hoveredCorridor.volume)}</span>
          </div>
          <div className={styles.tooltipRow}>
            <span>Typology</span><span>{hoveredCorridor.typology.replace('_', ' ')}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export type { Props as GlobalThreatMapProps };
