'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import styles from './globe.module.css';
import { ML_HUBS, ML_CORRIDORS, getHubById, riskColor, formatVolume, riskScoreToLevel } from '@/lib/geoData';
import type { MLHub, MLCorridor } from '@/lib/types';

// Dynamic import to avoid SSR issues with Leaflet
let L: typeof import('leaflet') | null = null;

interface Props {
  onCorridorSelect: (corridorId: string) => void;
  onHubSelect: (hubId: string) => void;
}

export default function GlobalThreatMap({ onCorridorSelect, onHubSelect }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<L.Map | null>(null);
  const svgOverlay = useRef<SVGSVGElement | null>(null);
  const [hoveredHub, setHoveredHub] = useState<MLHub | null>(null);
  const hoveredHubRef = useRef<MLHub | null>(null);
  const [hoveredCorridor, setHoveredCorridor] = useState<MLCorridor | null>(null);
  const hoveredCorridorRef = useRef<MLCorridor | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [mapReady, setMapReady] = useState(false);

  // Initialize Leaflet map
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    const initMap = async () => {
      L = await import('leaflet');
      await import('leaflet/dist/leaflet.css');

      const map = L.map(mapRef.current!, {
        center: [20, 10],
        zoom: 2.5,
        minZoom: 2,
        maxZoom: 8,
        zoomControl: true,
        attributionControl: false,
        worldCopyJump: true,
      });

      // CartoDB Dark Matter tiles
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);

      mapInstance.current = map;

      // Create SVG overlay for arcs
      const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svgEl.setAttribute('class', styles.arcOverlay);
      svgEl.style.position = 'absolute';
      svgEl.style.top = '0';
      svgEl.style.left = '0';
      svgEl.style.width = '100%';
      svgEl.style.height = '100%';
      svgEl.style.pointerEvents = 'none';
      svgEl.style.zIndex = '400';
      map.getContainer().appendChild(svgEl);
      svgOverlay.current = svgEl;

      // Add hub markers
      ML_HUBS.forEach(hub => {
        const color = riskColor(hub.riskLevel);
        const visualSize = hub.riskLevel === 'critical' ? 12 : hub.riskLevel === 'high' ? 10 : 8; // Slightly larger visually
        const hitAreaSize = 48; // Massive invisible hit area for easy clicking

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
          hoveredHubRef.current = hub;
          setHoveredHub(hub);
          setTooltipPos({ x: e.containerPoint.x, y: e.containerPoint.y });
        });
        marker.on('mousemove', (e: L.LeafletMouseEvent) => {
          setTooltipPos({ x: e.containerPoint.x, y: e.containerPoint.y });
        });
        marker.on('mouseout', () => {
          hoveredHubRef.current = null;
          setHoveredHub(null);
        });
      });

      // Draw arcs on map move/zoom
      const drawArcs = () => {
        if (!svgOverlay.current || !mapInstance.current) return;
        const svg = svgOverlay.current;
        const mapSize = mapInstance.current.getSize();
        svg.setAttribute('viewBox', `0 0 ${mapSize.x} ${mapSize.y}`);

        // Clear old arcs
        while (svg.firstChild) svg.removeChild(svg.firstChild);

        // Defs for glow filter
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
        svg.appendChild(defs);

        ML_CORRIDORS.forEach(corridor => {
          const src = getHubById(corridor.source);
          const tgt = getHubById(corridor.target);
          if (!src || !tgt) return;

          const p1 = mapInstance.current!.latLngToContainerPoint([src.lat, src.lng]);
          const p2 = mapInstance.current!.latLngToContainerPoint([tgt.lat, tgt.lng]);

          // Quadratic bezier for arc effect
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

          // Visual path
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
            path.setAttribute('opacity', '0.9');
          }
          svg.appendChild(path);

          // Invisible hitbox for easier hover
          const hitbox = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          hitbox.setAttribute('d', dPath);
          hitbox.setAttribute('fill', 'none');
          hitbox.setAttribute('stroke', 'transparent');
          hitbox.setAttribute('stroke-width', '20');
          hitbox.style.pointerEvents = 'stroke';
          hitbox.style.cursor = 'pointer';

          hitbox.addEventListener('click', (e) => {
            e.stopPropagation();
            onCorridorSelect(corridor.id);
          });
          hitbox.addEventListener('mouseenter', (e) => {
            hoveredCorridorRef.current = corridor;
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
            hoveredCorridorRef.current = null;
            setHoveredCorridor(null);
            path.setAttribute('stroke-width', String(width));
            path.setAttribute('opacity', corridor.riskScore >= 90 ? '0.9' : '0.7');
          });

          svg.appendChild(hitbox);
        });
      };

      map.on('move', drawArcs);
      map.on('moveend', drawArcs);
      map.on('zoom', drawArcs);
      map.on('zoomend', drawArcs);
      // Initial draw after a small delay for the map to settle
      setTimeout(drawArcs, 200);
      setMapReady(true);
    };

    initMap();

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, [onCorridorSelect, onHubSelect]);

  const flyTo = useCallback((lat: number, lng: number, zoom: number = 6) => {
    mapInstance.current?.flyTo([lat, lng], zoom, { duration: 1.5 });
  }, []);

  return (
    <div className={styles.mapContainer}>
      <div ref={mapRef} className={styles.map} />

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
        <div className={styles.statsHint}>Click a corridor or hub to investigate</div>
      </div>

      {/* Hub Tooltip */}
      {hoveredHub && (
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

      {/* Corridor Tooltip */}
      {hoveredCorridor && (
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

// Expose flyTo for the parent transition
export type { Props as GlobalThreatMapProps };
