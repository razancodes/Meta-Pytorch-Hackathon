// ═══════════════════════════════════════════════════════════════
// NEXUS — Geographic Data & ML Hub Coordinates
// ═══════════════════════════════════════════════════════════════

import { MLHub, MLCorridor } from './types';

// ── Major Money Laundering Hub Locations ──────────────────────
export const ML_HUBS: MLHub[] = [
  { id: 'panama',     name: 'Panama City',     lat: 8.9824,   lng: -79.5199,  riskLevel: 'critical', entityCount: 340, region: 'Americas' },
  { id: 'cayman',     name: 'Cayman Islands',  lat: 19.3133,  lng: -81.2546,  riskLevel: 'critical', entityCount: 520, region: 'Americas' },
  { id: 'bvi',        name: 'British Virgin Islands', lat: 18.4207, lng: -64.6400, riskLevel: 'critical', entityCount: 680, region: 'Americas' },
  { id: 'london',     name: 'London',          lat: 51.5074,  lng: -0.1278,   riskLevel: 'high',     entityCount: 890, region: 'Europe' },
  { id: 'zurich',     name: 'Zürich',          lat: 47.3769,  lng: 8.5417,    riskLevel: 'high',     entityCount: 420, region: 'Europe' },
  { id: 'cyprus',      name: 'Nicosia',         lat: 35.1856,  lng: 33.3823,   riskLevel: 'high',     entityCount: 310, region: 'Europe' },
  { id: 'dubai',      name: 'Dubai',           lat: 25.2048,  lng: 55.2708,   riskLevel: 'critical', entityCount: 760, region: 'MENA' },
  { id: 'singapore',  name: 'Singapore',       lat: 1.3521,   lng: 103.8198,  riskLevel: 'high',     entityCount: 540, region: 'Asia' },
  { id: 'hongkong',   name: 'Hong Kong',       lat: 22.3193,  lng: 114.1694,  riskLevel: 'high',     entityCount: 620, region: 'Asia' },
  { id: 'moscow',     name: 'Moscow',          lat: 55.7558,  lng: 37.6173,   riskLevel: 'critical', entityCount: 480, region: 'Europe' },
  { id: 'lagos',      name: 'Lagos',           lat: 6.5244,   lng: 3.3792,    riskLevel: 'high',     entityCount: 290, region: 'Africa' },
  { id: 'mumbai',     name: 'Mumbai',          lat: 19.0760,  lng: 72.8777,   riskLevel: 'medium',   entityCount: 350, region: 'Asia' },
  { id: 'seychelles', name: 'Seychelles',      lat: -4.6796,  lng: 55.4920,   riskLevel: 'critical', entityCount: 190, region: 'Africa' },
  { id: 'malta',      name: 'Malta',           lat: 35.8989,  lng: 14.5146,   riskLevel: 'high',     entityCount: 210, region: 'Europe' },
  { id: 'kualalumpur',name: 'Kuala Lumpur',    lat: 3.1390,   lng: 101.6869,  riskLevel: 'medium',   entityCount: 280, region: 'Asia' },
  { id: 'newyork',    name: 'New York',        lat: 40.7128,  lng: -74.0060,  riskLevel: 'medium',   entityCount: 450, region: 'Americas' },
  { id: 'liechtenstein', name: 'Vaduz',        lat: 47.1410,  lng: 9.5215,    riskLevel: 'high',     entityCount: 130, region: 'Europe' },
  { id: 'bahamas',    name: 'Nassau',          lat: 25.0343,  lng: -77.3963,  riskLevel: 'high',     entityCount: 240, region: 'Americas' },
  { id: 'macau',      name: 'Macau',           lat: 22.1987,  lng: 113.5439,  riskLevel: 'high',     entityCount: 180, region: 'Asia' },
  { id: 'myanmar',    name: 'Yangon',          lat: 16.8661,  lng: 96.1951,   riskLevel: 'critical', entityCount: 110, region: 'Asia' },
];

// ── Pre-built ML Corridors (static threat intel) ─────────────
export const ML_CORRIDORS: MLCorridor[] = [
  { id: 'c-01', source: 'panama',      target: 'cayman',     volume: 2400000000, riskScore: 92, typology: 'layering',        label: 'Shell Company Pipeline' },
  { id: 'c-02', source: 'bvi',         target: 'london',     volume: 1800000000, riskScore: 88, typology: 'layering',        label: 'Offshore → City Corridor' },
  { id: 'c-03', source: 'dubai',       target: 'mumbai',     volume: 3200000000, riskScore: 85, typology: 'trade_based_ml',  label: 'Trade Invoice Manipulation' },
  { id: 'c-04', source: 'hongkong',    target: 'singapore',  volume: 2100000000, riskScore: 78, typology: 'layering',        label: 'Asia-Pacific Layer' },
  { id: 'c-05', source: 'moscow',      target: 'cyprus',     volume: 4100000000, riskScore: 95, typology: 'layering',        label: 'Russian Capital Flight' },
  { id: 'c-06', source: 'lagos',       target: 'london',     volume: 1200000000, riskScore: 82, typology: 'structuring',     label: 'West Africa → UK Pipeline' },
  { id: 'c-07', source: 'zurich',      target: 'seychelles', volume: 890000000,  riskScore: 90, typology: 'layering',        label: 'Alpine → Island Route' },
  { id: 'c-08', source: 'kualalumpur', target: 'singapore',  volume: 681000000,  riskScore: 94, typology: 'layering',        label: '1MDB Corridor', caseId: 'case-1mdb' },
  { id: 'c-09', source: 'cayman',      target: 'newyork',    volume: 1500000000, riskScore: 75, typology: 'layering',        label: 'Caribbean Integration' },
  { id: 'c-10', source: 'dubai',       target: 'london',     volume: 2800000000, riskScore: 87, typology: 'trade_based_ml',  label: 'Gold & Commodities' },
  { id: 'c-11', source: 'malta',       target: 'liechtenstein', volume: 450000000, riskScore: 80, typology: 'layering',      label: 'EU Regulatory Arbitrage' },
  { id: 'c-12', source: 'seychelles',  target: 'bvi',        volume: 720000000,  riskScore: 91, typology: 'layering',        label: 'Island Hop Chain' },
  { id: 'c-13', source: 'macau',       target: 'hongkong',   volume: 1100000000, riskScore: 83, typology: 'structuring',     label: 'Casino → Banking' },
  { id: 'c-14', source: 'myanmar',     target: 'singapore',  volume: 340000000,  riskScore: 96, typology: 'trade_based_ml',  label: 'Narco-Trade Flow' },
  { id: 'c-15', source: 'bahamas',     target: 'panama',     volume: 560000000,  riskScore: 79, typology: 'layering',        label: 'Caribbean Circuit' },
];

// ── Jurisdiction → Coordinates Lookup ────────────────────────
export const JURISDICTION_COORDS: Record<string, [number, number]> = {
  'malaysia':               [3.1390,   101.6869],
  'kuala lumpur':           [3.1390,   101.6869],
  'seychelles':             [-4.6796,  55.4920],
  'british virgin islands': [18.4207,  -64.6400],
  'bvi':                    [18.4207,  -64.6400],
  'singapore':              [1.3521,   103.8198],
  'hong kong':              [22.3193,  114.1694],
  'london':                 [51.5074,  -0.1278],
  'united kingdom':         [51.5074,  -0.1278],
  'switzerland':            [47.3769,  8.5417],
  'zurich':                 [47.3769,  8.5417],
  'panama':                 [8.9824,   -79.5199],
  'cayman islands':         [19.3133,  -81.2546],
  'cyprus':                  [35.1856,  33.3823],
  'dubai':                  [25.2048,  55.2708],
  'uae':                    [25.2048,  55.2708],
  'united arab emirates':   [25.2048,  55.2708],
  'united states':          [40.7128,  -74.0060],
  'usa':                    [40.7128,  -74.0060],
  'new york':               [40.7128,  -74.0060],
  'moscow':                 [55.7558,  37.6173],
  'russia':                 [55.7558,  37.6173],
  'lagos':                  [6.5244,   3.3792],
  'nigeria':                [6.5244,   3.3792],
  'mumbai':                 [19.0760,  72.8777],
  'india':                  [19.0760,  72.8777],
  'malta':                  [35.8989,  14.5146],
  'liechtenstein':          [47.1410,  9.5215],
  'bahamas':                [25.0343,  -77.3963],
  'macau':                  [22.1987,  113.5439],
  'myanmar':                [16.8661,  96.1951],
  'isle of man':            [54.2361,  -4.5481],
  'jersey':                 [49.2144,  -2.1312],
  'belize':                 [17.1899,  -88.4976],
  'mauritius':              [-20.3484, 57.5522],
  'geneva':                 [46.2044,  6.1432],
  'frankfurt':              [50.1109,  8.6821],
  'germany':                [50.1109,  8.6821],
  'amsterdam':              [52.3676,  4.9041],
  'tokyo':                  [35.6762,  139.6503],
  'japan':                  [35.6762,  139.6503],
  'toronto':                [43.6532,  -79.3832],
  'canada':                 [43.6532,  -79.3832],
};

// ── Helpers ──────────────────────────────────────────────────
export function getHubById(id: string): MLHub | undefined {
  return ML_HUBS.find(h => h.id === id);
}

export function getCorridorCoords(corridor: MLCorridor): { from: [number, number]; to: [number, number] } | null {
  const src = getHubById(corridor.source);
  const tgt = getHubById(corridor.target);
  if (!src || !tgt) return null;
  return {
    from: [src.lat, src.lng],
    to: [tgt.lat, tgt.lng],
  };
}

export function geocodeJurisdiction(jurisdiction: string): [number, number] | null {
  const key = jurisdiction.toLowerCase().trim();
  return JURISDICTION_COORDS[key] || null;
}

export function riskColor(level: string): string {
  switch (level) {
    case 'critical': return '#E11D48';
    case 'high':     return '#EA580C';
    case 'medium':   return '#EAB308';
    case 'low':      return '#22C55E';
    default:         return '#404040';
  }
}

export function riskScoreToLevel(score: number): 'low' | 'medium' | 'high' | 'critical' {
  if (score >= 90) return 'critical';
  if (score >= 75) return 'high';
  if (score >= 50) return 'medium';
  return 'low';
}

export function formatVolume(volume: number): string {
  if (volume >= 1e9) return `$${(volume / 1e9).toFixed(1)}B`;
  if (volume >= 1e6) return `$${(volume / 1e6).toFixed(0)}M`;
  if (volume >= 1e3) return `$${(volume / 1e3).toFixed(0)}K`;
  return `$${volume}`;
}
