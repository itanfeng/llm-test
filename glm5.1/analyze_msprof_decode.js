#!/usr/bin/env node

// Stream a large msprof Chrome-trace JSON file without loading it all into
// memory. The report separates attention and post-attention/MoE time so that
// prefetch regressions are not incorrectly attributed to sparse attention.
const fs = require('fs');

const args = process.argv.slice(2);
const readIntOption = (name, fallback) => {
  const prefix = `${name}=`;
  const raw = args.find(arg => arg.startsWith(prefix));
  return raw ? Number.parseInt(raw.slice(prefix.length), 10) : fallback;
};
const file = args.find(arg => !arg.startsWith('--'));
const numLayers = readIntOption('--num-layers', 20);
const firstKDense = readIntOption('--first-k-dense', 3);
const tailSteps = readIntOption('--tail-steps', 20);
const timelineLayer = readIntOption('--timeline-layer', 0);
const timelineFromEnd = readIntOption('--timeline-from-end', 1);
const pretty = args.includes('--pretty');
if (!file) {
  throw new Error(
    'usage: node analyze_msprof_decode.js FILE [--num-layers=20] ' +
    '[--first-k-dense=3] [--tail-steps=20] [--pretty]'
  );
}

const keep = /mla_preprocess|LightningIndexer|AsuHbmIndexLookup|AsuHbmIndexMaintain|AsuKvGather|HiSparseFlashAttention|DaAttentionMerge|batch_matmul_transpose|MoeGatingTopK|MoeDistributeDispatchV2|MoeDistributeCombineV2|GroupedMatmul|ScatterNd|UpdateMean|indexer|Rope|MatMul|LayerNorm|InplaceCopy|Cast|Muls|Adds|Subs|Clamp|FillScalar|AscendQuant/i;
const events = [];
const namesByPidTid = new Map();
const pidNames = new Map();

function consume(obj) {
  if (!obj || typeof obj !== 'object') return;
  if (obj.ph === 'M' && obj.name === 'process_name') {
    pidNames.set(Number(obj.pid), obj.args?.name || '');
    return;
  }
  if (obj.ph !== 'X') return;
  const dur = Number(obj.dur);
  const ts = Number(obj.ts);
  if (!Number.isFinite(dur) || dur <= 0 || !Number.isFinite(ts)) return;
  const name = String(obj.name || '');
  const pid = Number(obj.pid);
  const tid = Number(obj.tid);
  const key = `${pid}|${tid}|${name}`;
  namesByPidTid.set(key, (namesByPidTid.get(key) || 0) + 1);
  if (!keep.test(name)) return;
  events.push({name, pid, tid, ts, dur, task: Number(obj.args?.['Task Id'] ?? obj.args?.['Task ID'] ?? -1)});
}

let depth = 0;
let inString = false;
let escape = false;
let objectBuffer = '';
const stream = fs.createReadStream(file, {encoding: 'utf8', highWaterMark: 8 * 1024 * 1024});
stream.on('data', chunk => {
  for (let i = 0; i < chunk.length; i++) {
    const c = chunk[i];
    if (depth > 0 || c === '{') objectBuffer += c;
    if (inString) {
      if (escape) escape = false;
      else if (c === '\\') escape = true;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') { inString = true; continue; }
    if (c === '{') {
      depth++;
    } else if (c === '}') {
      depth--;
      if (depth === 0) {
        try { consume(JSON.parse(objectBuffer)); } catch (_) {}
        objectBuffer = '';
      }
    }
  }
});

function quantile(xs, q) {
  if (!xs.length) return null;
  const a = [...xs].sort((x,y) => x-y);
  return a[Math.min(a.length - 1, Math.floor(q * a.length))];
}
function stats(xs) {
  if (!xs.length) return {n: 0};
  return {n: xs.length, avg: xs.reduce((a,b)=>a+b,0)/xs.length, p50: quantile(xs,.5), p90: quantile(xs,.9), min: Math.min(...xs), max: Math.max(...xs)};
}
function dedup(es) {
  const seen = new Set();
  return es.filter(e => {
    const k = `${e.name}|${e.pid}|${e.tid}|${e.ts}|${e.dur}|${e.task}`;
    if (seen.has(k)) return false;
    seen.add(k); return true;
  });
}
function sumDur(es, re) { return es.filter(e=>re.test(e.name)).reduce((a,e)=>a+e.dur,0); }

stream.on('end', () => {
  const es = dedup(events);
  const pidScore = new Map();
  for (const e of es) if (/HiSparseFlashAttention|AsuKvGather|mla_preprocess/.test(e.name)) pidScore.set(e.pid, (pidScore.get(e.pid)||0)+1);
  const pid = [...pidScore].sort((a,b)=>b[1]-a[1])[0]?.[0];
  const hw = es.filter(e=>e.pid===pid);
  const count = (re, tid) => hw.filter(e=>(tid===undefined||e.tid===tid)&&re.test(e.name)).length;
  const tids = [...new Set(hw.map(e=>e.tid))];
  const main = tids.sort((a,b)=>count(/HiSparseFlashAttention/,b)-count(/HiSparseFlashAttention/,a))[0];
  const prefetchCandidate = [...tids].sort((a,b)=>count(/LightningIndexerHiCached/,b)-count(/LightningIndexerHiCached/,a))[0];
  const prefetch = count(/LightningIndexerHiCached/, prefetchCandidate) > 0 ? prefetchCandidate : null;
  const gather = [...tids].sort((a,b)=>count(/AsuKvGather/,b)-count(/AsuKvGather/,a))[0];
  const maintain = [...tids].sort((a,b)=>count(/AsuHbmIndexMaintain/,b)-count(/AsuHbmIndexMaintain/,a))[0];
  const mainMla = hw.filter(e=>e.tid===main && e.name==='mla_preprocess').sort((a,b)=>a.ts-b.ts);
  const taskCounts = new Map();
  for (const e of mainMla) taskCounts.set(e.task,(taskCounts.get(e.task)||0)+1);
  const layerTasks = [...taskCounts].sort((a,b)=>b[1]-a[1]).slice(0,numLayers).sort((a,b)=>a[0]-b[0]).map(x=>x[0]);
  const layer0 = mainMla.filter(e=>e.task===layerTasks[0]);
  const steps = [];
  for (let i=0;i<layer0.length-1;i++) {
    const start=layer0[i].ts, end=layer0[i+1].ts;
    const anchors=[];
    for (const task of layerTasks) {
      const found=mainMla.find(e=>e.task===task && e.ts>=start && e.ts<end);
      if (found) anchors.push(found);
    }
    if (anchors.length!==numLayers) continue;
    const layers=[];
    for (let l=0;l<numLayers-1;l++) {
      const a=anchors[l], b=anchors[l+1];
      const interval=hw.filter(e=>e.ts>=a.ts&&e.ts<b.ts);
      const merges=interval.filter(e=>e.tid===main&&e.name==='DaAttentionMerge');
      const attnEnd=merges.length?Math.max(...merges.map(e=>e.ts+e.dur)):a.ts;
      layers.push({layer:l+1,start:a.ts,end:b.ts,lat:b.ts-a.ts,attn:attnEnd-a.ts,tail:b.ts-attnEnd,moeDur:sumDur(interval,/MoeGatingTopK|MoeDistributeDispatchV2|MoeDistributeCombineV2|GroupedMatmul/)});
    }
    steps.push({start,end,dur:end-start,layers});
  }
  const stable=steps.slice(-tailSteps);
  const selected=steps.at(-2);
  const groupStats = (ls) => ({
    lat: stats(ls.map(x=>x.lat)), attn: stats(ls.map(x=>x.attn)), tail: stats(ls.map(x=>x.tail)), moeDur: stats(ls.map(x=>x.moeDur))
  });
  const dense=stable.flatMap(s=>s.layers.filter(x=>x.layer<=firstKDense));
  const moe=stable.flatMap(s=>s.layers.filter(x=>x.layer>firstKDense));
  const firstMoeLayer=stable.map(s=>s.layers[firstKDense]).filter(Boolean);
  const windowStart=stable[0]?.start ?? -Infinity, windowEnd=stable.at(-1)?.end ?? Infinity;
  const window=hw.filter(e=>e.ts>=windowStart&&e.ts<windowEnd);
  const kstats=(name,tid)=>stats(window.filter(e=>e.name===name&&(tid===undefined||e.tid===tid)).map(e=>e.dur));
  const prefetchKernelGroups = new Map();
  if (prefetch !== null) {
    for (const event of window.filter(e=>e.tid===prefetch)) {
      if (!prefetchKernelGroups.has(event.name)) prefetchKernelGroups.set(event.name, []);
      prefetchKernelGroups.get(event.name).push(event.dur);
    }
  }
  const prefetchKernelStats = [...prefetchKernelGroups].map(([name,durations])=>({
    name,
    total:durations.reduce((a,b)=>a+b,0),
    ...stats(durations),
  })).sort((a,b)=>b.total-a.total).slice(0,30);
  const gatherEvents = window.filter(e=>e.tid===gather&&e.name==='AsuKvGather').sort((a,b)=>a.ts-b.ts);
  const prefetchLookups = prefetch === null ? [] : window.filter(e=>e.tid===prefetch&&/AsuHbmIndexLookup/.test(e.name)).sort((a,b)=>a.ts-b.ts);
  const prefetchGatherSet = new Set();
  let gatherCursor = 0;
  for (const lookup of prefetchLookups) {
    const lookupEnd = lookup.ts + lookup.dur;
    while (gatherCursor < gatherEvents.length && gatherEvents[gatherCursor].ts < lookupEnd) gatherCursor++;
    if (gatherCursor < gatherEvents.length) {
      prefetchGatherSet.add(gatherEvents[gatherCursor]);
      gatherCursor++;
    }
  }
  const prefetchGathers = gatherEvents.filter(e=>prefetchGatherSet.has(e));
  const mainGathers = gatherEvents.filter(e=>!prefetchGatherSet.has(e));
  const overlapDuration = (event, others) => others.reduce((total,other)=>
    total + Math.max(0, Math.min(event.ts+event.dur,other.ts+other.dur)-Math.max(event.ts,other.ts)), 0);
  const moeEvents = window.filter(e=>e.tid===main&&/MoeGatingTopK|MoeDistributeDispatchV2|MoeDistributeCombineV2|GroupedMatmul/.test(e.name));
  const moeGroups = new Map();
  for (const event of moeEvents) {
    if (!moeGroups.has(event.name)) moeGroups.set(event.name, []);
    moeGroups.get(event.name).push(event);
  }
  const moeKernelStats = [...moeGroups].map(([name,kernelEvents])=>{
    const prefetchOverlap = kernelEvents.map(e=>overlapDuration(e,prefetchGathers));
    const mainOverlap = kernelEvents.map(e=>overlapDuration(e,mainGathers));
    const overlappedDurations = kernelEvents.filter((_,i)=>prefetchOverlap[i]>0).map(e=>e.dur);
    const cleanDurations = kernelEvents.filter((_,i)=>prefetchOverlap[i]===0).map(e=>e.dur);
    return {
      name,
      duration:stats(kernelEvents.map(e=>e.dur)),
      prefetchGatherOverlap:{count:prefetchOverlap.filter(x=>x>0).length,total:prefetchOverlap.reduce((a,b)=>a+b,0),overlappedDuration:stats(overlappedDurations),cleanDuration:stats(cleanDurations)},
      mainGatherOverlap:{count:mainOverlap.filter(x=>x>0).length,total:mainOverlap.reduce((a,b)=>a+b,0)},
    };
  });
  const targetStep = steps.at(-timelineFromEnd);
  const targetLayer = targetStep?.layers[timelineLayer-1];
  const timeline = !targetLayer ? [] : hw.filter(e=>e.ts>=targetLayer.start&&e.ts<targetLayer.end&&(
    e.name==='AsuKvGather'||e.name==='mla_preprocess'||e.name==='LightningIndexerHiCached'||
    /AsuHbmIndexLookup|AsuHbmIndexMaintain|HiSparseFlashAttention|DaAttentionMerge|batch_matmul_transpose|MoeGatingTopK|MoeDistributeDispatchV2|MoeDistributeCombineV2|GroupedMatmul/.test(e.name)
  )).sort((a,b)=>a.ts-b.ts).map(e=>({name:e.name,stream:e.tid,task:e.task,rel:e.ts-targetLayer.start,dur:e.dur}));
  const topNames = prefetch === null ? [] : [...namesByPidTid].filter(([k])=>k.startsWith(`${pid}|${prefetch}|`)).map(([k,v])=>[k.split('|').slice(2).join('|'),v]).sort((a,b)=>b[1]-a[1]).slice(0,30);
  const out={file,pid,pidName:pidNames.get(pid),streams:{main,prefetch,gather,maintain},layerTasks,stepCount:steps.length,
    stableStep:stats(stable.map(s=>s.dur)),
    selectedSecondLast:selected?{dur:selected.dur,firstMoeLayer:selected.layers[firstKDense],layers:selected.layers}:null,
    aggregate:{dense:groupStats(dense),moe:groupStats(moe),firstMoeLayer:groupStats(firstMoeLayer)},
    kernels:{mainMla:kstats('mla_preprocess',main),prefetchMla:kstats('mla_preprocess',prefetch),prefetchIndexer:kstats('LightningIndexerHiCached',prefetch),prefetchLookup:kstats('AsuHbmIndexLookup',prefetch),allGather:kstats('AsuKvGather',gather)},
    prefetchKernelStats,
    gatherStats:{main:stats(mainGathers.map(e=>e.dur)),prefetch:stats(prefetchGathers.map(e=>e.dur))},
    moeKernelStats,
    timeline,
    prefetchNameCounts:topNames,
    firstMoeLayerHistory:steps.slice(-10).map((s,i)=>({fromEnd:10-i,stepDur:s.dur,...s.layers[firstKDense]}))};
  console.log(JSON.stringify(out, null, pretty ? 2 : 0));
});
