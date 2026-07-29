#!/usr/bin/env node

// Analyze attention-to-MoE gaps and prefetch overlap in a large msprof
// Chrome-trace JSON file without loading the complete trace into memory.
//
// Usage:
//   node msprof_gap_analysis.js TRACE.json > report.json
//   node msprof_gap_analysis.js TRACE.json --pretty > report.json
//   node msprof_gap_analysis.js TRACE.json --num-layers=20 \
//     --first-moe-layer=4 --pretty > report.json
//
// Run this script once for the base trace and once for the prefetch trace.
// Pair `records` by `fromEnd` and `layer` when comparing two reports; using
// `fromEnd` avoids aligning warm-up steps with steady-state decode steps.

"use strict";

const fs = require("fs");

const args = process.argv.slice(2);
const file = args.find(arg => !arg.startsWith("--"));
const readIntOption = (name, fallback) => {
  const prefix = `${name}=`;
  const raw = args.find(arg => arg.startsWith(prefix));
  return raw === undefined
    ? fallback
    : Number.parseInt(raw.slice(prefix.length), 10);
};
const numLayers = readIntOption("--num-layers", 20);
const firstMoeLayer = readIntOption("--first-moe-layer", 4);
const pretty = args.includes("--pretty");

if (!file || !Number.isInteger(numLayers) || numLayers < 2 ||
    !Number.isInteger(firstMoeLayer) || firstMoeLayer < 1 ||
    firstMoeLayer >= numLayers) {
  throw new Error(
    "usage: node msprof_gap_analysis.js TRACE.json " +
    "[--num-layers=20] [--first-moe-layer=4] [--pretty]",
  );
}

const KEEP_EVENT = new RegExp([
  "mla_preprocess",
  "mla_query_preprocess",
  "LightningIndexerHiCached",
  "AsuHbmIndexLookupOpt",
  "AsuHbmIndexLookup",
  "AsuKvGather",
  "HiSparseFlashAttention",
  "DaAttentionMerge",
  "MoeGatingTopK",
  "MoeDistributeDispatchV2",
  "GroupedMatmul",
  "MoeDistributeCombineV2",
].join("|"));

const events = [];
const processNames = new Map();

function consume(object) {
  if (!object || typeof object !== "object") return;
  if (object.ph === "M" && object.name === "process_name") {
    processNames.set(Number(object.pid), object.args?.name ?? "");
    return;
  }
  if (object.ph !== "X") return;

  const name = String(object.name ?? "");
  if (!KEEP_EVENT.test(name)) return;
  const ts = Number(object.ts);
  const dur = Number(object.dur);
  if (!Number.isFinite(ts) || !Number.isFinite(dur) || dur <= 0) return;

  events.push({
    name,
    pid: Number(object.pid),
    tid: Number(object.tid),
    ts,
    dur,
    task: Number(
      object.args?.["Task Id"] ?? object.args?.["Task ID"] ?? -1,
    ),
  });
}

let depth = 0;
let inString = false;
let escaped = false;
let objectBuffer = "";
const stream = fs.createReadStream(file, {
  encoding: "utf8",
  highWaterMark: 8 * 1024 * 1024,
});

stream.on("data", chunk => {
  for (const char of chunk) {
    if (depth > 0 || char === "{") objectBuffer += char;
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
    } else if (char === "{") {
      depth++;
    } else if (char === "}") {
      depth--;
      if (depth === 0) {
        try {
          consume(JSON.parse(objectBuffer));
        } catch (_) {
          // Chrome traces may contain non-event JSON objects. Ignore them.
        }
        objectBuffer = "";
      }
    }
  }
});

function quantile(values, q) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.floor(q * sorted.length),
  );
  return sorted[index];
}

function stats(values) {
  const finiteValues = values.filter(Number.isFinite);
  if (finiteValues.length === 0) return {n: 0};
  return {
    n: finiteValues.length,
    avg: finiteValues.reduce((sum, value) => sum + value, 0) /
      finiteValues.length,
    p50: quantile(finiteValues, 0.5),
    p90: quantile(finiteValues, 0.9),
    min: Math.min(...finiteValues),
    max: Math.max(...finiteValues),
  };
}

function deduplicate(inputEvents) {
  const seen = new Set();
  return inputEvents.filter(event => {
    const key = [
      event.name,
      event.pid,
      event.tid,
      event.ts,
      event.dur,
      event.task,
    ].join("|");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function overlapDuration(left, right) {
  if (!left || !right) return 0;
  return Math.max(
    0,
    Math.min(left.ts + left.dur, right.ts + right.dur) -
      Math.max(left.ts, right.ts),
  );
}

function intervalOverlap(event, start, end) {
  if (!event || !Number.isFinite(start) || !Number.isFinite(end)) return 0;
  return Math.max(
    0,
    Math.min(event.ts + event.dur, end) - Math.max(event.ts, start),
  );
}

function selectFirst(inputEvents, predicate) {
  return inputEvents.find(predicate);
}

function selectLast(inputEvents, predicate) {
  const selected = inputEvents.filter(predicate);
  return selected.at(-1);
}

function duration(event) {
  return event?.dur;
}

function summarizeRecords(records) {
  const numericFields = [
    "layerLatency",
    "attention",
    "preGatingGap",
    "dispatch",
    "gmm1",
    "gmm2",
    "combine",
    "prefetchMla",
    "prefetchIndexer",
    "prefetchLookup",
    "prefetchGather",
    "mlaOverlapGap",
    "indexerOverlapGap",
    "indexerOverlapDispatch",
    "lookupOverlapGap",
    "lookupOverlapDispatch",
    "lookupOverlapGmm1",
    "lookupOverlapGmm2",
    "lookupOverlapCombine",
    "gatherOverlapGap",
    "gatherOverlapDispatch",
    "gatherOverlapGmm1",
    "gatherOverlapGmm2",
    "gatherOverlapCombine",
  ];
  return Object.fromEntries(numericFields.map(field => [
    field,
    stats(records.map(record => record[field])),
  ]));
}

stream.on("error", error => {
  throw error;
});

stream.on("end", () => {
  const uniqueEvents = deduplicate(events);
  const processScores = new Map();
  for (const event of uniqueEvents) {
    if (/mla_preprocess|HiSparseFlashAttention|AsuKvGather/.test(
      event.name,
    )) {
      processScores.set(
        event.pid,
        (processScores.get(event.pid) ?? 0) + 1,
      );
    }
  }
  const pid = [...processScores.entries()]
    .sort((left, right) => right[1] - left[1])[0]?.[0];
  if (pid === undefined) {
    throw new Error("no Ascend hardware process found in trace");
  }

  const hardwareEvents = uniqueEvents
    .filter(event => event.pid === pid)
    .sort((left, right) => left.ts - right.ts || left.tid - right.tid);
  const tids = [...new Set(hardwareEvents.map(event => event.tid))];
  const countOnStream = (pattern, tid) => hardwareEvents.filter(
    event => event.tid === tid && pattern.test(event.name),
  ).length;
  const streamWithMost = pattern => [...tids].sort(
    (left, right) =>
      countOnStream(pattern, right) - countOnStream(pattern, left),
  )[0];

  const mainStream = streamWithMost(/HiSparseFlashAttention/);
  const prefetchCandidate = streamWithMost(/LightningIndexerHiCached/);
  const prefetchStream =
    countOnStream(/LightningIndexerHiCached/, prefetchCandidate) > 0
      ? prefetchCandidate
      : null;
  const gatherStream = streamWithMost(/AsuKvGather/);

  const mainMlaEvents = hardwareEvents.filter(
    event => event.tid === mainStream && event.name === "mla_preprocess",
  );
  const taskCounts = new Map();
  for (const event of mainMlaEvents) {
    taskCounts.set(event.task, (taskCounts.get(event.task) ?? 0) + 1);
  }
  const layerTasks = [...taskCounts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, numLayers)
    .sort((left, right) => left[0] - right[0])
    .map(([task]) => task);
  if (layerTasks.length !== numLayers) {
    throw new Error(
      `expected ${numLayers} layer anchors, found ${layerTasks.length}`,
    );
  }

  const layerZeroAnchors = mainMlaEvents.filter(
    event => event.task === layerTasks[0],
  );
  const steps = [];
  for (let stepIndex = 0;
    stepIndex < layerZeroAnchors.length - 1;
    stepIndex++) {
    const start = layerZeroAnchors[stepIndex].ts;
    const end = layerZeroAnchors[stepIndex + 1].ts;
    const anchors = layerTasks.map(task => mainMlaEvents.find(
      event => event.task === task && event.ts >= start && event.ts < end,
    ));
    if (anchors.some(anchor => anchor === undefined)) continue;
    steps.push({start, end, anchors});
  }

  const allGathers = hardwareEvents.filter(
    event => event.tid === gatherStream && event.name === "AsuKvGather",
  );
  const prefetchLookups = prefetchStream === null
    ? []
    : hardwareEvents.filter(
      event => event.tid === prefetchStream &&
        /AsuHbmIndexLookup/.test(event.name),
    );
  const gatherByLookup = new Map();
  let gatherCursor = 0;
  for (const lookup of prefetchLookups) {
    const lookupEnd = lookup.ts + lookup.dur;
    while (gatherCursor < allGathers.length &&
      allGathers[gatherCursor].ts < lookupEnd) {
      gatherCursor++;
    }
    if (gatherCursor < allGathers.length) {
      gatherByLookup.set(lookup, allGathers[gatherCursor]);
      gatherCursor++;
    }
  }

  const records = [];
  for (let stepIndex = 0; stepIndex < steps.length; stepIndex++) {
    const step = steps[stepIndex];
    for (let layer = firstMoeLayer; layer < numLayers; layer++) {
      const startAnchor = step.anchors[layer - 1];
      const endAnchor = step.anchors[layer];
      const intervalEvents = hardwareEvents.filter(
        event => event.ts >= startAnchor.ts && event.ts < endAnchor.ts,
      );
      const mainEvents = intervalEvents.filter(
        event => event.tid === mainStream,
      );
      const attentionTail = selectLast(
        mainEvents,
        event => event.name === "DaAttentionMerge",
      );
      const gating = selectFirst(
        mainEvents,
        event => event.name === "MoeGatingTopK",
      );
      const dispatch = selectFirst(
        mainEvents,
        event => event.name === "MoeDistributeDispatchV2",
      );
      const groupedMatmuls = mainEvents.filter(
        event => /GroupedMatmul/.test(event.name),
      );
      const combine = selectFirst(
        mainEvents,
        event => event.name === "MoeDistributeCombineV2",
      );
      if (!attentionTail || !gating || !dispatch ||
        groupedMatmuls.length < 2 || !combine) {
        continue;
      }

      const prefetchEvents = prefetchStream === null
        ? []
        : intervalEvents.filter(event => event.tid === prefetchStream);
      const prefetchMla = selectFirst(
        prefetchEvents,
        event => /^mla_(?:query_)?preprocess$/.test(event.name),
      );
      const prefetchIndexer = selectFirst(
        prefetchEvents,
        event => event.name === "LightningIndexerHiCached",
      );
      const prefetchLookup = selectFirst(
        prefetchEvents,
        event => /AsuHbmIndexLookup/.test(event.name),
      );
      const prefetchGather = gatherByLookup.get(prefetchLookup);

      const attentionEnd = attentionTail.ts + attentionTail.dur;
      const gatingStart = gating.ts;
      const dispatchStart = dispatch.ts;
      const dispatchEnd = dispatch.ts + dispatch.dur;
      const gmm1 = groupedMatmuls[0];
      const gmm2 = groupedMatmuls[1];
      const combineEnd = combine.ts + combine.dur;
      records.push({
        step: stepIndex,
        fromEnd: steps.length - stepIndex,
        layer,
        layerLatency: endAnchor.ts - startAnchor.ts,
        attention: attentionEnd - startAnchor.ts,
        preGatingGap: gatingStart - attentionEnd,
        dispatch: dispatch.dur,
        gmm1: gmm1.dur,
        gmm2: gmm2.dur,
        combine: combine.dur,
        prefetchMla: duration(prefetchMla),
        prefetchIndexer: duration(prefetchIndexer),
        prefetchLookup: duration(prefetchLookup),
        prefetchGather: duration(prefetchGather),
        mlaOverlapGap: intervalOverlap(
          prefetchMla,
          attentionEnd,
          gatingStart,
        ),
        indexerOverlapGap: intervalOverlap(
          prefetchIndexer,
          attentionEnd,
          gatingStart,
        ),
        indexerOverlapDispatch: overlapDuration(
          prefetchIndexer,
          dispatch,
        ),
        lookupOverlapGap: intervalOverlap(
          prefetchLookup,
          attentionEnd,
          gatingStart,
        ),
        lookupOverlapDispatch: overlapDuration(prefetchLookup, dispatch),
        lookupOverlapGmm1: overlapDuration(prefetchLookup, gmm1),
        lookupOverlapGmm2: overlapDuration(prefetchLookup, gmm2),
        lookupOverlapCombine: overlapDuration(prefetchLookup, combine),
        gatherOverlapGap: intervalOverlap(
          prefetchGather,
          attentionEnd,
          gatingStart,
        ),
        gatherOverlapDispatch: overlapDuration(prefetchGather, dispatch),
        gatherOverlapGmm1: overlapDuration(prefetchGather, gmm1),
        gatherOverlapGmm2: overlapDuration(prefetchGather, gmm2),
        gatherOverlapCombine: overlapDuration(prefetchGather, combine),
        timeline: {
          layerStart: startAnchor.ts,
          attentionEnd,
          gatingStart,
          dispatchStart,
          dispatchEnd,
          gmm1Start: gmm1.ts,
          gmm1End: gmm1.ts + gmm1.dur,
          gmm2Start: gmm2.ts,
          gmm2End: gmm2.ts + gmm2.dur,
          combineStart: combine.ts,
          combineEnd,
          layerEnd: endAnchor.ts,
          prefetchMlaStart: prefetchMla?.ts,
          prefetchIndexerStart: prefetchIndexer?.ts,
          prefetchLookupStart: prefetchLookup?.ts,
          prefetchGatherStart: prefetchGather?.ts,
          prefetchGatherEnd: prefetchGather === undefined
            ? undefined
            : prefetchGather.ts + prefetchGather.dur,
        },
      });
    }
  }

  const perLayer = Object.fromEntries(
    Array.from(
      {length: numLayers - firstMoeLayer},
      (_, index) => firstMoeLayer + index,
    ).map(layer => [
      layer,
      summarizeRecords(records.filter(record => record.layer === layer)),
    ]),
  );

  const output = {
    file,
    pid,
    processName: processNames.get(pid),
    streams: {
      main: mainStream,
      prefetch: prefetchStream,
      gather: gatherStream,
    },
    layerTasks,
    stepCount: steps.length,
    recordCount: records.length,
    aggregate: summarizeRecords(records),
    perLayer,
    records,
  };
  console.log(JSON.stringify(output, null, pretty ? 2 : 0));
});
