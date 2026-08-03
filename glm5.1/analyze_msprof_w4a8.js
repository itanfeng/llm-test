#!/usr/bin/env node

// Stream a large msprof Chrome trace and summarize the operator signatures
// needed to distinguish static-W8 MLAPO, dynamic-W8 native MLA, W4A8 MoE,
// and hidden-state prefetch. The input trace is never loaded as one JSON
// object, so multi-gigabyte profiling files can be analyzed directly.
//
// Usage:
//   node analyze_msprof_w4a8.js TRACE.json [--num-layers=78]
//     [--first-k-dense=3] [--tail-steps=20] [--top=30] [--pretty]
//     [--output=REPORT.json]

"use strict";

const fs = require("fs");

const args = process.argv.slice(2);
const file = args.find(argument => !argument.startsWith("--"));
const pretty = args.includes("--pretty");
const readIntOption = (name, fallback) => {
  const option = args.find(argument => argument.startsWith(`${name}=`));
  return option === undefined
    ? fallback
    : Number.parseInt(option.split("=")[1], 10);
};
const topOption = args.find(argument => argument.startsWith("--top="));
const top = topOption === undefined
  ? 30
  : Number.parseInt(topOption.split("=")[1], 10);
const numLayers = readIntOption("--num-layers", 78);
const firstKDense = readIntOption("--first-k-dense", 3);
const tailSteps = readIntOption("--tail-steps", 20);
const outputOption = args.find(argument => argument.startsWith("--output="));
const outputFile = outputOption?.slice("--output=".length);

if (
  file === undefined ||
  !Number.isInteger(top) ||
  top < 1 ||
  !Number.isInteger(numLayers) ||
  numLayers < 2 ||
  !Number.isInteger(firstKDense) ||
  firstKDense < 0 ||
  firstKDense >= numLayers ||
  !Number.isInteger(tailSteps) ||
  tailSteps < 1
) {
  throw new Error(
    "usage: node analyze_msprof_w4a8.js TRACE.json [--num-layers=78] " +
    "[--first-k-dense=3] [--tail-steps=20] [--top=30] [--pretty] " +
    "[--output=REPORT.json]",
  );
}
if (!fs.existsSync(file)) {
  throw new Error(`trace file does not exist: ${file}`);
}

const keep = /mla_(?:query_)?preprocess|DynamicQuant|AscendQuant|QuantBatchMatmul|WeightQuantBatchMatmul|KvRmsNormRopeCache|InterleaveRope|BatchMatMul|MatMul|LayerNorm|RmsNorm|ScatterNdUpdate|LightningIndexer|AsuHbmIndexLookup|AsuHbmIndexMaintain|AsuKvGather|(?:Hi|Seg)SparseFlashAttention|DaAttentionMerge|batch_matmul_transpose|MoeGatingTopK|MoeDistributeDispatch|MoeDistributeCombine|GroupedMatmul|_swiglu_quant_kernel/;
const segmentedSfa = /^(?:HiSparse|SegSparse)FlashAttention$/;
const processNames = new Map();
const threadNames = new Map();
const statistics = new Map();
const seen = new Set();
const events = [];

function addEvent(event) {
  const key = [
    event.pid,
    event.tid,
    event.name,
    event.ts,
    event.dur,
    event.task,
  ].join("|");
  if (seen.has(key)) return;
  seen.add(key);
  events.push(event);

  const groupKey = `${event.pid}|${event.tid}|${event.name}`;
  let group = statistics.get(groupKey);
  if (group === undefined) {
    group = {
      pid: event.pid,
      tid: event.tid,
      name: event.name,
      count: 0,
      total: 0,
      min: Infinity,
      max: -Infinity,
      durations: [],
    };
    statistics.set(groupKey, group);
  }
  group.count++;
  group.total += event.dur;
  group.min = Math.min(group.min, event.dur);
  group.max = Math.max(group.max, event.dur);
  group.durations.push(event.dur);
}

function consume(object) {
  if (object === null || typeof object !== "object") return;
  if (object.ph === "M" && object.name === "process_name") {
    processNames.set(Number(object.pid), String(object.args?.name ?? ""));
    return;
  }
  if (object.ph === "M" && object.name === "thread_name") {
    threadNames.set(
      `${Number(object.pid)}|${Number(object.tid)}`,
      String(object.args?.name ?? ""),
    );
    return;
  }
  if (object.ph !== "X") return;

  const name = String(object.name ?? "");
  if (!keep.test(name)) return;
  const ts = Number(object.ts);
  const dur = Number(object.dur);
  if (!Number.isFinite(ts) || !Number.isFinite(dur) || dur <= 0) return;
  addEvent({
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
  for (const character of chunk) {
    if (depth > 0 || character === "{") objectBuffer += character;
    if (inString) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === "\"") inString = false;
      continue;
    }
    if (character === "\"") {
      inString = true;
    } else if (character === "{") {
      depth++;
    } else if (character === "}") {
      depth--;
      if (depth === 0) {
        try {
          consume(JSON.parse(objectBuffer));
        } catch (_) {}
        objectBuffer = "";
      }
    }
  }
});

function quantile(values, ratio) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(
    sorted.length - 1,
    Math.floor(sorted.length * ratio),
  )];
}

function summarize(group) {
  return {
    count: group.count,
    total: group.total,
    avg: group.total / group.count,
    p50: quantile(group.durations, 0.5),
    p90: quantile(group.durations, 0.9),
    min: group.min,
    max: group.max,
  };
}

function summarizeValues(values) {
  if (values.length === 0) return {count: 0};
  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    count: values.length,
    total,
    avg: total / values.length,
    p50: quantile(values, 0.5),
    p90: quantile(values, 0.9),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

stream.on("end", () => {
  const groups = [...statistics.values()];
  const pidScores = new Map();
  for (const group of groups) {
    if (
      /LightningIndexer|AsuHbmIndexLookup|AsuKvGather|(?:Hi|Seg)SparseFlashAttention/.test(
        group.name,
      )
    ) {
      pidScores.set(
        group.pid,
        (pidScores.get(group.pid) ?? 0) + group.count,
      );
    }
  }
  const hardwarePid = [...pidScores].sort(
    (left, right) => right[1] - left[1],
  )[0]?.[0];
  if (hardwarePid === undefined) {
    throw new Error("no HiSparse hardware operator was found in the trace");
  }

  const hardwareGroups = groups.filter(group => group.pid === hardwarePid);
  const tids = [...new Set(hardwareGroups.map(group => group.tid))];
  const count = (pattern, tid) => hardwareGroups
    .filter(group => (tid === undefined || group.tid === tid) &&
      pattern.test(group.name))
    .reduce((sum, group) => sum + group.count, 0);
  const pickStream = pattern => [...tids].sort(
    (left, right) => count(pattern, right) - count(pattern, left),
  )[0];

  const streamCountRows = tids.map(tid => ({
    tid,
    kvRmsNormRopeCache: count(/KvRmsNormRopeCache/, tid),
    lightningIndexer: count(/^LightningIndexer$/, tid),
    lightningIndexerHiCached: count(/^LightningIndexerHiCached$/, tid),
    asuLookup: count(/AsuHbmIndexLookup/, tid),
    asuGather: count(/^AsuKvGather$/, tid),
    hiSparseFlashAttention: count(segmentedSfa, tid),
  }));
  // A full-model ACL graph may partition the main calculation over several
  // physical stream IDs. Decode main streams have one native MLA anchor and
  // Indexer per layer plus two segmented SFA calls; prefill has only one SFA.
  const mainStreams = streamCountRows
    .filter(stream =>
      stream.kvRmsNormRopeCache > 0 &&
      stream.lightningIndexer > 0 &&
      stream.hiSparseFlashAttention >=
        1.8 * stream.kvRmsNormRopeCache
    )
    .map(stream => stream.tid);
  const mainStream = [...mainStreams].sort(
    (left, right) =>
      count(segmentedSfa, right) -
      count(segmentedSfa, left),
  )[0];
  const prefetchStreams = streamCountRows
    .filter(stream => stream.lightningIndexerHiCached > 0)
    .map(stream => stream.tid);
  const prefetchStream = [...prefetchStreams].sort(
    (left, right) =>
      count(/^LightningIndexerHiCached$/, right) -
      count(/^LightningIndexerHiCached$/, left),
  )[0] ?? null;
  const gatherCandidate = pickStream(/^AsuKvGather$/);
  const gatherStream = count(/^AsuKvGather$/, gatherCandidate) > 0
    ? gatherCandidate
    : null;
  const maintainCandidate = pickStream(/AsuHbmIndexMaintain/);
  const maintainStream = count(
    /AsuHbmIndexMaintain/,
    maintainCandidate,
  ) > 0
    ? maintainCandidate
    : null;

  const aggregateByName = new Map();
  for (const group of hardwareGroups) {
    let aggregate = aggregateByName.get(group.name);
    if (aggregate === undefined) {
      aggregate = {
        name: group.name,
        count: 0,
        total: 0,
        min: Infinity,
        max: -Infinity,
        durations: [],
      };
      aggregateByName.set(group.name, aggregate);
    }
    aggregate.count += group.count;
    aggregate.total += group.total;
    aggregate.min = Math.min(aggregate.min, group.min);
    aggregate.max = Math.max(aggregate.max, group.max);
    aggregate.durations.push(...group.durations);
  }

  const streamSummary = tid => hardwareGroups
    .filter(group => group.tid === tid)
    .sort((left, right) => right.total - left.total)
    .slice(0, top)
    .map(group => ({
      name: group.name,
      ...summarize(group),
    }));
  const aggregate = pattern => {
    const matched = [...aggregateByName.values()].filter(
      group => pattern.test(group.name),
    );
    if (matched.length === 0) return {count: 0};
    const merged = {
      count: 0,
      total: 0,
      min: Infinity,
      max: -Infinity,
      durations: [],
    };
    for (const group of matched) {
      merged.count += group.count;
      merged.total += group.total;
      merged.min = Math.min(merged.min, group.min);
      merged.max = Math.max(merged.max, group.max);
      merged.durations.push(...group.durations);
    }
    return summarize(merged);
  };
  const countMany = (pattern, targetTids) => targetTids.reduce(
    (total, tid) => total + count(pattern, tid),
    0,
  );
  const mainCount = pattern => countMany(pattern, mainStreams);

  const mlapoCount = mainCount(/^mla_preprocess$/);
  const nativeKvCount = mainCount(/KvRmsNormRopeCache/);
  const dynamicQuantCount = mainCount(/DynamicQuant/);
  const hiCachedCount = aggregate(/LightningIndexerHiCached/).count;
  const queryPreprocessCount = aggregate(/^mla_query_preprocess$/).count;
  const prefetchDynamicQuantCount = prefetchStreams.length === 0
    ? 0
    : countMany(/DynamicQuant/, prefetchStreams);
  const mainIndexerCount = mainCount(/^LightningIndexer$/);
  const mainLookupCount = mainCount(/AsuHbmIndexLookup/);
  const sfaCount = mainCount(segmentedSfa);
  const meanUpdateCount = aggregate(/ScatterNdUpdateMean/).count;

  let attentionPath = "unknown";
  if (mlapoCount > 0) attentionPath = "static_w8a8_mlapo";
  else if (nativeKvCount > 0 && dynamicQuantCount > 0) {
    attentionPath = "dynamic_w8a8_native";
  }

  const warnings = [];
  if (hiCachedCount === 0 && meanUpdateCount > 0) {
    warnings.push(
      "HiCached mean/status is updated, but no LightningIndexerHiCached " +
      "prefetch consumer is present.",
    );
  }
  if (mainIndexerCount > 0 && mainLookupCount !== mainIndexerCount) {
    warnings.push(
      `main Indexer/Lookup counts differ: ${mainIndexerCount}/` +
      `${mainLookupCount}.`,
    );
  }
  if (mainIndexerCount > 0 && sfaCount !== 2 * mainIndexerCount) {
    warnings.push(
      `segmented SFA count is ${sfaCount}; expected ` +
      `${2 * mainIndexerCount} for one hit and one miss call per layer.`,
    );
  }

  const hardwareEvents = events.filter(event => event.pid === hardwarePid);
  const mainStreamSet = new Set(mainStreams);
  const prefetchStreamSet = new Set(prefetchStreams);
  const mainAnchors = hardwareEvents
    .filter(event =>
      mainStreamSet.has(event.tid) &&
      /KvRmsNormRopeCache/.test(event.name)
    )
    .sort((left, right) => left.ts - right.ts);
  // Task Id is local to the compiled graph variant and is reused across
  // layers, so it cannot identify a GLM layer. The first three GLM layers
  // are dense and form a stable, visibly shorter three-interval signature.
  // Align the modulo-78 anchor sequence by maximizing the median separation
  // between those first three intervals and the remaining MoE intervals.
  let anchorAlignmentOffset = 0;
  let alignmentScore = -Infinity;
  let denseGapMedian = null;
  let moeGapMedian = null;
  for (let offset = 0; offset < numLayers; offset++) {
    const denseGapGroups = [];
    const moeGapGroups = [];
    for (
      let start = offset;
      start + numLayers < mainAnchors.length;
      start += numLayers
    ) {
      const groupDenseGaps = [];
      const groupMoeGaps = [];
      for (let layer = 0; layer < numLayers - 1; layer++) {
        const gap = mainAnchors[start + layer + 1].ts -
          mainAnchors[start + layer].ts;
        if (layer < firstKDense) groupDenseGaps.push(gap);
        else groupMoeGaps.push(gap);
      }
      denseGapGroups.push(groupDenseGaps);
      moeGapGroups.push(groupMoeGaps);
    }
    const denseGaps = denseGapGroups.slice(-tailSteps).flat();
    const moeGaps = moeGapGroups.slice(-tailSteps).flat();
    if (denseGaps.length < firstKDense * 2 || moeGaps.length < 2) continue;
    const candidateDenseMedian = quantile(denseGaps, 0.5);
    const candidateMoeMedian = quantile(moeGaps, 0.5);
    const candidateScore = candidateMoeMedian - candidateDenseMedian;
    if (candidateScore > alignmentScore) {
      anchorAlignmentOffset = offset;
      alignmentScore = candidateScore;
      denseGapMedian = candidateDenseMedian;
      moeGapMedian = candidateMoeMedian;
    }
  }
  const steps = [];
  for (
    let startIndex = anchorAlignmentOffset;
    startIndex + numLayers < mainAnchors.length;
    startIndex += numLayers
  ) {
    const anchors = mainAnchors.slice(startIndex, startIndex + numLayers);
    const start = anchors[0].ts;
    const end = mainAnchors[startIndex + numLayers].ts;
    const layers = [];
    for (let layer = 0; layer < anchors.length - 1; layer++) {
      layers.push({
        layer: layer + 1,
        start: anchors[layer].ts,
        end: anchors[layer + 1].ts,
        latency: anchors[layer + 1].ts - anchors[layer].ts,
      });
    }
    steps.push({
      start,
      end,
      duration: end - start,
      layers,
    });
  }
  const boundaryGapMedian = quantile(
    steps.map(step => step.duration -
      step.layers.reduce((total, layer) => total + layer.latency, 0)),
    0.5,
  );
  const stableSteps = steps.slice(-tailSteps);
  const stableLayerRecords = stableSteps.flatMap((step, index) =>
    step.layers.map(layer => ({
      fromEnd: stableSteps.length - index,
      layer: layer.layer,
      latency: layer.latency,
    }))
  );
  const stableMoeLayerRecords = stableLayerRecords.filter(
    record => record.layer > firstKDense,
  );
  const stableStepHistory = stableSteps.map((step, index) => {
    const anchoredLayerDuration = step.layers.reduce(
      (total, layer) => total + layer.latency,
      0,
    );
    const moeLayers = step.layers.filter(layer => layer.layer > firstKDense);
    return {
      fromEnd: stableSteps.length - index,
      duration: step.duration,
      anchoredLayerDuration,
      stepRemainder: step.duration - anchoredLayerDuration,
      moeLayerAvg: moeLayers.length === 0
        ? null
        : moeLayers.reduce(
          (total, layer) => total + layer.latency,
          0,
        ) / moeLayers.length,
    };
  });
  const perLayer = [];
  for (let layer = 1; layer < numLayers; layer++) {
    const values = stableLayerRecords
      .filter(record => record.layer === layer)
      .map(record => record.latency);
    perLayer.push({layer, ...summarizeValues(values)});
  }

  const stableWindowStart = stableSteps[0]?.start ?? Infinity;
  const stableWindowEnd = stableSteps.at(-1)?.end ?? -Infinity;
  const stableWindowEvents = hardwareEvents.filter(
    event => event.ts >= stableWindowStart && event.ts < stableWindowEnd,
  );
  const stableOperator = (pattern, targetTids) => {
    const targetSet = targetTids === undefined
      ? null
      : new Set(Array.isArray(targetTids) ? targetTids : [targetTids]);
    return summarizeValues(
      stableWindowEvents
        .filter(event =>
          pattern.test(event.name) &&
          (targetSet === null || targetSet.has(event.tid))
        )
        .map(event => event.dur),
    );
  };
  const gatherEvents = stableWindowEvents
    .filter(event =>
      event.tid === gatherStream && event.name === "AsuKvGather"
    )
    .sort((left, right) => left.ts - right.ts);
  const prefetchLookups = prefetchStreams.length === 0
    ? []
    : stableWindowEvents
      .filter(event =>
        prefetchStreamSet.has(event.tid) &&
        /AsuHbmIndexLookup/.test(event.name)
      )
      .sort((left, right) => left.ts - right.ts);
  const prefetchGatherSet = new Set();
  let gatherCursor = 0;
  for (const lookup of prefetchLookups) {
    const lookupEnd = lookup.ts + lookup.dur;
    while (
      gatherCursor < gatherEvents.length &&
      gatherEvents[gatherCursor].ts < lookupEnd
    ) {
      gatherCursor++;
    }
    if (gatherCursor < gatherEvents.length) {
      prefetchGatherSet.add(gatherEvents[gatherCursor]);
      gatherCursor++;
    }
  }
  const prefetchGathers = gatherEvents.filter(event =>
    prefetchGatherSet.has(event)
  );
  const mainGathers = gatherEvents.filter(event =>
    !prefetchGatherSet.has(event)
  );

  const roles = new Map([
    [gatherStream, "gather"],
    [maintainStream, "maintain"],
  ]);
  for (const tid of mainStreams) roles.set(tid, "main");
  for (const tid of prefetchStreams) roles.set(tid, "prefetch_compute");
  const operatorCountsByTid = streamCountRows.map(stream => ({
    ...stream,
    threadName: threadNames.get(`${hardwarePid}|${stream.tid}`) ?? "",
  })).filter(stream =>
    stream.kvRmsNormRopeCache > 0 ||
    stream.lightningIndexer > 0 ||
    stream.lightningIndexerHiCached > 0 ||
    stream.asuLookup > 0 ||
    stream.asuGather > 0 ||
    stream.hiSparseFlashAttention > 0
  );

  const output = {
    file,
    hardwarePid,
    processName: processNames.get(hardwarePid) ?? "",
    diagnosis: {
      attentionPath,
      hiddenStatePrefetchActive: hiCachedCount > 0,
      prefetchQueryPreprocess: queryPreprocessCount > 0
        ? "static_w8a8_mla_query_preprocess"
        : hiCachedCount > 0 && prefetchDynamicQuantCount > 0
          ? "dynamic_w8a8_native"
          : "none",
      mainFlowRatio: {
        lightningIndexer: mainIndexerCount,
        asuLookup: mainLookupCount,
        hiSparseFlashAttention: sfaCount,
      },
      warnings,
    },
    streams: {
      main: mainStream,
      mainAll: mainStreams,
      prefetchCompute: prefetchStream,
      prefetchComputeAll: prefetchStreams,
      gather: gatherStream,
      maintain: maintainStream,
    },
    operatorCountsByTid,
    decodeTiming: {
      anchor: "KvRmsNormRopeCache",
      numLayers,
      firstKDense,
      mainAnchorCount: mainAnchors.length,
      anchorAlignmentOffset,
      alignmentScore,
      denseGapMedian,
      moeGapMedian,
      boundaryGapMedian,
      stepCount: steps.length,
      stableStepCount: stableSteps.length,
      stableStep: summarizeValues(
        stableStepHistory.map(step => step.duration),
      ),
      stableAnchoredLayersPerStep: summarizeValues(
        stableStepHistory.map(step => step.anchoredLayerDuration),
      ),
      stableStepRemainder: summarizeValues(
        stableStepHistory.map(step => step.stepRemainder),
      ),
      stableAllClosedLayer: summarizeValues(
        stableLayerRecords.map(record => record.latency),
      ),
      stableMoeLayer: summarizeValues(
        stableMoeLayerRecords.map(record => record.latency),
      ),
      stableStepHistory,
      stableLayerRecords,
      perLayer,
    },
    stableKeyOperators: {
      mainIndexer: stableOperator(/^LightningIndexer$/, mainStreams),
      mainLookup: stableOperator(/AsuHbmIndexLookup/, mainStreams),
      prefetchIndexer: prefetchStreams.length === 0
        ? {count: 0}
        : stableOperator(
          /^LightningIndexerHiCached$/,
          prefetchStreams,
        ),
      prefetchLookup: prefetchStreams.length === 0
        ? {count: 0}
        : stableOperator(/AsuHbmIndexLookup/, prefetchStreams),
      mainGather: summarizeValues(mainGathers.map(event => event.dur)),
      prefetchGather: summarizeValues(
        prefetchGathers.map(event => event.dur),
      ),
      allGather: summarizeValues(gatherEvents.map(event => event.dur)),
      hitAndMissSfa: stableOperator(
        segmentedSfa,
        mainStreams,
      ),
      dynamicQuantOnPrefetch: prefetchStreams.length === 0
        ? {count: 0}
        : stableOperator(/DynamicQuant/, prefetchStreams),
      quantBatchMatmulOnPrefetch: prefetchStreams.length === 0
        ? {count: 0}
        : stableOperator(/QuantBatchMatmul/, prefetchStreams),
    },
    keyOperators: Object.fromEntries([
      ["mla_preprocess", /^mla_preprocess$/],
      ["mla_query_preprocess", /^mla_query_preprocess$/],
      ["DynamicQuant", /DynamicQuant/],
      ["AscendQuant", /AscendQuant/],
      ["QuantBatchMatmul", /QuantBatchMatmul/],
      ["KvRmsNormRopeCache", /KvRmsNormRopeCache/],
      ["LightningIndexer", /^LightningIndexer$/],
      ["LightningIndexerHiCached", /LightningIndexerHiCached/],
      ["AsuHbmIndexLookup", /AsuHbmIndexLookup/],
      ["AsuKvGather", /^AsuKvGather$/],
      ["AsuHbmIndexMaintain", /AsuHbmIndexMaintain/],
      ["HiSparseFlashAttention", segmentedSfa],
      ["DaAttentionMerge", /^DaAttentionMerge$/],
      ["ScatterNdUpdate", /ScatterNdUpdate(?!Mean)/],
      ["ScatterNdUpdateMean", /ScatterNdUpdateMean/],
      ["GroupedMatmulSwigluQuant", /GroupedMatmulSwigluQuant/],
      ["GroupedMatmul", /GroupedMatmul(?!SwigluQuant)/],
      ["_swiglu_quant_kernel", /^_swiglu_quant_kernel$/],
    ].map(([name, pattern]) => [name, aggregate(pattern)])),
    streamSummaries: tids
      .filter(tid => roles.has(tid))
      .map(tid => ({
        tid,
        role: roles.get(tid),
        threadName: threadNames.get(`${hardwarePid}|${tid}`) ?? "",
        operators: streamSummary(tid),
      })),
  };

  const serialized = JSON.stringify(output, null, pretty ? 2 : 0);
  if (outputFile === undefined) console.log(serialized);
  else fs.writeFileSync(outputFile, `${serialized}\n`);
});
