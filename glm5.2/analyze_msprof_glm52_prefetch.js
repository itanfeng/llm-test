#!/usr/bin/env node

// Compare GLM-5.2 Host-Memory DSA offload and grouped hidden-state prefetch
// from large msprof Chrome traces without loading the whole JSON into memory.
//
// The main asynchronous AsuKvGather stream is used as the decode anchor.  A
// GLM-5.2 decode step has exactly 78 exact gathers.  The original prefetch
// layout puts 19 x 4 speculative gathers on the Prefetch Compute stream.  The
// first pipelined layout disables the 1 -> 2 bootstrap and interleaves 18 x 4
// speculative gathers with exact gathers on the same stream.  The 2+2 layout
// keeps 18 x 4 gathers on a dedicated Prefetch Gather stream.  A concentrated
// no-bootstrap layout has the same 72-Gather count as 2+2, so select it
// explicitly with --prefetch-gather-schedule.
//
// Usage:
//   node analyze_msprof_glm52_prefetch.js BASE.json PREFETCH.json \
//     [--tail-steps=20] [--prefetch-gather-schedule=auto] \
//     [--pretty] [--output=REPORT.json]

"use strict";

const fs = require("fs");

const argv = process.argv.slice(2);
const inputFiles = argv.filter(argument => !argument.startsWith("--"));
const baseFile = inputFiles[0];
const prefetchFile = inputFiles[1];
const pretty = argv.includes("--pretty");
const readIntOption = (name, fallback) => {
  const option = argv.find(argument => argument.startsWith(`${name}=`));
  return option === undefined
    ? fallback
    : Number.parseInt(option.slice(name.length + 1), 10);
};
const numLayers = readIntOption("--num-layers", 78);
const groupSize = readIntOption("--group-size", 4);
const prefetchGatherBatchSize = groupSize / 2;
const firstGroupedLayer = readIntOption("--first-grouped-layer", 2);
const tailSteps = readIntOption("--tail-steps", 20);
const scheduleOption = argv.find(argument =>
  argument.startsWith("--prefetch-gather-schedule=")
);
const forcedPrefetchGatherSchedule = scheduleOption === undefined
  ? "auto"
  : scheduleOption.slice("--prefetch-gather-schedule=".length);
const validPrefetchGatherSchedules = new Set([
  "auto",
  "main_stream_1x4",
  "prefetch_stream_2x2",
  "prefetch_stream_4",
  "prefetch_stream_4_no_bootstrap",
]);
const outputOption = argv.find(argument => argument.startsWith("--output="));
const outputFile = outputOption?.slice("--output=".length);

if (
  inputFiles.length !== 2 ||
  !Number.isInteger(numLayers) ||
  numLayers < 2 ||
  !Number.isInteger(groupSize) ||
  groupSize < 1 ||
  !Number.isInteger(prefetchGatherBatchSize) ||
  !Number.isInteger(firstGroupedLayer) ||
  firstGroupedLayer < 0 ||
  firstGroupedLayer >= numLayers ||
  !Number.isInteger(tailSteps) ||
  tailSteps < 1 ||
  !validPrefetchGatherSchedules.has(forcedPrefetchGatherSchedule)
) {
  throw new Error(
    "usage: node analyze_msprof_glm52_prefetch.js BASE.json PREFETCH.json " +
    "[--tail-steps=20] [--prefetch-gather-schedule=auto] " +
    "[--pretty] [--output=REPORT.json]",
  );
}
for (const file of inputFiles) {
  if (!fs.existsSync(file)) throw new Error(`trace does not exist: ${file}`);
}

const KEEP = /PrefetchQliFusion|KvRmsNormRopeCache|InterleaveRope|_triton_rope_siso|DynamicQuant|QuantBatchMatmul|MatMul|RmsNorm|aclnnMul_|aclnnAdd_|aclnnIndex|Muls|Adds|Subs|Clamp|Contiguous|Slice|Arange|FloorDiv|GatherV2|ScatterNdUpdate|LightningIndexer|AsuHbmIndexLookup|AsuHbmIndexMaintain|AsuKvGather|SparseFlashAttention|batch_matmul_transpose|MoeGatingTopK|MoeDistributeDispatch|MoeDistributeCombine|GroupedMatmul|EVENT_(?:WAIT|RECORD)/;
const MAIN_GATHER = /^AsuKvGather$/;
const PREFETCH_INDEXER = /LightningIndexerHiCached/;
const PREFETCH_QLI = /^PrefetchQliFusion$/;
const LOOKUP = /AsuHbmIndexLookup/;
const MAINTAIN = /AsuHbmIndexMaintain/;

function quantile(values, ratio) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(
    sorted.length - 1,
    Math.floor(sorted.length * ratio),
  )];
}

function summarize(values) {
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

function trimmedAverage(values, ratio = 0.1) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const trimCount = Math.floor(sorted.length * ratio);
  const retained = sorted.slice(trimCount, sorted.length - trimCount);
  return retained.reduce((total, value) => total + value, 0) /
    retained.length;
}

function patternCount(events, pattern, tid = null) {
  return events.filter(event =>
    (tid === null || event.tid === tid) && pattern.test(event.name)
  ).length;
}

function intervalsOverlap(left, right) {
  return Math.max(
    0,
    Math.min(left.ts + left.dur, right.ts + right.dur) -
      Math.max(left.ts, right.ts),
  );
}

function overlapWithSortedIntervals(event, intervals) {
  let total = 0;
  const end = event.ts + event.dur;
  for (const interval of intervals) {
    if (interval.ts >= end) break;
    if (interval.ts + interval.dur <= event.ts) continue;
    total += intervalsOverlap(event, interval);
  }
  return total;
}

async function parseTrace(file) {
  const processNames = new Map();
  const threadNames = new Map();
  const events = [];
  const seen = new Set();

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
    if (!KEEP.test(name)) return;
    const ts = Number(object.ts);
    const dur = Number(object.dur);
    if (!Number.isFinite(ts) || !Number.isFinite(dur) || dur <= 0) return;
    const event = {
      name,
      pid: Number(object.pid),
      tid: Number(object.tid),
      ts,
      dur,
      task: Number(
        object.args?.["Task Id"] ?? object.args?.["Task ID"] ?? -1,
      ),
    };
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
  }

  let depth = 0;
  let inString = false;
  let escaped = false;
  let objectBuffer = "";
  const stream = fs.createReadStream(file, {
    encoding: "utf8",
    highWaterMark: 8 * 1024 * 1024,
  });
  for await (const chunk of stream) {
    for (const character of chunk) {
      if (depth > 0 || character === "{") objectBuffer += character;
      if (inString) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === "\"") inString = false;
        continue;
      }
      if (character === "\"") inString = true;
      else if (character === "{") depth++;
      else if (character === "}") {
        depth--;
        if (depth === 0) {
          try {
            consume(JSON.parse(objectBuffer));
          } catch (_) {}
          objectBuffer = "";
        }
      }
    }
  }
  return {file, processNames, threadNames, events};
}

function pickHardwarePid(parsed) {
  const scores = new Map();
  for (const event of parsed.events) {
    if (!MAIN_GATHER.test(event.name) &&
        !/SparseFlashAttention/.test(event.name)) continue;
    scores.set(event.pid, (scores.get(event.pid) ?? 0) + 1);
  }
  const result = [...scores].sort(
    (left, right) => right[1] - left[1],
  )[0]?.[0];
  if (result === undefined) {
    throw new Error(`no DSA hardware timeline found in ${parsed.file}`);
  }
  return result;
}

function repeatingTaskPeriod(gatherEvents, candidatePeriods) {
  const taskCounts = new Map();
  for (const event of gatherEvents) {
    taskCounts.set(event.task, (taskCounts.get(event.task) ?? 0) + 1);
  }
  const maximumTaskCount = Math.max(...taskCounts.values());
  const startTask = Math.min(
    ...[...taskCounts]
      .filter(([, count]) => count === maximumTaskCount)
      .map(([task]) => task),
  );
  const startIndices = [];
  for (let index = 0; index < gatherEvents.length; index++) {
    if (gatherEvents[index].task === startTask) startIndices.push(index);
  }
  const periodScores = candidatePeriods.map(period => ({
    period,
    count: startIndices.filter((startIndex, index) =>
      index + 1 < startIndices.length &&
      startIndices[index + 1] - startIndex === period
    ).length,
  })).sort((left, right) => right.count - left.count);
  const period = periodScores[0]?.count > 0 ? periodScores[0].period : null;
  return {startTask, taskCounts, startIndices, period, periodScores};
}

function splitGatherLayout(gatherEvents, prefetchActive) {
  const predictionSources = [];
  for (
    let source = firstGroupedLayer;
    source + groupSize < numLayers;
    source += groupSize
  ) {
    predictionSources.push(source);
  }
  const staggeredPrefetchCount = predictionSources.length * groupSize;
  const candidatePeriods = prefetchActive
    ? [numLayers, numLayers + staggeredPrefetchCount]
    : [numLayers];
  const repeating = repeatingTaskPeriod(gatherEvents, candidatePeriods);
  if (repeating.period === null) {
    throw new Error("no repeating AsuKvGather task-id period found");
  }

  const exact = [];
  const prefetch = [];
  const periods = [];
  let lastCompletePeriodEnd = null;
  for (let index = 0; index + 1 < repeating.startIndices.length; index++) {
    const startIndex = repeating.startIndices[index];
    const nextStartIndex = repeating.startIndices[index + 1];
    if (nextStartIndex - startIndex !== repeating.period) continue;
    lastCompletePeriodEnd = nextStartIndex;
    const periodEvents = gatherEvents.slice(startIndex, nextStartIndex);
    if (repeating.period === numLayers) {
      exact.push(...periodEvents);
      periods.push({exact: periodEvents, prefetch: []});
      continue;
    }

    const exactPeriod = [];
    const prefetchPeriod = [];
    let cursor = 0;
    for (let layer = 0; layer < numLayers; layer++) {
      exactPeriod.push(periodEvents[cursor++]);
      if (
        layer >= firstGroupedLayer &&
        layer < firstGroupedLayer + staggeredPrefetchCount
      ) {
        prefetchPeriod.push(periodEvents[cursor++]);
      }
    }
    if (cursor !== periodEvents.length) {
      throw new Error(
        `invalid interleaved Gather layout: consumed ${cursor}, ` +
        `observed ${periodEvents.length}`,
      );
    }
    exact.push(...exactPeriod);
    prefetch.push(...prefetchPeriod);
    periods.push({exact: exactPeriod, prefetch: prefetchPeriod});
  }
  // Preserve the first exact Gather of the following period.  It is the end
  // timestamp of the last complete decode step; it is excluded again by the
  // stable half-open time range and therefore does not affect operator counts.
  if (lastCompletePeriodEnd !== null) {
    exact.push(gatherEvents[lastCompletePeriodEnd]);
  }
  return {
    kind: repeating.period === numLayers ? "separate" : "interleaved",
    repeating,
    predictionSources,
    staggeredPrefetchCount,
    exact,
    prefetch,
    periods,
  };
}

function buildDecodeSteps(mainGatherEvents) {
  const repeating = repeatingTaskPeriod(mainGatherEvents, [numLayers]);
  const {startTask, taskCounts, startIndices} = repeating;
  const steps = [];
  for (let index = 0; index + 1 < startIndices.length; index++) {
    const startIndex = startIndices[index];
    const nextStartIndex = startIndices[index + 1];
    if (nextStartIndex - startIndex !== numLayers) continue;
    const anchors = mainGatherEvents.slice(startIndex, nextStartIndex);
    const start = anchors[0].ts;
    const end = mainGatherEvents[nextStartIndex].ts;
    const layers = anchors.map((anchor, layer) => ({
      layer,
      start: anchor.ts,
      end: layer + 1 < anchors.length ? anchors[layer + 1].ts : end,
      latency: (layer + 1 < anchors.length ? anchors[layer + 1].ts : end) -
        anchor.ts,
    }));
    steps.push({start, end, duration: end - start, anchors, layers});
  }
  return {startTask, taskCounts, steps};
}

function operatorSummary(stableEvents, pattern, tids = null) {
  const tidSet = tids === null
    ? null
    : new Set(Array.isArray(tids) ? tids : [tids]);
  return summarize(stableEvents
    .filter(event =>
      pattern.test(event.name) &&
      (tidSet === null || tidSet.has(event.tid))
    )
    .map(event => event.dur));
}

function summarizeByName(events) {
  const durations = new Map();
  for (const event of events) {
    const values = durations.get(event.name) ?? [];
    values.push(event.dur);
    durations.set(event.name, values);
  }
  return Object.fromEntries(
    [...durations.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, values]) => [name, summarize(values)]),
  );
}

function summarizeComputeChainsToIndexer(events) {
  const ordered = [...events].sort((left, right) => left.ts - right.ts);
  const spans = [];
  const operatorSums = [];
  const projectionReadySpans = [];
  const projectionReadyOperatorSums = [];
  for (let index = 0; index < ordered.length; index++) {
    if (!PREFETCH_INDEXER.test(ordered[index].name)) continue;
    let start = index;
    while (start > 0) {
      const previous = ordered[start - 1];
      if (LOOKUP.test(previous.name) || PREFETCH_INDEXER.test(previous.name)) {
        break;
      }
      start--;
    }
    const chain = ordered.slice(start, index + 1);
    if (chain.length === 0) continue;
    spans.push(
      ordered[index].ts + ordered[index].dur - chain[0].ts,
    );
    operatorSums.push(chain.reduce((total, event) => total + event.dur, 0));
    const projectionReadyIndex = chain.findLastIndex(event =>
      PREFETCH_QLI.test(event.name) || /_triton_rope_siso/.test(event.name)
    );
    if (projectionReadyIndex >= 0) {
      const projectionReadyChain = chain.slice(0, projectionReadyIndex + 1);
      const readyEvent = projectionReadyChain.at(-1);
      projectionReadySpans.push(
        readyEvent.ts + readyEvent.dur - projectionReadyChain[0].ts,
      );
      projectionReadyOperatorSums.push(projectionReadyChain.reduce(
        (total, event) => total + event.dur,
        0,
      ));
    }
  }
  return {
    span: summarize(spans),
    operatorSum: summarize(operatorSums),
    projectionReadySpan: summarize(projectionReadySpans),
    projectionReadyOperatorSum: summarize(projectionReadyOperatorSums),
  };
}

function overlapBreakdown(events, competingIntervals) {
  const overlapping = [];
  const clean = [];
  const overlapDurations = [];
  for (const event of events) {
    const overlap = overlapWithSortedIntervals(event, competingIntervals);
    if (overlap > 0) {
      overlapping.push(event.dur);
      overlapDurations.push(overlap);
    } else {
      clean.push(event.dur);
    }
  }
  return {
    all: summarize(events.map(event => event.dur)),
    overlapping: summarize(overlapping),
    clean: summarize(clean),
    overlapDuration: summarize(overlapDurations),
  };
}

function analyzeParsed(parsed, label) {
  const hardwarePid = pickHardwarePid(parsed);
  const hardwareEvents = parsed.events
    .filter(event => event.pid === hardwarePid)
    .sort((left, right) => left.ts - right.ts);
  const tids = [...new Set(hardwareEvents.map(event => event.tid))];
  const gatherCounts = tids.map(tid => ({
    tid,
    count: patternCount(hardwareEvents, MAIN_GATHER, tid),
  })).filter(row => row.count > 0)
    .sort((left, right) => right.count - left.count);
  const mainGatherStream = gatherCounts[0]?.tid;
  if (mainGatherStream === undefined) {
    throw new Error(`no AsuKvGather stream found in ${parsed.file}`);
  }

  const rawMainGatherEvents = hardwareEvents.filter(event =>
    event.tid === mainGatherStream && MAIN_GATHER.test(event.name)
  );
  const hiCachedCountAll = patternCount(hardwareEvents, PREFETCH_INDEXER);
  const gatherLayout = splitGatherLayout(
    rawMainGatherEvents,
    hiCachedCountAll > 0,
  );
  const mainGatherEvents = gatherLayout.exact;
  const decoded = buildDecodeSteps(mainGatherEvents);
  const stableSteps = decoded.steps.slice(-tailSteps);
  if (stableSteps.length === 0) {
    throw new Error(`no complete 78-layer decode step found in ${parsed.file}`);
  }
  const stableStart = stableSteps[0].start;
  const stableEnd = stableSteps.at(-1).end;
  const stableEvents = hardwareEvents.filter(event =>
    event.ts >= stableStart && event.ts < stableEnd
  );

  const hiCachedCount = patternCount(stableEvents, PREFETCH_INDEXER);
  const legacyPrefetchGathers = (
    1 + gatherLayout.predictionSources.length
  ) * groupSize;
  const candidatePrefetchGathers = gatherLayout.kind === "interleaved"
    ? [gatherLayout.staggeredPrefetchCount]
    : [legacyPrefetchGathers, gatherLayout.staggeredPrefetchCount];
  let prefetchComputeStream = null;
  if (hiCachedCount > 0) {
    const candidates = tids.filter(tid => tid !== mainGatherStream)
      .map(tid => ({
      tid,
      hiCached: patternCount(stableEvents, PREFETCH_INDEXER, tid),
      lookup: patternCount(stableEvents, LOOKUP, tid),
    }));
    prefetchComputeStream = candidates.sort((left, right) =>
      right.hiCached - left.hiCached ||
      right.lookup - left.lookup
    )[0]?.tid ?? null;
  }
  let prefetchGatherStream = null;
  if (gatherLayout.kind === "interleaved") {
    prefetchGatherStream = mainGatherStream;
  } else if (prefetchComputeStream !== null) {
    const expectedTotals = candidatePrefetchGathers.map(
      count => count * stableSteps.length,
    );
    const candidates = tids.filter(tid => tid !== mainGatherStream)
      .map(tid => {
        const count = patternCount(stableEvents, MAIN_GATHER, tid);
        return {
          tid,
          count,
          distance: Math.min(...expectedTotals.map(expectedTotal =>
            Math.abs(count - expectedTotal)
          )),
        };
      })
      .filter(candidate => candidate.count > 0)
      .sort((left, right) =>
        left.distance - right.distance || right.count - left.count
      );
    prefetchGatherStream = candidates[0]?.tid ?? null;
  }
  const maintainStream = tids.map(tid => ({
    tid,
    count: patternCount(hardwareEvents, MAINTAIN, tid),
  })).sort((left, right) => right.count - left.count)[0];
  const mainComputeStreams = tids.filter(tid =>
    patternCount(hardwareEvents, /SparseFlashAttention/, tid) > 0 ||
    patternCount(hardwareEvents, /KvRmsNormRopeCache/, tid) > 0
  );
  const prefetchGatherEvents = gatherLayout.kind === "interleaved"
    ? gatherLayout.prefetch.filter(event =>
      event.ts >= stableStart && event.ts < stableEnd
    )
    : prefetchGatherStream === null
      ? []
      : stableEvents.filter(event =>
        event.tid === prefetchGatherStream && MAIN_GATHER.test(event.name)
      );
  const observedPrefetchGathersPerStep = prefetchGatherEvents.length /
    stableSteps.length;
  const detectedPrefetchGatherSchedule = gatherLayout.kind === "interleaved"
    ? "main_stream_1x4"
    : prefetchGatherStream === null
      ? "disabled"
      : Math.abs(
        observedPrefetchGathersPerStep - gatherLayout.staggeredPrefetchCount,
      ) < 0.5
        ? "prefetch_stream_2x2"
        : "prefetch_stream_4";
  const prefetchGatherSchedule = label === "prefetch" &&
    forcedPrefetchGatherSchedule !== "auto"
    ? forcedPrefetchGatherSchedule
    : detectedPrefetchGatherSchedule;
  if (
    (prefetchGatherSchedule === "main_stream_1x4") !==
    (gatherLayout.kind === "interleaved")
  ) {
    throw new Error(
      `prefetch Gather schedule ${prefetchGatherSchedule} conflicts with ` +
      `${gatherLayout.kind} stream layout in ${parsed.file}`,
    );
  }
  const expectedPrefetchGathers = prefetchGatherSchedule === "disabled"
    ? 0
    : prefetchGatherSchedule === "prefetch_stream_4"
      ? legacyPrefetchGathers
      : gatherLayout.staggeredPrefetchCount;
  if (
    label === "prefetch" &&
    Math.abs(observedPrefetchGathersPerStep - expectedPrefetchGathers) >= 0.5
  ) {
    throw new Error(
      `prefetch Gather schedule ${prefetchGatherSchedule} expects ` +
      `${expectedPrefetchGathers} calls/step, observed ` +
      `${observedPrefetchGathersPerStep}`,
    );
  }

  const stableLayerRecords = stableSteps.flatMap((step, stepIndex) =>
    step.layers.map(layer => ({
      fromEnd: stableSteps.length - stepIndex,
      layer: layer.layer,
      latency: layer.latency,
    }))
  );
  let prefetchReleaseLayers;
  if (prefetchGatherSchedule === "main_stream_1x4") {
    prefetchReleaseLayers = new Set(Array.from(
      {length: gatherLayout.staggeredPrefetchCount},
      (_, index) => firstGroupedLayer + index,
    ));
  } else if (prefetchGatherSchedule === "prefetch_stream_2x2") {
    prefetchReleaseLayers = new Set(gatherLayout.predictionSources.flatMap(
      layer => [layer, layer + prefetchGatherBatchSize],
    ));
  } else if (prefetchGatherSchedule === "prefetch_stream_4_no_bootstrap") {
    prefetchReleaseLayers = new Set(
      gatherLayout.predictionSources.map(layer => layer + 1),
    );
  } else {
    prefetchReleaseLayers = new Set([
      firstGroupedLayer,
      ...gatherLayout.predictionSources.map(layer => layer + 1),
    ]);
  }
  const releaseLayerRecords = stableLayerRecords.filter(record =>
    prefetchReleaseLayers.has(record.layer)
  );
  const nonReleaseLayerRecords = stableLayerRecords.filter(record =>
    record.layer < numLayers - 1 &&
    !prefetchReleaseLayers.has(record.layer)
  );
  const perLayer = [];
  for (let layer = 0; layer < numLayers; layer++) {
    perLayer.push({
      layer,
      ...summarize(stableLayerRecords
        .filter(record => record.layer === layer)
        .map(record => record.latency)),
    });
  }

  const groupRecords = [];
  for (let stepIndex = 0; stepIndex < stableSteps.length; stepIndex++) {
    const step = stableSteps[stepIndex];
    for (
      let leader = firstGroupedLayer;
      leader < numLayers;
      leader += groupSize
    ) {
      const groupEnd = Math.min(leader + groupSize, numLayers);
      const start = step.anchors[leader].ts;
      const end = groupEnd < numLayers
        ? step.anchors[groupEnd].ts
        : step.end;
      groupRecords.push({
        fromEnd: stableSteps.length - stepIndex,
        leader,
        latency: end - start,
        includesStepBoundary: groupEnd === numLayers,
      });
    }
  }
  const perGroup = [];
  for (
    let leader = firstGroupedLayer;
    leader < numLayers;
    leader += groupSize
  ) {
    perGroup.push({
      leader,
      ...summarize(groupRecords
        .filter(record => record.leader === leader)
        .map(record => record.latency)),
    });
  }

  const prefetchComputeEvents = prefetchComputeStream === null
    ? []
    : stableEvents.filter(event =>
      event.tid === prefetchComputeStream &&
      !MAIN_GATHER.test(event.name) &&
      !/^EVENT_/.test(event.name)
    );
  const prefetchLookupEvents = prefetchComputeStream === null
    ? []
    : stableEvents.filter(event =>
      event.tid === prefetchComputeStream && LOOKUP.test(event.name)
    );
  const prefetchIndexerEvents = prefetchComputeStream === null
    ? []
    : stableEvents.filter(event =>
      event.tid === prefetchComputeStream && PREFETCH_INDEXER.test(event.name)
    );

  const prefetchGatherQuartets = [];
  for (let index = 0; index + groupSize <= prefetchGatherEvents.length;) {
    const quartet = prefetchGatherEvents.slice(index, index + groupSize);
    if (quartet.length < groupSize) break;
    prefetchGatherQuartets.push(
      quartet.at(-1).ts + quartet.at(-1).dur - quartet[0].ts,
    );
    index += groupSize;
  }

  const sideStreams = new Set([
    mainGatherStream,
    prefetchComputeStream,
    prefetchGatherStream,
  ].filter(tid => tid !== null));
  const mainOperatorEvents = pattern => stableEvents.filter(event =>
    !sideStreams.has(event.tid) &&
    !MAINTAIN.test(event.name) &&
    pattern.test(event.name)
  );
  const competitionPatterns = {
    sparseAttention: /^SparseFlashAttention$/,
    attentionMerge: /^batch_matmul_transpose$/,
    moeGating: /^MoeGatingTopK$/,
    moeDispatch: /MoeDistributeDispatch/,
    groupedMatmulSwiglu: /GroupedMatmulSwigluQuant/,
    groupedMatmul: /GroupedMatmul(?!SwigluQuant)/,
    moeCombine: /MoeDistributeCombine/,
  };
  const competition = {};
  const stableMainGatherEvents = mainGatherEvents.filter(event =>
    event.ts >= stableStart && event.ts < stableEnd
  );
  competition.prefetchIndexer = {
    againstPrefetchGather: overlapBreakdown(
      prefetchIndexerEvents,
      prefetchGatherEvents,
    ),
    againstMainGather: overlapBreakdown(
      prefetchIndexerEvents,
      stableMainGatherEvents,
    ),
  };
  competition.prefetchLookup = {
    againstPrefetchGather: overlapBreakdown(
      prefetchLookupEvents,
      prefetchGatherEvents,
    ),
    againstMainGather: overlapBreakdown(
      prefetchLookupEvents,
      stableMainGatherEvents,
    ),
  };
  competition.mainGather = {
    againstPrefetchGather: overlapBreakdown(
      stableMainGatherEvents,
      prefetchGatherEvents,
    ),
    prefetchGatherAgainstMainGather: overlapBreakdown(
      prefetchGatherEvents,
      stableMainGatherEvents,
    ),
  };
  for (const [name, pattern] of Object.entries(competitionPatterns)) {
    const targetEvents = mainOperatorEvents(pattern);
    competition[name] = {
      againstPrefetchGather: overlapBreakdown(
        targetEvents,
        prefetchGatherEvents,
      ),
      againstPrefetchLookup: overlapBreakdown(
        targetEvents,
        prefetchLookupEvents,
      ),
    };
  }

  const prefetchGathersPerStep = stableSteps.map(step =>
    prefetchGatherEvents.filter(event =>
      event.ts >= step.start && event.ts < step.end
    ).length
  );
  const mainGathersPerStep = stableSteps.map(step =>
    mainGatherEvents.filter(event =>
      event.ts >= step.start && event.ts < step.end
    ).length
  );

  return {
    label,
    file: parsed.file,
    hardwarePid,
    processName: parsed.processNames.get(hardwarePid) ?? "",
    streams: {
      mainGather: mainGatherStream,
      prefetch: prefetchComputeStream,
      prefetchCompute: prefetchComputeStream,
      prefetchGather: prefetchGatherStream,
      maintain: maintainStream?.count > 0 ? maintainStream.tid : null,
      mainCompute: mainComputeStreams,
    },
    streamGatherCounts: gatherCounts,
    decode: {
      anchor: "main asynchronous AsuKvGather",
      startTask: decoded.startTask,
      completeStepCount: decoded.steps.length,
      stableStepCount: stableSteps.length,
      stableStep: summarize(stableSteps.map(step => step.duration)),
      stableStepTrimmedAverage: trimmedAverage(
        stableSteps.map(step => step.duration),
      ),
      stableLayer: summarize(stableLayerRecords.map(record => record.latency)),
      stableLayerWithoutBoundary: summarize(stableLayerRecords
        .filter(record => record.layer < numLayers - 1)
        .map(record => record.latency)),
      stableLastLayerToNextStep: summarize(stableLayerRecords
        .filter(record => record.layer === numLayers - 1)
        .map(record => record.latency)),
      prefetchReleaseLayers: [...prefetchReleaseLayers],
      stablePrefetchReleaseLayer: summarize(
        releaseLayerRecords.map(record => record.latency),
      ),
      stableNonReleaseLayer: summarize(
        nonReleaseLayerRecords.map(record => record.latency),
      ),
      stableGroup: summarize(groupRecords.map(record => record.latency)),
      stableGroupWithoutBoundary: summarize(groupRecords
        .filter(record => !record.includesStepBoundary)
        .map(record => record.latency)),
      perLayer,
      perGroup,
      stepHistory: stableSteps.map((step, index) => ({
        fromEnd: stableSteps.length - index,
        duration: step.duration,
      })),
    },
    stableKeyOperators: {
      mainIndexer: operatorSummary(
        stableEvents.filter(event => event.tid !== prefetchComputeStream),
        /^LightningIndexer$/,
      ),
      mainLookup: operatorSummary(
        stableEvents.filter(event => event.tid !== prefetchComputeStream),
        LOOKUP,
      ),
      mainGather: summarize(stableMainGatherEvents.map(event => event.dur)),
      prefetchIndexer: prefetchComputeStream === null
        ? {count: 0}
        : operatorSummary(stableEvents, PREFETCH_INDEXER, prefetchComputeStream),
      prefetchQliFusion: prefetchComputeStream === null
        ? {count: 0}
        : operatorSummary(stableEvents, PREFETCH_QLI, prefetchComputeStream),
      prefetchLookup: prefetchComputeStream === null
        ? {count: 0}
        : operatorSummary(stableEvents, LOOKUP, prefetchComputeStream),
      prefetchGather: summarize(prefetchGatherEvents.map(event => event.dur)),
      prefetchGatherQuartet: summarize(prefetchGatherQuartets),
      maintain: operatorSummary(stableEvents, MAINTAIN),
      sparseAttention: operatorSummary(stableEvents, /^SparseFlashAttention$/),
      moeGating: operatorSummary(stableEvents, /^MoeGatingTopK$/),
      moeDispatch: operatorSummary(stableEvents, /MoeDistributeDispatch/),
      groupedMatmulSwiglu: operatorSummary(stableEvents, /GroupedMatmulSwigluQuant/),
      groupedMatmul: operatorSummary(stableEvents, /GroupedMatmul(?!SwigluQuant)/),
      moeCombine: operatorSummary(stableEvents, /MoeDistributeCombine/),
    },
    prefetch: {
      active: prefetchComputeStream !== null,
      gatherLayout: prefetchGatherSchedule,
      gatherStream: prefetchGatherStream,
      expectedGathersPerStep: expectedPrefetchGathers,
      observedMainGathersPerStep: summarize(mainGathersPerStep),
      observedPrefetchGathersPerStep: summarize(prefetchGathersPerStep),
      computeOperator: summarize(prefetchComputeEvents.map(event => event.dur)),
      computeOperatorByName: summarizeByName(prefetchComputeEvents),
      computeChainToIndexer: summarizeComputeChainsToIndexer(
        prefetchComputeEvents,
      ),
      competition,
    },
  };
}

function ratioChange(base, candidate) {
  if (!Number.isFinite(base) || !Number.isFinite(candidate) || base === 0) {
    return null;
  }
  return (candidate - base) / base;
}

function summarizeLayerSelection(report, selectedLayers) {
  let total = 0;
  let count = 0;
  for (const layer of report.decode.perLayer) {
    if (!selectedLayers.has(layer.layer)) continue;
    total += layer.total ?? 0;
    count += layer.count ?? 0;
  }
  return count === 0 ? {count: 0} : {count, total, avg: total / count};
}

function compareReports(base, prefetch) {
  const releaseLayers = new Set(prefetch.decode.prefetchReleaseLayers);
  const nonReleaseLayers = new Set();
  for (let layer = 0; layer < numLayers - 1; layer++) {
    if (!releaseLayers.has(layer)) nonReleaseLayers.add(layer);
  }
  const baseRelease = summarizeLayerSelection(base, releaseLayers);
  const prefetchRelease = summarizeLayerSelection(prefetch, releaseLayers);
  const baseNonRelease = summarizeLayerSelection(base, nonReleaseLayers);
  const prefetchNonRelease = summarizeLayerSelection(
    prefetch,
    nonReleaseLayers,
  );
  const operatorChanges = {};
  for (const name of Object.keys(base.stableKeyOperators)) {
    const baseValue = base.stableKeyOperators[name]?.avg;
    const prefetchValue = prefetch.stableKeyOperators[name]?.avg;
    operatorChanges[name] = {
      baseAvg: baseValue ?? null,
      prefetchAvg: prefetchValue ?? null,
      relativeChange: ratioChange(baseValue, prefetchValue),
    };
  }
  return {
    stableStep: {
      baseAvg: base.decode.stableStep.avg,
      prefetchAvg: prefetch.decode.stableStep.avg,
      delta: prefetch.decode.stableStep.avg - base.decode.stableStep.avg,
      relativeChange: ratioChange(
        base.decode.stableStep.avg,
        prefetch.decode.stableStep.avg,
      ),
    },
    stableStepTrimmedAverage: {
      baseAvg: base.decode.stableStepTrimmedAverage,
      prefetchAvg: prefetch.decode.stableStepTrimmedAverage,
      delta: prefetch.decode.stableStepTrimmedAverage -
        base.decode.stableStepTrimmedAverage,
      relativeChange: ratioChange(
        base.decode.stableStepTrimmedAverage,
        prefetch.decode.stableStepTrimmedAverage,
      ),
    },
    stableLayerWithoutBoundary: {
      baseAvg: base.decode.stableLayerWithoutBoundary.avg,
      prefetchAvg: prefetch.decode.stableLayerWithoutBoundary.avg,
      delta: prefetch.decode.stableLayerWithoutBoundary.avg -
        base.decode.stableLayerWithoutBoundary.avg,
      relativeChange: ratioChange(
        base.decode.stableLayerWithoutBoundary.avg,
        prefetch.decode.stableLayerWithoutBoundary.avg,
      ),
    },
    stableGroupWithoutBoundary: {
      baseAvg: base.decode.stableGroupWithoutBoundary.avg,
      prefetchAvg: prefetch.decode.stableGroupWithoutBoundary.avg,
      delta: prefetch.decode.stableGroupWithoutBoundary.avg -
        base.decode.stableGroupWithoutBoundary.avg,
      relativeChange: ratioChange(
        base.decode.stableGroupWithoutBoundary.avg,
        prefetch.decode.stableGroupWithoutBoundary.avg,
      ),
    },
    stablePrefetchReleaseLayer: {
      baseAvg: baseRelease.avg,
      prefetchAvg: prefetchRelease.avg,
      delta: prefetchRelease.avg - baseRelease.avg,
      relativeChange: ratioChange(
        baseRelease.avg,
        prefetchRelease.avg,
      ),
    },
    stableNonReleaseLayer: {
      baseAvg: baseNonRelease.avg,
      prefetchAvg: prefetchNonRelease.avg,
      delta: prefetchNonRelease.avg - baseNonRelease.avg,
      relativeChange: ratioChange(
        baseNonRelease.avg,
        prefetchNonRelease.avg,
      ),
    },
    operatorChanges,
  };
}

function formatUs(value) {
  return value === null || value === undefined
    ? "n/a"
    : `${value.toFixed(3)} us`;
}

function formatPercent(value) {
  return value === null || value === undefined
    ? "n/a"
    : `${(value * 100).toFixed(2)}%`;
}

function printSummary(report) {
  const {base, prefetch, comparison} = report;
  console.log("GLM-5.2 DSA offload vs hidden-state prefetch");
  console.log("");
  console.log("| Metric | Offload | Prefetch | Delta | Change |");
  console.log("|---|---:|---:|---:|---:|");
  for (const [name, value] of [
    ["stable decode step", comparison.stableStep],
    ["stable decode step (10% trimmed)", comparison.stableStepTrimmedAverage],
    ["stable layer (0-76)", comparison.stableLayerWithoutBoundary],
    ["stable 4-layer group (except last)", comparison.stableGroupWithoutBoundary],
    ["prefetch Gather release layer", comparison.stablePrefetchReleaseLayer],
    ["non-release layer (0-76)", comparison.stableNonReleaseLayer],
  ]) {
    console.log(
      `| ${name} | ${formatUs(value.baseAvg)} | ` +
      `${formatUs(value.prefetchAvg)} | ${formatUs(value.delta)} | ` +
      `${formatPercent(value.relativeChange)} |`,
    );
  }
  console.log("");
  console.log("| Operator | Offload avg | Prefetch avg | Change |");
  console.log("|---|---:|---:|---:|");
  for (const name of [
    "mainIndexer",
    "mainLookup",
    "mainGather",
    "sparseAttention",
    "moeDispatch",
    "groupedMatmulSwiglu",
    "groupedMatmul",
    "moeCombine",
  ]) {
    const value = comparison.operatorChanges[name];
    console.log(
      `| ${name} | ${formatUs(value.baseAvg)} | ` +
      `${formatUs(value.prefetchAvg)} | ` +
      `${formatPercent(value.relativeChange)} |`,
    );
  }
  console.log("");
  console.log(
    `Streams: offload main-gather=${base.streams.mainGather}; ` +
    `prefetch main-gather=${prefetch.streams.mainGather}, ` +
    `prefetch-compute=${prefetch.streams.prefetchCompute}, ` +
    `prefetch-gather=${prefetch.streams.prefetchGather}, ` +
    `maintain=${prefetch.streams.maintain}.`,
  );
  console.log(
    `Prefetch gather: ${prefetch.stableKeyOperators.prefetchGather.count} ` +
    `calls, avg=${formatUs(prefetch.stableKeyOperators.prefetchGather.avg)}, ` +
    `quartet avg=${formatUs(prefetch.stableKeyOperators.prefetchGatherQuartet.avg)}.`,
  );
}

async function main() {
  const baseParsed = await parseTrace(baseFile);
  const prefetchParsed = await parseTrace(prefetchFile);
  const base = analyzeParsed(baseParsed, "offload");
  const prefetch = analyzeParsed(prefetchParsed, "prefetch");
  const report = {
    config: {
      numLayers,
      groupSize,
      firstGroupedLayer,
      tailSteps,
      forcedPrefetchGatherSchedule,
    },
    base,
    prefetch,
    comparison: compareReports(base, prefetch),
  };
  printSummary(report);
  if (outputFile !== undefined) {
    fs.writeFileSync(
      outputFile,
      `${JSON.stringify(report, null, pretty ? 2 : 0)}\n`,
    );
  }
}

main().catch(error => {
  console.error(error.stack ?? error.message ?? String(error));
  process.exitCode = 1;
});
