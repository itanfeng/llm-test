#!/usr/bin/env node

// Stream a large msprof Chrome trace and summarize the operator signatures
// needed to distinguish static-W8 MLAPO, dynamic-W8 native MLA, W4A8 MoE,
// and hidden-state prefetch. The input trace is never loaded as one JSON
// object, so multi-gigabyte profiling files can be analyzed directly.
//
// Usage:
//   node analyze_msprof_w4a8.js TRACE.json [--top=30] [--pretty]

"use strict";

const fs = require("fs");

const args = process.argv.slice(2);
const file = args.find(argument => !argument.startsWith("--"));
const pretty = args.includes("--pretty");
const topOption = args.find(argument => argument.startsWith("--top="));
const top = topOption === undefined
  ? 30
  : Number.parseInt(topOption.split("=")[1], 10);

if (file === undefined || !Number.isInteger(top) || top < 1) {
  throw new Error(
    "usage: node analyze_msprof_w4a8.js TRACE.json [--top=30] [--pretty]",
  );
}
if (!fs.existsSync(file)) {
  throw new Error(`trace file does not exist: ${file}`);
}

const keep = /mla_(?:query_)?preprocess|DynamicQuant|AscendQuant|QuantBatchMatmul|WeightQuantBatchMatmul|KvRmsNormRopeCache|InterleaveRope|BatchMatMul|MatMul|LayerNorm|RmsNorm|ScatterNdUpdate|LightningIndexer|AsuHbmIndexLookup|AsuHbmIndexMaintain|AsuKvGather|HiSparseFlashAttention|DaAttentionMerge|batch_matmul_transpose|MoeGatingTopK|MoeDistributeDispatch|MoeDistributeCombine|GroupedMatmul|_swiglu_quant_kernel/;
const processNames = new Map();
const threadNames = new Map();
const statistics = new Map();
const seen = new Set();

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

stream.on("end", () => {
  const groups = [...statistics.values()];
  const pidScores = new Map();
  for (const group of groups) {
    if (
      /LightningIndexer|AsuHbmIndexLookup|AsuKvGather|HiSparseFlashAttention/.test(
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

  const mainStream = pickStream(/^HiSparseFlashAttention$/);
  const prefetchCandidate = pickStream(/^LightningIndexerHiCached$/);
  const prefetchStream = count(
    /^LightningIndexerHiCached$/,
    prefetchCandidate,
  ) > 0
    ? prefetchCandidate
    : null;
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
  const mainCount = pattern => count(pattern, mainStream);

  const mlapoCount = mainCount(/^mla_preprocess$/);
  const nativeKvCount = mainCount(/KvRmsNormRopeCache/);
  const dynamicQuantCount = mainCount(/DynamicQuant/);
  const hiCachedCount = aggregate(/LightningIndexerHiCached/).count;
  const queryPreprocessCount = aggregate(/^mla_query_preprocess$/).count;
  const prefetchDynamicQuantCount = prefetchStream === null
    ? 0
    : count(/DynamicQuant/, prefetchStream);
  const mainIndexerCount = mainCount(/^LightningIndexer$/);
  const mainLookupCount = mainCount(/AsuHbmIndexLookup/);
  const sfaCount = mainCount(/^HiSparseFlashAttention$/);
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

  const roles = new Map([
    [mainStream, "main"],
    [gatherStream, "gather"],
    [maintainStream, "maintain"],
  ]);
  if (prefetchStream !== null) roles.set(prefetchStream, "prefetch_compute");

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
      prefetchCompute: prefetchStream,
      gather: gatherStream,
      maintain: maintainStream,
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
      ["HiSparseFlashAttention", /^HiSparseFlashAttention$/],
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

  console.log(JSON.stringify(output, null, pretty ? 2 : 0));
});
