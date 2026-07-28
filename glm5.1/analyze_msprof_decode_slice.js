#!/usr/bin/env node

// Extract every hardware event that overlaps one timestamp interval without
// loading a multi-gigabyte msprof Chrome trace into memory.
const fs = require("fs");

const [file, pidRaw, startRaw, endRaw] = process.argv.slice(2);
const pidWanted = Number(pidRaw);
const start = Number(startRaw);
const end = Number(endRaw);
if (!file || !Number.isFinite(pidWanted) || !Number.isFinite(start) ||
    !Number.isFinite(end)) {
  throw new Error(
    "usage: trace_msprof_slice.js FILE ASCEND_HARDWARE_PID START_TS END_TS",
  );
}

const events = [];
const threadNames = new Map();
const processNames = new Map();

function consume(obj) {
  if (!obj || typeof obj !== "object") return;
  if (obj.ph === "M" && obj.name === "thread_name") {
    threadNames.set(`${obj.pid}|${obj.tid}`, obj.args?.name ?? "");
    return;
  }
  if (obj.ph === "M" && obj.name === "process_name") {
    processNames.set(Number(obj.pid), obj.args?.name ?? "");
    return;
  }
  if (obj.ph !== "X" || Number(obj.pid) !== pidWanted) return;
  const ts = Number(obj.ts);
  const dur = Number(obj.dur);
  if (!Number.isFinite(ts) || !Number.isFinite(dur)) return;
  if (ts + dur < start || ts >= end) return;
  events.push({
    name: String(obj.name ?? ""),
    pid: Number(obj.pid),
    tid: Number(obj.tid),
    ts,
    rel: ts - start,
    dur,
    endRel: ts + dur - start,
    task: Number(
      obj.args?.["Task Id"] ?? obj.args?.["Task ID"] ?? -1,
    ),
  });
}

let depth = 0;
let inString = false;
let escape = false;
let objectBuffer = "";
const stream = fs.createReadStream(file, {
  encoding: "utf8",
  highWaterMark: 8 * 1024 * 1024,
});
stream.on("data", chunk => {
  for (const c of chunk) {
    if (depth > 0 || c === "{") objectBuffer += c;
    if (inString) {
      if (escape) escape = false;
      else if (c === "\\") escape = true;
      else if (c === "\"") inString = false;
      continue;
    }
    if (c === "\"") {
      inString = true;
    } else if (c === "{") {
      depth++;
    } else if (c === "}") {
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
stream.on("end", () => {
  const seen = new Set();
  const unique = events.filter(event => {
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
  unique.sort((a, b) => a.ts - b.ts || a.tid - b.tid);
  const streams = [...new Set(unique.map(event => event.tid))].sort(
    (a, b) => a - b,
  );
  console.log(JSON.stringify({
    process: processNames.get(pidWanted),
    streams: streams.map(tid => ({
      tid,
      name: threadNames.get(`${pidWanted}|${tid}`) ?? "",
      events: unique.filter(event => event.tid === tid),
    })),
  }, null, 2));
});
