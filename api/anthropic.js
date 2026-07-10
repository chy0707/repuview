// Serverless proxy for the Anthropic API.
// The browser never sees the API key — it lives only in the ANTHROPIC_API_KEY
// environment variable on Vercel. The frontend POSTs to /api/anthropic and this
// function forwards the request to Anthropic server-side.

const MIN_MS = 60 * 1000;
const DAY_MS = 24 * 60 * 60 * 1000;
const MAX_PER_MIN = 30;   // one LIVE NEWS click ≈ 7 sub-requests; ~4 clicks/min
const MAX_PER_DAY = 200;  // ~28 clicks/day
const minHits = new Map();
const dayHits = new Map();

function bump(map, ip, windowMs, max) {
  const now = Date.now();
  const rec = map.get(ip);
  if (!rec || now > rec.resetAt) {
    map.set(ip, { count: 1, resetAt: now + windowMs });
    return false;
  }
  rec.count += 1;
  return rec.count > max;
}

function rateLimited(ip) {
  return bump(minHits, ip, MIN_MS, MAX_PER_MIN) || bump(dayHits, ip, DAY_MS, MAX_PER_DAY);
}

function json(res, status, obj) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(obj));
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return json(res, 405, { error: "Method not allowed" });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return json(res, 500, { error: "Missing ANTHROPIC_API_KEY" });
  }

  const fwd = req.headers ? (req.headers["x-forwarded-for"] || "") : "";
  const ip = String(fwd).split(",")[0].trim() || "unknown";
  if (rateLimited(ip)) {
    return json(res, 429, { error: "Rate limit reached. Try again in a minute." });
  }

  // Read the request body manually — robust across Vercel body-parsing modes.
  let raw = "";
  try {
    for await (const chunk of req) raw += chunk;
  } catch (e) {
    return json(res, 400, { error: "Could not read body" });
  }

  let body;
  try {
    body = raw ? JSON.parse(raw) : {};
  } catch (e) {
    return json(res, 400, { error: "Invalid JSON body" });
  }

  const { messages, search } = body || {};
  if (!Array.isArray(messages) || messages.length === 0) {
    return json(res, 400, { error: "messages array is required" });
  }

  const payload = { model: "claude-sonnet-4-20250514", max_tokens: 1000, messages };
  if (search) payload.tools = [{ type: "web_search_20250305", name: "web_search" }];

  try {
    const upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(payload),
    });
    const data = await upstream.json();
    return json(res, upstream.status, data);
  } catch (err) {
    return json(res, 502, { error: "Upstream request failed" });
  }
}
