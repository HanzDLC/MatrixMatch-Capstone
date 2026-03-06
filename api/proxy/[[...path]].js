const http = require("http");
const https = require("https");
const { URL } = require("url");

function normalizeBaseUrl(rawUrl) {
    if (!rawUrl) {
        return null;
    }

    const trimmed = rawUrl.trim().replace(/\/+$/, "");
    if (!trimmed) {
        return null;
    }

    return trimmed;
}

function buildTargetUrl(baseUrl, pathParts, queryString) {
    const safeBase = normalizeBaseUrl(baseUrl);
    const joinedPath = Array.isArray(pathParts) ? pathParts.join("/") : pathParts || "";
    return new URL(`${safeBase}/${joinedPath}${queryString || ""}`);
}

module.exports = (req, res) => {
    const baseUrl = normalizeBaseUrl(process.env.BACKEND_PROXY_URL);
    if (!baseUrl) {
        res.statusCode = 500;
        res.setHeader("content-type", "application/json; charset=utf-8");
        res.end(JSON.stringify({
            error: "BACKEND_PROXY_URL is not configured in Vercel.",
        }));
        return;
    }

    const queryStart = req.url.indexOf("?");
    const queryString = queryStart >= 0 ? req.url.slice(queryStart) : "";
    const targetUrl = buildTargetUrl(baseUrl, req.query.path, queryString);
    const client = targetUrl.protocol === "https:" ? https : http;

    const upstreamHeaders = { ...req.headers };
    upstreamHeaders.host = targetUrl.host;
    upstreamHeaders["x-forwarded-host"] = req.headers.host || "";
    upstreamHeaders["x-forwarded-proto"] = "https";
    upstreamHeaders["ngrok-skip-browser-warning"] = "1";

    const upstreamReq = client.request(
        targetUrl,
        {
            method: req.method,
            headers: upstreamHeaders,
        },
        (upstreamRes) => {
            res.statusCode = upstreamRes.statusCode || 502;

            for (const [headerName, headerValue] of Object.entries(upstreamRes.headers)) {
                if (headerValue !== undefined) {
                    res.setHeader(headerName, headerValue);
                }
            }

            upstreamRes.pipe(res);
        }
    );

    upstreamReq.on("error", (error) => {
        res.statusCode = 502;
        res.setHeader("content-type", "application/json; charset=utf-8");
        res.end(JSON.stringify({
            error: "Failed to reach the local backend through ngrok.",
            detail: error.message,
        }));
    });

    if (req.method === "GET" || req.method === "HEAD") {
        upstreamReq.end();
        return;
    }

    req.pipe(upstreamReq);
};
