const DEFAULT_BACKEND_PROXY_URL = "https://guidebooky-gideon-pellucid.ngrok-free.dev";

function normalizeBaseUrl(rawUrl) {
    if (!rawUrl) {
        return null;
    }

    const trimmed = rawUrl.trim().replace(/\/+$/, "");
    return trimmed || null;
}

function stripApiPrefix(pathname) {
    const stripped = pathname.replace(/^\/api(?:\/|$)/, "/");
    return stripped || "/";
}

function getForwardedPathname(incomingUrl) {
    const explicitPathname = incomingUrl.searchParams.get("__pathname");
    if (explicitPathname) {
        return explicitPathname.startsWith("/")
            ? explicitPathname
            : `/${explicitPathname}`;
    }
    return stripApiPrefix(incomingUrl.pathname);
}

function joinPath(basePath, requestPath) {
    const normalizedBase = (basePath || "/").replace(/\/+$/, "");
    const normalizedRequest = (requestPath || "/").replace(/^\/+/, "");

    if (!normalizedRequest) {
        return normalizedBase || "/";
    }

    if (!normalizedBase || normalizedBase === "/") {
        return `/${normalizedRequest}`;
    }

    return `${normalizedBase}/${normalizedRequest}`;
}

function copyRequestHeaders(request) {
    const headers = new Headers(request.headers);
    const incomingUrl = new URL(request.url);

    headers.set("x-forwarded-host", incomingUrl.host);
    headers.set("x-forwarded-proto", "https");
    headers.set("ngrok-skip-browser-warning", "1");

    return headers;
}

async function buildRequestInit(request) {
    const init = {
        method: request.method,
        headers: copyRequestHeaders(request),
        redirect: "manual",
    };

    if (request.method !== "GET" && request.method !== "HEAD") {
        init.body = await request.arrayBuffer();
    }

    return init;
}

function buildTargetUrl(request, baseUrl) {
    const incomingUrl = new URL(request.url);
    const targetUrl = new URL(baseUrl);
    const forwardedPath = getForwardedPathname(incomingUrl);
    const targetSearchParams = new URLSearchParams(incomingUrl.search);

    targetSearchParams.delete("__pathname");

    targetUrl.pathname = joinPath(targetUrl.pathname, forwardedPath);
    const searchString = targetSearchParams.toString();
    targetUrl.search = searchString ? `?${searchString}` : "";

    return targetUrl;
}

function jsonResponse(status, payload) {
    return new Response(JSON.stringify(payload), {
        status,
        headers: {
            "content-type": "application/json; charset=utf-8",
        },
    });
}

export async function proxyRequest(request) {
    const baseUrl = normalizeBaseUrl(
        process.env.BACKEND_PROXY_URL ||
        process.env.BACKEND_URL ||
        DEFAULT_BACKEND_PROXY_URL
    );
    if (!baseUrl) {
        return jsonResponse(500, {
            error: "No backend proxy URL is configured.",
        });
    }

    const targetUrl = buildTargetUrl(request, baseUrl);

    try {
        const upstreamResponse = await fetch(
            targetUrl,
            await buildRequestInit(request)
        );

        return new Response(upstreamResponse.body, {
            status: upstreamResponse.status,
            headers: upstreamResponse.headers,
        });
    } catch (error) {
        return jsonResponse(502, {
            error: "Failed to reach the local backend through ngrok.",
            detail: error instanceof Error ? error.message : String(error),
        });
    }
}
