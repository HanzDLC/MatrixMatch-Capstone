# Vercel + Ngrok Setup

This project is still a server-rendered Flask app, so Vercel is configured here as a reverse proxy in front of your local backend.

## Backend

1. Start Flask locally on a fixed port, for example:
   `gunicorn --bind 0.0.0.0:8000 app:app`
2. Start ngrok against that port:
   `ngrok http 8000`
3. Copy the HTTPS forwarding URL from ngrok.

## Vercel

1. Import this repository into Vercel.
2. Add an environment variable named `BACKEND_PROXY_URL`.
3. Set `BACKEND_PROXY_URL` to your ngrok HTTPS URL, for example:
   `https://example.ngrok-free.app`
4. Deploy.

## Why Vercel Won't Install Python Packages

This repository is configured so the Vercel project behaves like a plain "Other" project with:

- `framework: null`
- empty `installCommand`
- empty `buildCommand`

That prevents Vercel from trying to build the Flask app or install the heavy Python dependencies from `requirements.txt`.

## Important

- Your computer must stay on while Vercel proxies to ngrok.
- If your ngrok URL changes, update `BACKEND_PROXY_URL` in Vercel and redeploy.
- This is suitable for demos and testing, not production hosting.
