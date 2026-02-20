#!/bin/sh
set -eu

: "${MYSQL_HOST:?MYSQL_HOST is required}"
: "${MYSQL_PORT:=3306}"
: "${MYSQL_DB:?MYSQL_DB is required}"
: "${MYSQL_USER:?MYSQL_USER is required}"
: "${MYSQL_PASSWORD:=}"

: "${PG_HOST:?PG_HOST is required}"
: "${PG_PORT:=5432}"
: "${PG_DB:?PG_DB is required}"
: "${PG_USER:?PG_USER is required}"
: "${PG_PASSWORD:=}"

MYSQL_URL="mysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}"
PG_URL="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${PG_DB}"

echo "Starting migration from ${MYSQL_HOST}/${MYSQL_DB} to ${PG_HOST}/${PG_DB}..."
exec pgloader "$MYSQL_URL" "$PG_URL"
