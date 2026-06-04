-- Run once on first container creation. Idempotent (`IF NOT EXISTS`).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram index for BM25-style fallback
CREATE EXTENSION IF NOT EXISTS unaccent;  -- German diacritics normalization
