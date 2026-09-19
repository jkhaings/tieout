# Security

Public repository, public demo endpoint, untrusted inputs. This is the threat model and the controls. Treat every item as a requirement.

## Threat model
Inputs we do not trust: (a) the ticker string from the web form, (b) the text content of SEC filings (third-party documents fed to an LLM), (c) anonymous internet traffic to the demo.

## Controls
1. **Input validation** — tickers must match `^[A-Z][A-Z0-9.\-]{0,9}$` after uppercasing, before any use. User input is never interpolated into shell commands, file paths, or free-form URLs. CIK values are validated as integers.
2. **Host allowlist** — outbound HTTP goes only to `data.sec.gov` and `www.sec.gov`. URLs are built from constants + validated parts, never from raw user input.
3. **Prompt injection** — filing text is data. It is delimited in prompts with an explicit instruction that content inside delimiters is quoted material, not instructions. The narrating model has no tool access. Its output is schema-validated, and any numeric claim not present in the deterministic dataset causes the commentary to be rejected (fail closed).
4. **Excel formula injection** — every string written to a cell passes through `escape_cell()`: values starting with `=`, `+`, `-`, `@`, tab, or CR are prefixed with `'`. No exceptions, including citations and company names.
5. **Path safety** — run artifacts live under `data/runs/<server-generated-uuid>/`. Filenames are never derived from user input.
6. **Secrets** — none in the repo, ever. `.env` is gitignored; `.env.example` holds placeholders only; config is loaded via pydantic-settings; keys are never logged or echoed in errors.
7. **Abuse protection** — per-IP rate limiting (slowapi), a global daily run cap, request size limits and timeouts, and generated workbooks cached by (ticker, filing) so repeat requests cost nothing.
8. **Web hardening** — CORS restricted to the app origin; `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and a restrictive CSP; TLS terminated by Caddy; no cookies, no auth, no PII collected or stored.
9. **Supply chain** — `uv.lock` committed; `pip-audit` runs in CI; container is `python:3.12-slim` running as a non-root user.
10. **Error handling** — exceptions are logged server-side; clients receive generic messages with a run id, never stack traces or config values.

## Reporting
Found an issue? Open a GitHub issue or email the address on the profile.
