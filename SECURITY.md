# Security policy

TriScan processes untrusted documents. Treat model output as untrusted data too.

## Safe defaults

- File signatures are checked instead of trusting the uploaded filename.
- File size, PDF page count and rendered image size are bounded.
- Document text is explicitly isolated from system instructions in prompts.
- JSON parsing uses `json.loads`; the fallback uses only `ast.literal_eval`.
- Artifact routes validate run IDs and filenames against stored metadata.
- The default provider is a mock and makes no external network call.

## Deployment requirements

Do not expose the development server directly to the public internet. Add identity,
authorization, rate limiting, TLS, isolated workers, malware scanning, storage
encryption, retention/deletion controls and observability. Never send confidential
documents to a remote model provider without permission and an appropriate data
processing agreement.

Report security issues privately to the repository owner. Do not include real
documents, credentials, model API keys or personal data in a public report.

