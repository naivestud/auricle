# Security Policy

## Supported versions

Only the latest release receives security fixes. Older versions are best-effort.

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them
privately so they can be addressed before disclosure:

- Use GitHub's **Report a vulnerability** feature on the repository
  (Security tab), or
- Contact the maintainer directly.

You should receive an acknowledgement within a few days. If the report is
accepted, a fix will be prepared and credited appropriately.

## Scope notes

auricle ships model code and tooling; it does not bundle pretrained weights.
The OpenAI-compatible backend sends prompts over HTTP to a URL you configure —
point it only at endpoints you trust, and keep API keys out of your
checkpoints and manifests.
