# Security policy

## Supported versions

Until 1.0, only the latest released version receives security fixes.

## Reporting a vulnerability

Please use GitHub's private security-advisory reporting flow for this
repository. Do not open a public issue containing an unpatched vulnerability.
Include affected versions, a minimal reproduction, impact, and any suggested
mitigation. Maintainers will acknowledge a report within seven days and will
coordinate disclosure after a fix is available.

Malformed model or explanation input should return an error rather than panic.
Reports of panics, unbounded allocation, or unsafe resource consumption caused
by untrusted serialized input are considered security relevant.
