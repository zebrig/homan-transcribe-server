# Upstream attribution

`server.cpp` is derived from
[`whisper.cpp/examples/server/server.cpp`](https://github.com/ggml-org/whisper.cpp)
and remains available under the upstream MIT License.

The Docker build pins the upstream source image by digest:

```text
ghcr.io/ggml-org/whisper.cpp@sha256:0ae4f56c4ef3e499092fe0fb8cdc581c1270fd51e44e96978ec43b25137b8070
```

Material changes in this project include:

- a persistent, single-request-worker Vulkan lifecycle;
- a conditionally OpenAI-compatible transcription route with a server-owned
  decoder profile;
- the Homan-native multipart batch contract;
- receiver-side packing of short VAD clips, timeline mapping, and safe
  per-item fallback;
- Docker, Caddy, authentication, and long-meeting deployment configuration.

This project is not affiliated with or endorsed by OpenAI. “OpenAI-compatible”
describes a limited HTTP compatibility surface, not behavioral or schema
identity with the hosted OpenAI service.
