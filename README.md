# Homan Transcribe Server

A Dockerized `whisper.cpp` service with Vulkan acceleration and one
server-managed `large-v3-turbo` model. It exposes:

- `POST /v1/audio/transcriptions` — a limited, conditionally
  OpenAI-compatible single-file endpoint;
- `POST /v1/homan/audio/transcriptions` — a Homan-native batch endpoint for
  role-preserving microphone/system VAD clips;
- Caddy TLS, automatic Let's Encrypt certificates, and bearer authentication.

Diarization is not part of this version.

## Why the native batch endpoint exists

Homan records microphone and system audio separately, runs AEC on the
microphone path, and applies Silero VAD before local ASR. Sending one mixed WAV
would lose `You`/`Others` source identity and waste bandwidth. Sending every
short VAD clip as a separate HTTP request is correct but slow: the GPU pays
Whisper's fixed setup and language-detection cost for every utterance.

The native endpoint receives all compressed VAD clips in one multipart body,
preserves each clip's ID/source/timeline, and internally packs short clips from
the same source into approximately 30-second inference reels. A 1.2-second
silence separates clips. If a returned Whisper segment crosses clip boundaries
or an item cannot be mapped safely, that item is transcribed again in isolation.

## Quick start

Requirements:

- Linux with Docker Compose;
- a Vulkan-capable GPU exposed as `/dev/dri`;
- `ggml-large-v3-turbo.bin` in `models/`;
- DNS for the configured hostname pointing at the server;
- TCP 80/443 and UDP 443 available to Caddy.

```bash
mkdir -p models
cp /path/to/ggml-large-v3-turbo.bin models/
cp .env.example .env
chmod 600 .env
# Edit CADDY_SITE_ADDRESS and TRANSCRIBE_API_KEY.
docker compose up --build -d
docker compose ps
```

The build disables AVX, AVX2, FMA, F16C, BMI2, and native CPU tuning so the
binary also starts on the production Ivy Bridge Celeron. `RADV_PERFTEST` is set
to `nogttspill` for the AMD/RADV deployment.

## Authentication

Both transcription routes require an exact header:

```text
Authorization: Bearer <TRANSCRIBE_API_KEY>
```

`GET /healthz` is intentionally unauthenticated and only checks Caddy. The
container's `/health` endpoint additionally reports whether the lazily loaded
model is resident.

Never commit `.env`, models, audio, multipart bodies, or transcription output.

## Conditionally OpenAI-compatible endpoint

The route follows the basic multipart shape of the official
[`POST /audio/transcriptions`](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
endpoint, but it is not a full implementation of the current OpenAI API.

```bash
curl https://stt.example.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $TRANSCRIBE_API_KEY" \
  -F "file=@meeting.m4a;type=audio/mp4" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json"
```

Field behavior:

| Field | Behavior |
|---|---|
| `file` | Required and decoded with FFmpeg. |
| `model` | Accepted and ignored; the server always uses `large-v3-turbo`. |
| `language` | Optional ISO-639-1 hint; omitted/invalid means automatic detection. |
| `prompt` | Mapped to Whisper and capped at 4096 bytes. |
| `response_format` | `json`, `text`, `srt`, `verbose_json`, or `vtt`. |
| `temperature` | Mapped only when it is a valid number from 0 to 1. |
| Any other field | Silently ignored. |

In particular, clients cannot change `best_of=5`, `beam_size=1`, token
timestamps, VAD, diarization, translation, model selection, or GPU/CPU
behavior. The nonstandard full language-probability map is disabled globally;
the detected language remains in `verbose_json`. Streaming, word timestamps,
logprobs, known speakers, and `diarized_json` are not implemented.

FFmpeg accepts the formats used by the official file endpoint—FLAC, MP3/MP4,
MPEG/MPGA, M4A, OGG, WAV, and WebM—and other formats supported by the installed
FFmpeg build. M4A/AAC is the preferred network format.

## Homan-native batch endpoint

The request is `multipart/form-data` with:

- one `manifest` part containing UTF-8 JSON;
- one compressed audio part for every manifest item;
- each `items[].file` value naming its corresponding multipart field.

Example manifest:

```json
{
  "schema_version": 1,
  "request_id": "8bd40400-6df5-4ad7-a558-813a64ca2e55",
  "options": { "concurrency": 2 },
  "items": [
    {
      "id": "microphone-0000",
      "source": "microphone",
      "start": 4.22,
      "end": 12.81,
      "file": "audio_0000"
    },
    {
      "id": "system-0000",
      "source": "system",
      "start": 13.10,
      "end": 25.40,
      "file": "audio_0001"
    }
  ]
}
```

```bash
curl https://stt.example.com/v1/homan/audio/transcriptions \
  -H "Authorization: Bearer $TRANSCRIBE_API_KEY" \
  -F "manifest=@manifest.json;type=application/json" \
  -F "audio_0000=@microphone-0000.m4a;type=audio/mp4" \
  -F "audio_0001=@system-0000.m4a;type=audio/mp4"
```

Contract rules:

- `schema_version` is currently `1`;
- `source` is `microphone`, `system`, or `legacy_mixed`;
- item IDs and multipart field names must be unique;
- one item represents at most 35 seconds of decoded audio;
- production accepts up to 2048 items and a 512 MB Caddy body;
- language is always automatic and model/decoder tuning is server-owned;
- client concurrency is capped by the server; production uses at most `2`;
- the response preserves every `id`, `source`, `start`, and `end`;
- item segment timestamps are relative to that item; item `start/end` remain
  absolute on the original source timeline;
- unknown manifest options and unrelated multipart fields have no effect.

The formal files are [openapi.yaml](openapi.yaml) and
[api/homan-batch-manifest.schema.json](api/homan-batch-manifest.schema.json).

## Long meetings

Homan should keep its existing AEC and VAD stages, encode each speech item as
mono AAC-LC in M4A with an explicit 64 kbit/s target, and send one native batch.
The verified native Apple profile uses `AVAssetWriter` with a 32 kHz container
rate (Apple rejects 64 kbit/s AAC at 16 kHz); the server normalizes decoded PCM
to 16 kHz. `startSession(atSourceTime: .zero)` produced about 66 ms of AAC
priming instead of 132 ms from the old preset. It must not upload a giant WAV.
Clients should not rely on `AVAssetExportPresetAppleM4A` defaults: the measured
Swift preset produced about 25.7 kbit/s plus a 132 ms non-zero start timestamp,
which degraded multilingual clips. The native target-64 profile recovered the
tested Russian/English code-switch; its selected clips measured roughly
50–61 kbit/s. FFmpeg 96 kbit/s provided no benefit over FFmpeg 64 kbit/s. The
item cap is sized for roughly two-hour meetings at the VAD density
observed in the real test (129 items for 17.1 minutes); a two-hour recording is
expected to produce about 900 items.

The server serializes independent HTTP requests. Parallelism only applies to
reels inside one authenticated batch. This is appropriate for the intended
load of a few files per hour, not a multi-tenant high-throughput service.

### Batch fallback strategy

The default server profile decodes Homan audio items with two workers and
repacks ambiguous items no longer than five seconds before falling back to
isolated inference:

```text
--batch-decode-concurrency 2
--batch-fallback-repack-short
```

The endpoint and multipart contract do not change. A repacked item is accepted
only when every returned segment maps unambiguously to one original item. Any
empty or cross-item result continues through the previous isolated fallback.
The automatically detected first-pass language is retained as the repack and
isolated-fallback language.

The endpoint and response shape are identical whether an item is handled in a
reel, a short fallback reel, or isolated inference. These are internal
execution details and require no client-side branching.

## Verified benchmark

The benchmark used the last retained Muesli recording on the development Mac,
not a synthetic fixture:

- meeting duration: 1028.58 s;
- two source timelines: 2057.16 s aggregate;
- VAD speech: 765.23 s in 129 items;
- upload body: 2.673 MB;
- local Muesli/WhisperKit final transcription: 163.687 s;
- public Homan-native run: 10.895 s preparation + 72.599 s HTTPS
  request = 83.503 s end-to-end;
- server wall inside that public request: 72.144 s;
- measured upload/TLS/proxy overhead beyond server wall: 0.455 s.

This gives:

| Pipeline | Meeting-wall rate | Two-source aggregate rate |
|---|---:|---:|
| Local Muesli | 6.284× realtime | 12.568× realtime |
| Public Homan-native | 12.318× realtime | 24.636× realtime |

The public remote path was **1.960× faster end-to-end** than the saved local
run. The response was also passed through Homan's real `TranscriptReconciler`
and `TranscriptFormatter`, producing 94 attributed turns.

A repeated warm public-path verification measured 89.367 s end to end and
remained **1.832× faster** than the local baseline. Server wall varied by less
than 2.1% between the two public runs. Linear scaling of the measured workload
gives approximately five minutes for a one-hour meeting with similar VAD
density; this remains an extrapolation until a real one-hour recording is run.

Internal concurrency measurements on exactly the same multipart were:

| Internal concurrency | Server wall |
|---:|---:|
| 1 | 84.112 s |
| 2 | 72.323 s |
| 3 | 72.498 s |

Concurrency 2 is therefore the server cap. Runs with concurrency 1, 2, and 3
returned identical text for all 129 items.

### Accuracy interpretation

WhisperKit and whisper.cpp are different inference implementations. On the
same prepared Homan items their normalized text differed by roughly 17–25%,
while two warm WhisperKit repeats differed by 1.33%. A requirement such as
"remote text must be within 5% of local WhisperKit" therefore cannot be used as
a packing regression gate: even isolated whisper.cpp does not satisfy it.
The benchmark is a performance and regression check, not an accuracy claim. A
labelled multilingual corpus is required for a meaningful WER comparison.

## Deployment

The model and `.env` are deployment-local and are not part of the repository.
Set `WHISPER_IMAGE` in `.env` when a deployment uses a prebuilt image; otherwise
Compose builds the service from this source tree.

```bash
docker compose build whisper
docker compose up -d
docker compose ps
```

## License

The patched server is derived from `whisper.cpp` and distributed under its MIT
License. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
