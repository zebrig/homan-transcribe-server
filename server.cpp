#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "httplib.h"
#include "json.hpp"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <csignal>
#include <atomic>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_set>
#if defined (_WIN32)
#include <windows.h>
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

namespace {

// output formats
const std::string json_format   = "json";
const std::string text_format   = "text";
const std::string srt_format    = "srt";
const std::string vjson_format  = "verbose_json";
const std::string vtt_format    = "vtt";

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

struct server_params
{
    std::string hostname       = "127.0.0.1";
    std::string public_path    = "examples/server/public";
    std::string request_path   = "";
    std::string inference_path = "/inference";
    std::string homan_inference_path = "/v1/homan/audio/transcriptions";
    std::string tmp_dir        = ".";

    int32_t port          = 8080;
    int32_t read_timeout  = 3600;
    int32_t write_timeout = 3600;
    int32_t batch_max_concurrency = 1;
    int32_t batch_decode_concurrency = 1;
    int32_t batch_max_items       = 256;

    bool ffmpeg_converter = false;
    bool batch_fallback_repack_short = false;
};

struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = 2;
    int32_t beam_size     = -1;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float temperature     =  0.00f;
    float temperature_inc =  0.20f;
    float no_speech_thold =  0.6f;

    bool debug_mode                = false;
    bool translate                 = false;
    bool detect_language           = false;
    bool diarize                   = false;
    bool tinydiarize               = false;
    bool split_on_word             = false;
    bool no_fallback               = false;
    bool print_special             = false;
    bool print_colors              = false;
    bool print_realtime            = false;
    bool print_progress            = false;
    bool no_timestamps             = false;
    bool token_timestamps          = false;
    bool use_gpu                   = true;
    bool flash_attn                = true;
    int32_t gpu_device             = 0;
    bool suppress_nst              = false;
    bool no_context                = true;
    bool no_language_probabilities = false;
    bool carry_initial_prompt      = false;

    std::string language               = "en";
    std::string prompt                 = "";
    std::string font_path              = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model                  = "models/ggml-base.en.bin";
    std::string response_format        = json_format;
    std::string tdrz_speaker_turn      = " [SPEAKER_TURN]"; // TODO: set from command line
    std::string openvino_encode_device = "CPU";
    std::string dtw                    = "";

    // Voice Activity Detection (VAD) parameters
    bool        vad                         = false;
    std::string vad_model                   = "";
    float       vad_threshold               = 0.5f;
    int         vad_min_speech_duration_ms  = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s   = FLT_MAX;
    int         vad_speech_pad_ms           = 30;
    float       vad_samples_overlap         = 0.1f;
};

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params, const server_params& sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] \n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help                      [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N                 [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N              [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N                [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N                [%-7d] segment index offset\n",                           params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N                [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N             [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N                 [%-7d] maximum segment length in characters\n",           params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word             [%-7s] split on word rather than on token\n",             params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N                 [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N               [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N               [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N              [%-7.2f] word timestamp probability threshold\n",         params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N           [%-7.2f] entropy threshold for decoder fail\n",           params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N           [%-7.2f] log probability threshold for decoder fail\n",   params.logprob_thold);
    fprintf(stderr, "  -debug,    --debug-mode                [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate                 [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize                   [%-7s] stereo audio diarization\n",                       params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize               [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback               [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special             [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors              [%-7s] print colors\n",                                   params.print_colors ? "true" : "false");
    fprintf(stderr, "  -pr,       --print-realtime            [%-7s] print output in realtime\n",                       params.print_realtime ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress            [%-7s] print progress\n",                                 params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps             [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG             [%-7s] spoken language ('auto' for auto-detect)\n",       params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language           [%-7s] exit after automatically detecting language\n",    params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT             [%-7s] initial prompt\n",                                 params.prompt.c_str());
    fprintf(stderr, "             --carry-initial-prompt      [%-7s] always prepend initial prompt\n",                  params.carry_initial_prompt ? "true" : "false");
    fprintf(stderr, "  -m FNAME,  --model FNAME               [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME         [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    // server params
    fprintf(stderr, "  -dtw MODEL --dtw MODEL                 [%-7s] compute token-level timestamps\n",                          params.dtw.c_str());
    fprintf(stderr, "  --host HOST,                           [%-7s] Hostname/ip-adress for the server\n",                       sparams.hostname.c_str());
    fprintf(stderr, "  --port PORT,                           [%-7d] Port number for the server\n",                              sparams.port);
    fprintf(stderr, "  --public PATH,                         [%-7s] Path to the public folder\n",                               sparams.public_path.c_str());
    fprintf(stderr, "  --request-path PATH,                   [%-7s] Request path for all requests\n",                           sparams.request_path.c_str());
    fprintf(stderr, "  --inference-path PATH,                 [%-7s] Inference path for all requests\n",                         sparams.inference_path.c_str());
    fprintf(stderr, "  --homan-inference-path PATH,           [%-7s] Homan-native batch inference path\n",                       sparams.homan_inference_path.c_str());
    fprintf(stderr, "  --batch-max-concurrency N,             [%-7d] maximum parallel states within one Homan batch\n",             sparams.batch_max_concurrency);
    fprintf(stderr, "  --batch-decode-concurrency N,          [%-7d] maximum parallel audio decoders within one Homan batch\n",         sparams.batch_decode_concurrency);
    fprintf(stderr, "  --batch-max-items N,                   [%-7d] maximum audio items in one Homan batch\n",                      sparams.batch_max_items);
    fprintf(stderr, "  --batch-fallback-repack-short,         [%-7s] repack short ambiguous Homan items before isolated fallback\n",     sparams.batch_fallback_repack_short ? "true" : "false");
    fprintf(stderr, "  --convert,                             [%-7s] Convert audio to WAV, requires ffmpeg on the server\n",     sparams.ffmpeg_converter ? "true" : "false");
    fprintf(stderr, "  --tmp-dir,                             [%-7s] Temporary directory for ffmpeg transcoded files\n",         sparams.tmp_dir.c_str());
    fprintf(stderr, "  -sns,      --suppress-nst              [%-7s] suppress non-speech tokens\n",                              params.suppress_nst ? "true" : "false");
    fprintf(stderr, "  -nth N,    --no-speech-thold N         [%-7.2f] no speech threshold\n",                                   params.no_speech_thold);
    fprintf(stderr, "  -ng,       --no-gpu                    [%-7s] do not use gpu\n",                                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev N,    --device N                  [%-7d] GPU device ID (default: 0)\n",                              params.gpu_device);
    fprintf(stderr, "  -fa,       --flash-attn                [%-7s] enable flash attention\n",                                  params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,      --no-flash-attn             [%-7s] disable flash attention\n",                                 params.flash_attn ? "false" : "true");
    fprintf(stderr, "  -nlp,      --no-language-probabilities [%-7s] exclude language probabilities from verbose_json output\n", params.no_language_probabilities ? "true" : "false");
    // Voice Activity Detection (VAD) parameters
    fprintf(stderr, "\nVoice Activity Detection (VAD) options:\n");
    fprintf(stderr, "             --vad                           [%-7s] enable Voice Activity Detection (VAD)\n",            params.vad ? "true" : "false");
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME               [%-7s] VAD model path\n",                                   params.vad_model.c_str());
    fprintf(stderr, "  -vt N,     --vad-threshold N               [%-7.2f] VAD threshold for speech recognition\n",           params.vad_threshold);
    fprintf(stderr, "  -vspd N,   --vad-min-speech-duration-ms  N [%-7d] VAD min speech duration (0.0-1.0)\n",                params.vad_min_speech_duration_ms);
    fprintf(stderr, "  -vsd N,    --vad-min-silence-duration-ms N [%-7d] VAD min silence duration (to split segments)\n",     params.vad_min_silence_duration_ms);
    fprintf(stderr, "  -vmsd N,   --vad-max-speech-duration-s   N [%-7s] VAD max speech duration (auto-split longer)\n",      params.vad_max_speech_duration_s == FLT_MAX ? std::string("FLT_MAX").c_str() : std::to_string(params.vad_max_speech_duration_s).c_str());
    fprintf(stderr, "  -vp N,     --vad-speech-pad-ms           N [%-7d] VAD speech padding (extend segments)\n",             params.vad_speech_pad_ms);
    fprintf(stderr, "  -vo N,     --vad-samples-overlap         N [%-7.2f] VAD samples overlap (seconds between segments)\n", params.vad_samples_overlap);
    fprintf(stderr, "\n");
}

bool whisper_params_parse(int argc, char ** argv, whisper_params & params, server_params & sparams) {
    if (const char * env_device = std::getenv("WHISPER_ARG_DEVICE")) {
        params.gpu_device = std::stoi(env_device);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params, sparams);
            exit(0);
        }
        else if (arg == "-t"     || arg == "--threads")                   { params.n_threads                 = std::stoi(argv[++i]); }
        else if (arg == "-p"     || arg == "--processors")                { params.n_processors              = std::stoi(argv[++i]); }
        else if (arg == "-ot"    || arg == "--offset-t")                  { params.offset_t_ms               = std::stoi(argv[++i]); }
        else if (arg == "-on"    || arg == "--offset-n")                  { params.offset_n                  = std::stoi(argv[++i]); }
        else if (arg == "-d"     || arg == "--duration")                  { params.duration_ms               = std::stoi(argv[++i]); }
        else if (arg == "-mc"    || arg == "--max-context")               { params.max_context               = std::stoi(argv[++i]); }
        else if (arg == "-ml"    || arg == "--max-len")                   { params.max_len                   = std::stoi(argv[++i]); }
        else if (arg == "-bo"    || arg == "--best-of")                   { params.best_of                   = std::stoi(argv[++i]); }
        else if (arg == "-bs"    || arg == "--beam-size")                 { params.beam_size                 = std::stoi(argv[++i]); }
        else if (arg == "-ac"    || arg == "--audio-ctx")                 { params.audio_ctx                 = std::stoi(argv[++i]); }
        else if (arg == "-wt"    || arg == "--word-thold")                { params.word_thold                = std::stof(argv[++i]); }
        else if (arg == "-et"    || arg == "--entropy-thold")             { params.entropy_thold             = std::stof(argv[++i]); }
        else if (arg == "-lpt"   || arg == "--logprob-thold")             { params.logprob_thold             = std::stof(argv[++i]); }
        else if (arg == "-debug" || arg == "--debug-mode")                { params.debug_mode                = true; }
        else if (arg == "-tr"    || arg == "--translate")                 { params.translate                 = true; }
        else if (arg == "-di"    || arg == "--diarize")                   { params.diarize                   = true; }
        else if (arg == "-tdrz"  || arg == "--tinydiarize")               { params.tinydiarize               = true; }
        else if (arg == "-sow"   || arg == "--split-on-word")             { params.split_on_word             = true; }
        else if (arg == "-nf"    || arg == "--no-fallback")               { params.no_fallback               = true; }
        else if (arg == "-fp"    || arg == "--font-path")                 { params.font_path                 = argv[++i]; }
        else if (arg == "-ps"    || arg == "--print-special")             { params.print_special             = true; }
        else if (arg == "-pc"    || arg == "--print-colors")              { params.print_colors              = true; }
        else if (arg == "-pr"    || arg == "--print-realtime")            { params.print_realtime            = true; }
        else if (arg == "-pp"    || arg == "--print-progress")            { params.print_progress            = true; }
        else if (arg == "-nt"    || arg == "--no-timestamps")             { params.no_timestamps             = true; }
        else if (arg == "-l"     || arg == "--language")                  { params.language                  = argv[++i]; }
        else if (arg == "-dl"    || arg == "--detect-language")           { params.detect_language           = true; }
        else if (                   arg == "--prompt")                    { params.prompt                    = argv[++i]; }
        else if (                   arg == "--carry-initial-prompt")      { params.carry_initial_prompt      = true; }
        else if (arg == "-m"     || arg == "--model")                     { params.model                     = argv[++i]; }
        else if (arg == "-oved"  || arg == "--ov-e-device")               { params.openvino_encode_device    = argv[++i]; }
        else if (arg == "-dtw"   || arg == "--dtw")                       { params.dtw                       = argv[++i]; }
        else if (arg == "-ng"    || arg == "--no-gpu")                    { params.use_gpu                   = false; }
        else if (arg == "-dev"   || arg == "--device")                    { params.gpu_device                = std::stoi(argv[++i]); }
        else if (arg == "-fa"    || arg == "--flash-attn")                { params.flash_attn                = true; }
        else if (arg == "-nfa"   || arg == "--no-flash-attn")             { params.flash_attn                = false; }
        else if (arg == "-sns"   || arg == "--suppress-nst")              { params.suppress_nst              = true; }
        else if (arg == "-nth"   || arg == "--no-speech-thold")           { params.no_speech_thold           = std::stof(argv[++i]); }
        else if (arg == "-nlp"   || arg == "--no-language-probabilities") { params.no_language_probabilities = true; }

        // server params
        else if (                   arg == "--port")            { sparams.port        = std::stoi(argv[++i]); }
        else if (                   arg == "--host")            { sparams.hostname    = argv[++i]; }
        else if (                   arg == "--public")          { sparams.public_path = argv[++i]; }
        else if (                   arg == "--request-path")    { sparams.request_path = argv[++i]; }
        else if (                   arg == "--inference-path")  { sparams.inference_path = argv[++i]; }
        else if (                   arg == "--homan-inference-path") { sparams.homan_inference_path = argv[++i]; }
        else if (                   arg == "--batch-max-concurrency") { sparams.batch_max_concurrency = std::stoi(argv[++i]); }
        else if (                   arg == "--batch-decode-concurrency") { sparams.batch_decode_concurrency = std::stoi(argv[++i]); }
        else if (                   arg == "--batch-max-items") { sparams.batch_max_items = std::stoi(argv[++i]); }
        else if (                   arg == "--batch-fallback-repack-short") { sparams.batch_fallback_repack_short = true; }
        else if (                   arg == "--convert")         { sparams.ffmpeg_converter     = true; }
        else if (                   arg == "--tmp-dir")         { sparams.tmp_dir     = argv[++i]; }

        // Voice Activity Detection (VAD)
        else if (                   arg == "--vad")                         { params.vad                         = true; }
        else if (arg == "-vm"    || arg == "--vad-model")                   { params.vad_model                   = argv[++i]; }
        else if (arg == "-vt"    || arg == "--vad-threshold")               { params.vad_threshold               = std::stof(argv[++i]); }
        else if (arg == "-vspd"  || arg == "--vad-min-speech-duration-ms")  { params.vad_min_speech_duration_ms  = std::stoi(argv[++i]); }
        else if (arg == "-vsd"   || arg == "--vad-min-silence-duration-ms") { params.vad_min_silence_duration_ms = std::stoi(argv[++i]); }
        else if (arg == "-vmsd"  || arg == "--vad-max-speech-duration-s")   { params.vad_max_speech_duration_s   = std::stof(argv[++i]); }
        else if (arg == "-vp"    || arg == "--vad-speech-pad-ms")           { params.vad_speech_pad_ms           = std::stoi(argv[++i]); }
        else if (arg == "-vo"    || arg == "--vad-samples-overlap")         { params.vad_samples_overlap         = std::stof(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params, sparams);
            exit(0);
        }
    }

    return true;
}

struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

void check_ffmpeg_availibility() {
    int result = system("ffmpeg -version");

    if (result == 0) {
        std::cout << "ffmpeg is available." << std::endl;
    } else {
        // ffmpeg is not available
        std::cout << "ffmpeg is not found. Please ensure that ffmpeg is installed ";
        std::cout << "and that its executable is included in your system's PATH. ";
        exit(0);
    }
}

std::string generate_temp_filename(const std::string &path, const std::string &prefix, const std::string &extension) {
    static std::atomic<uint64_t> next_temp_id{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();

    std::stringstream ss;
    ss << path
       << std::filesystem::path::preferred_separator
       << prefix
       << "-" << now
       << "-" << next_temp_id.fetch_add(1, std::memory_order_relaxed)
       << extension;

    return ss.str();
}

bool convert_to_wav(const std::string & temp_filename, std::string & error_resp, bool stereo) {
    std::ostringstream cmd_stream;
    std::string converted_filename_temp = temp_filename + "_temp.wav";
    cmd_stream << "ffmpeg -i \"" << temp_filename << "\" -y -ar 16000 -ac " << (stereo ? 2 : 1) << " -c:a pcm_s16le \"" << converted_filename_temp << "\" 2>&1";
    std::string cmd = cmd_stream.str();

    int status = std::system(cmd.c_str());
    if (status != 0) {
        error_resp = "{\"error\":\"FFmpeg conversion failed.\"}";
        return false;
    }

    // Remove the original file
    if (remove(temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to remove the original file.\"}";
        return false;
    }

    // Rename the temporary file to match the original filename
    if (rename(converted_filename_temp.c_str(), temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to rename the temporary file.\"}";
        return false;
    }
    return true;
}

std::string estimate_diarization_speaker(const std::vector<std::vector<float>> & pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
    int progress_step = ((whisper_print_user_data *) user_data)->params->progress_step;
    int * progress_prev  = &(((whisper_print_user_data *) user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step) {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        if (params.print_colors) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p(ctx, i, j);

                const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        } else {
            const char * text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }
        fflush(stdout);
    }
}

std::string output_str(struct whisper_context * ctx, const whisper_params & params, const std::vector<std::vector<float>> & pcmf32s) {
    std::stringstream result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        std::string speaker = "";

        if (params.diarize && pcmf32s.size() == 2)
        {
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        result << speaker << text << "\n";
    }
    return result.str();
}

void get_req_parameters(const Request & req, whisper_params & params)
{
    // This route is intentionally only conditionally OpenAI-compatible. Keep
    // the verified decoder profile server-owned; in particular, clients must
    // not be able to change best_of, beam size, token timestamps, VAD,
    // diarization, translation, or GPU/CPU behavior. OpenAI fields that are
    // meaningful for this single-model Whisper deployment are mapped below.
    // model and every unknown/unsupported multipart field are ignored.
    if (req.has_file("language"))
    {
        const std::string language = req.get_file_value("language").content;
        if (language == "auto" || whisper_lang_id(language.c_str()) >= 0) {
            params.language = language;
        }
    }
    if (req.has_file("prompt"))
    {
        params.prompt = req.get_file_value("prompt").content.substr(0, 4096);
    }
    if (req.has_file("response_format"))
    {
        const std::string format = req.get_file_value("response_format").content;
        if (format == json_format || format == text_format ||
                format == srt_format || format == vjson_format ||
                format == vtt_format) {
            params.response_format = format;
        }
    }
    if (req.has_file("temperature"))
    {
        try {
            const float temperature = std::stof(
                req.get_file_value("temperature").content);
            if (std::isfinite(temperature) && temperature >= 0.0f &&
                    temperature <= 1.0f) {
                params.temperature = temperature;
            }
        } catch (const std::exception &) {
            // Invalid optional compatibility fields are ignored as well.
        }
    }
}

struct homan_batch_item {
    std::string id;
    std::string source;
    std::string file_field;
    std::string filename;
    std::string encoded_audio;
    std::string packing_language;
    double start = 0.0;
    double end   = 0.0;
    std::vector<float> pcmf32;
};

struct homan_batch_reel_placement {
    size_t item_index  = 0;
    size_t start_sample = 0;
    size_t end_sample   = 0;
};

struct homan_batch_reel {
    std::string source;
    std::string packing_language;
    std::vector<float> pcmf32;
    std::vector<homan_batch_reel_placement> placements;
};

struct homan_batch_reel_result {
    double processing_ms = 0.0;
    int cross_item_segments = 0;
    std::vector<size_t> fallback_item_indices;
};

constexpr double homan_reel_max_seconds = 29.5;
constexpr double homan_reel_separator_seconds = 1.20;
constexpr double homan_fallback_repack_max_item_seconds = 5.0;
constexpr double homan_fallback_repack_separator_seconds = 2.40;

std::string trim_transcript(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool decode_homan_audio(
        const std::string & content,
        const std::string & filename,
        const server_params & sparams,
        std::vector<float> & pcmf32,
        std::string & error_message) {
    std::vector<std::vector<float>> pcmf32s;

    if (!sparams.ffmpeg_converter) {
        if (!::read_audio_data(content.data(), content.size(), pcmf32, pcmf32s, false)) {
            error_message = "could not decode audio item '" + filename + "'";
            return false;
        }
        return true;
    }

    const std::string temp_filename = generate_temp_filename(
        sparams.tmp_dir, "homan-batch", ".audio");
    {
        std::ofstream temp_file{temp_filename, std::ios::binary};
        if (!temp_file) {
            error_message = "could not create a temporary audio file";
            return false;
        }
        temp_file << content;
    }

    std::string converter_error;
    const bool converted = convert_to_wav(temp_filename, converter_error, false);
    if (!converted) {
        std::remove(temp_filename.c_str());
        std::remove((temp_filename + "_temp.wav").c_str());
        error_message = "ffmpeg could not decode audio item '" + filename + "'";
        return false;
    }

    const bool read = ::read_audio_data(
        temp_filename, pcmf32, pcmf32s, false);
    std::remove(temp_filename.c_str());
    if (!read) {
        error_message = "could not read decoded audio item '" + filename + "'";
        return false;
    }
    return true;
}

whisper_full_params make_homan_batch_params(const whisper_params & params) {
    whisper_full_params wparams = whisper_full_default_params(
        WHISPER_SAMPLING_GREEDY);

    wparams.strategy = params.beam_size > 1
        ? WHISPER_SAMPLING_BEAM_SEARCH
        : WHISPER_SAMPLING_GREEDY;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.language         = params.language.c_str();
    wparams.detect_language  = false;
    wparams.n_threads        = params.n_threads;
    wparams.n_max_text_ctx   = params.max_context >= 0
        ? params.max_context
        : wparams.n_max_text_ctx;
    wparams.offset_ms        = 0;
    wparams.duration_ms      = 0;
    wparams.thold_pt         = params.word_thold;
    wparams.max_len          = 0;
    wparams.split_on_word    = false;
    wparams.audio_ctx        = params.audio_ctx;
    wparams.debug_mode       = false;
    wparams.greedy.best_of        = params.best_of;
    wparams.beam_search.beam_size = params.beam_size;
    wparams.temperature      = params.temperature;
    wparams.no_speech_thold  = params.no_speech_thold;
    wparams.temperature_inc  = params.temperature_inc;
    wparams.entropy_thold    = params.entropy_thold;
    wparams.logprob_thold    = params.logprob_thold;
    wparams.no_timestamps    = false;
    wparams.token_timestamps = false;
    wparams.no_context       = true;
    wparams.suppress_nst     = params.suppress_nst;
    wparams.vad              = false;
    wparams.abort_callback   = [](void *) { return false; };
    wparams.abort_callback_user_data = nullptr;
    return wparams;
}

json transcribe_homan_batch_item(
        struct whisper_context * ctx,
        struct whisper_state * state,
        const whisper_params & params,
        const homan_batch_item & item,
        std::string & error_message) {
    whisper_full_params wparams = make_homan_batch_params(params);
    const auto started = std::chrono::steady_clock::now();
    const int result = whisper_full_with_state(
        ctx,
        state,
        wparams,
        item.pcmf32.data(),
        static_cast<int>(item.pcmf32.size()));
    const auto finished = std::chrono::steady_clock::now();
    const double processing_ms = std::chrono::duration<double, std::milli>(
        finished - started).count();

    if (result != 0) {
        error_message = "whisper inference failed for item '" + item.id + "'";
        return json::object();
    }

    json segments = json::array();
    std::string text;
    const int segment_count = whisper_full_n_segments_from_state(state);
    for (int segment_index = 0; segment_index < segment_count; ++segment_index) {
        const char * segment_text = whisper_full_get_segment_text_from_state(
            state, segment_index);
        text += segment_text == nullptr ? "" : segment_text;
        segments.push_back(json{
            {"id", segment_index},
            {"start", whisper_full_get_segment_t0_from_state(state, segment_index) * 0.01},
            {"end", whisper_full_get_segment_t1_from_state(state, segment_index) * 0.01},
            {"text", segment_text == nullptr ? "" : segment_text},
        });
    }

    const int language_id = whisper_full_lang_id_from_state(state);
    return json{
        {"id", item.id},
        {"source", item.source},
        {"start", item.start},
        {"end", item.end},
        {"text", trim_transcript(text)},
        {"language", whisper_lang_str(language_id)},
        {"language_name", whisper_lang_str_full(language_id)},
        {"audio_duration", static_cast<double>(item.pcmf32.size()) / WHISPER_SAMPLE_RATE},
        {"processing_ms", processing_ms},
        {"segments", std::move(segments)},
    };
}

std::vector<homan_batch_reel> pack_homan_batch_reels(
        const std::vector<homan_batch_item> & items,
        double separator_seconds = homan_reel_separator_seconds) {
    struct source_group {
        std::string key;
        std::string source;
        std::string packing_language;
        std::vector<size_t> item_indices;
    };

    std::vector<source_group> groups;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const std::string group_key = items[item_index].source + "\x1f" +
            items[item_index].packing_language;
        auto group = std::find_if(
            groups.begin(), groups.end(), [&](const source_group & candidate) {
                return candidate.key == group_key;
            });
        if (group == groups.end()) {
            groups.push_back(source_group{
                group_key,
                items[item_index].source,
                items[item_index].packing_language,
                {},
            });
            group = std::prev(groups.end());
        }
        group->item_indices.push_back(item_index);
    }

    const size_t max_reel_samples = static_cast<size_t>(
        std::llround(homan_reel_max_seconds * WHISPER_SAMPLE_RATE));
    const size_t separator_samples = static_cast<size_t>(
        std::llround(separator_seconds * WHISPER_SAMPLE_RATE));
    std::vector<homan_batch_reel> reels;

    for (auto & group : groups) {
        std::stable_sort(
            group.item_indices.begin(), group.item_indices.end(),
            [&](size_t left, size_t right) {
                if (items[left].start == items[right].start) {
                    return left < right;
                }
                return items[left].start < items[right].start;
            });

        homan_batch_reel reel;
        reel.source = group.source;
        reel.packing_language = group.packing_language;
        for (const size_t item_index : group.item_indices) {
            const auto & item = items[item_index];
            const size_t separator = reel.placements.empty() ? 0 : separator_samples;
            if (!reel.placements.empty() &&
                    reel.pcmf32.size() + separator + item.pcmf32.size() > max_reel_samples) {
                reels.push_back(std::move(reel));
                reel = homan_batch_reel{};
                reel.source = group.source;
                reel.packing_language = group.packing_language;
            }

            if (!reel.placements.empty()) {
                reel.pcmf32.insert(reel.pcmf32.end(), separator_samples, 0.0f);
            }
            const size_t start_sample = reel.pcmf32.size();
            reel.pcmf32.insert(
                reel.pcmf32.end(), item.pcmf32.begin(), item.pcmf32.end());
            reel.placements.push_back(homan_batch_reel_placement{
                item_index,
                start_sample,
                reel.pcmf32.size(),
            });
        }
        if (!reel.placements.empty()) {
            reels.push_back(std::move(reel));
        }
    }
    return reels;
}

json make_homan_batch_item_result(const homan_batch_item & item) {
    return json{
        {"id", item.id},
        {"source", item.source},
        {"start", item.start},
        {"end", item.end},
        {"text", ""},
        {"language", ""},
        {"language_name", ""},
        {"audio_duration", static_cast<double>(item.pcmf32.size()) / WHISPER_SAMPLE_RATE},
        {"processing_ms", 0.0},
        {"segments", json::array()},
    };
}

bool transcribe_homan_batch_reel(
        struct whisper_context * ctx,
        struct whisper_state * state,
        const whisper_params & params,
        const homan_batch_reel & reel,
        const std::vector<homan_batch_item> & items,
        std::vector<json> & item_results,
        homan_batch_reel_result & reel_result,
        std::string & error_message) {
    whisper_params reel_params = params;
    if (!reel.packing_language.empty()) {
        reel_params.language = reel.packing_language;
    }
    whisper_full_params wparams = make_homan_batch_params(reel_params);
    const auto started = std::chrono::steady_clock::now();
    const int result = whisper_full_with_state(
        ctx,
        state,
        wparams,
        reel.pcmf32.data(),
        static_cast<int>(reel.pcmf32.size()));
    const auto finished = std::chrono::steady_clock::now();
    reel_result.processing_ms = std::chrono::duration<double, std::milli>(
        finished - started).count();

    if (result != 0) {
        error_message = "whisper inference failed for a packed reel";
        return false;
    }

    const int language_id = whisper_full_lang_id_from_state(state);
    size_t speech_samples = 0;
    for (const auto & placement : reel.placements) {
        speech_samples += placement.end_sample - placement.start_sample;
    }
    for (const auto & placement : reel.placements) {
        auto & item_result = item_results[placement.item_index];
        item_result["language"] = whisper_lang_str(language_id);
        item_result["language_name"] = whisper_lang_str_full(language_id);
        const size_t item_samples = placement.end_sample - placement.start_sample;
        item_result["processing_ms"] = speech_samples == 0
            ? 0.0
            : reel_result.processing_ms * static_cast<double>(item_samples) /
                static_cast<double>(speech_samples);
    }

    const int segment_count = whisper_full_n_segments_from_state(state);
    for (int segment_index = 0; segment_index < segment_count; ++segment_index) {
        const char * raw_segment_text = whisper_full_get_segment_text_from_state(
            state, segment_index);
        const std::string segment_text = raw_segment_text == nullptr
            ? ""
            : raw_segment_text;
        if (trim_transcript(segment_text).empty()) {
            continue;
        }

        const int64_t segment_start_sample = static_cast<int64_t>(std::llround(
            whisper_full_get_segment_t0_from_state(state, segment_index) *
            0.01 * WHISPER_SAMPLE_RATE));
        const int64_t segment_end_sample = std::max(
            segment_start_sample + 1,
            static_cast<int64_t>(std::llround(
                whisper_full_get_segment_t1_from_state(state, segment_index) *
                0.01 * WHISPER_SAMPLE_RATE)));

        size_t best_placement_index = 0;
        int64_t best_overlap = -1;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        std::vector<size_t> overlapping_placement_indices;
        constexpr int64_t significant_overlap_samples =
            WHISPER_SAMPLE_RATE / 20;
        for (size_t placement_index = 0;
                placement_index < reel.placements.size();
                ++placement_index) {
            const auto & placement = reel.placements[placement_index];
            const int64_t placement_start = static_cast<int64_t>(placement.start_sample);
            const int64_t placement_end = static_cast<int64_t>(placement.end_sample);
            const int64_t overlap = std::max<int64_t>(
                0,
                std::min(segment_end_sample, placement_end) -
                    std::max(segment_start_sample, placement_start));
            if (overlap >= significant_overlap_samples) {
                overlapping_placement_indices.push_back(placement_index);
            }
            const int64_t distance = segment_end_sample < placement_start
                ? placement_start - segment_end_sample
                : (segment_start_sample > placement_end
                    ? segment_start_sample - placement_end
                    : 0);
            if (overlap > best_overlap ||
                    (overlap == best_overlap && distance < best_distance)) {
                best_overlap = overlap;
                best_distance = distance;
                best_placement_index = placement_index;
            }
        }
        if (overlapping_placement_indices.size() > 1) {
            ++reel_result.cross_item_segments;
            for (const size_t placement_index : overlapping_placement_indices) {
                const size_t item_index =
                    reel.placements[placement_index].item_index;
                if (std::find(
                        reel_result.fallback_item_indices.begin(),
                        reel_result.fallback_item_indices.end(),
                        item_index) == reel_result.fallback_item_indices.end()) {
                    reel_result.fallback_item_indices.push_back(item_index);
                }
            }
            continue;
        }

        const auto & placement = reel.placements[best_placement_index];
        auto & item_result = item_results[placement.item_index];
        std::string item_text = item_result["text"].get<std::string>();
        item_text += segment_text;
        item_result["text"] = trim_transcript(item_text);

        const double item_duration = static_cast<double>(
            placement.end_sample - placement.start_sample) / WHISPER_SAMPLE_RATE;
        const double relative_start = std::clamp(
            static_cast<double>(segment_start_sample -
                static_cast<int64_t>(placement.start_sample)) / WHISPER_SAMPLE_RATE,
            0.0,
            item_duration);
        const double relative_end = std::clamp(
            static_cast<double>(segment_end_sample -
                static_cast<int64_t>(placement.start_sample)) / WHISPER_SAMPLE_RATE,
            relative_start,
            item_duration);
        auto & item_segments = item_result["segments"];
        item_segments.push_back(json{
            {"id", item_segments.size()},
            {"start", relative_start},
            {"end", relative_end},
            {"text", segment_text},
        });
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;
    server_params sparams;

    std::mutex whisper_mutex;

    if (whisper_params_parse(argc, argv, params, sparams) == false) {
        whisper_print_usage(argc, argv, params, sparams);
        return 1;
    }

    if (sparams.batch_max_concurrency < 1 || sparams.batch_max_concurrency > 16) {
        fprintf(stderr, "error: --batch-max-concurrency must be between 1 and 16\n");
        return 1;
    }
    if (sparams.batch_decode_concurrency < 1 || sparams.batch_decode_concurrency > 16) {
        fprintf(stderr, "error: --batch-decode-concurrency must be between 1 and 16\n");
        return 1;
    }
    if (sparams.batch_max_items < 1 || sparams.batch_max_items > 4096) {
        fprintf(stderr, "error: --batch-max-items must be between 1 and 4096\n");
        return 1;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params, sparams);
        exit(0);
    }

    if (params.diarize && params.tinydiarize) {
        fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
        whisper_print_usage(argc, argv, params, sparams);
        exit(0);
    }

    if (sparams.ffmpeg_converter) {
        check_ffmpeg_availibility();
    }
    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.gpu_device = params.gpu_device;
    cparams.flash_attn = params.flash_attn;

    // Stored once and copied per request. Homan-native intentionally ignores
    // client model/decoder tuning and always uses this server-owned profile.
    whisper_params default_params = params;

    if (!params.dtw.empty()) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        if (params.dtw == "tiny") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        }
        if (params.dtw == "tiny.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        }
        if (params.dtw == "base") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        }
        if (params.dtw == "base.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        }
        if (params.dtw == "small") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        }
        if (params.dtw == "small.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        }
        if (params.dtw == "medium") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        }
        if (params.dtw == "medium.en") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        }
        if (params.dtw == "large.v1") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        }
        if (params.dtw == "large.v2") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        }
        if (params.dtw == "large.v3") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        }
        if (params.dtw == "large.v3.turbo") {
            cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3_TURBO;
        }

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());
            return 3;
        }
    }

    std::unique_ptr<httplib::Server> svr = std::make_unique<httplib::Server>();
    // Keep Vulkan context creation and all inference work on one OS thread.
    // Requests are intentionally serialized for this deployment.
    svr->new_task_queue = [] { return new httplib::ThreadPool(1); };
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    // Created lazily by the request worker so the Vulkan context is initialized
    // and used on the same OS thread. The model context then remains resident.
    struct whisper_context * ctx = nullptr;

    auto ensure_context_loaded = [&](Response & res) -> bool {
        if (ctx != nullptr) {
            return true;
        }
        fprintf(stderr, "loading persistent whisper model on inference worker\n");
        ctx = whisper_init_from_file_with_params(
            default_params.model.c_str(), cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "error: failed to initialize whisper context\n");
            res.status = 500;
            res.set_content(
                "{\"error\":{\"message\":\"model init failed\",\"type\":\"server_error\"}}",
                "application/json");
            return false;
        }
        state.store(SERVER_STATE_READY);
        return true;
    };

    svr->set_default_headers({{"Server", "whisper.cpp"},
                             {"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "content-type, authorization"}});

    std::string const default_content = R"(
    <html>
    <head>
        <title>Whisper.cpp Server</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <style>
        body {
            font-family: sans-serif;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        label {
            margin-bottom: 0.5rem;
        }
        input, select {
            margin-bottom: 1rem;
        }
        button {
            margin-top: 1rem;
        }
        </style>
    </head>
    <body>
        <h1>Whisper.cpp Server</h1>

        <h2>)" + sparams.request_path + sparams.inference_path + R"(</h2>
        <pre>
    curl 127.0.0.1:)" + std::to_string(sparams.port) + sparams.request_path + sparams.inference_path + R"( \
    -H "Content-Type: multipart/form-data" \
    -F file="@&lt;file-path&gt;" \
    -F temperature="0.0" \
    -F temperature_inc="0.2" \
    -F no_speech_thold="0.6" \
    -F response_format="json"
        </pre>

        <h2>/load</h2>
        <pre>
    curl 127.0.0.1:)" + std::to_string(sparams.port) + R"(/load \
    -H "Content-Type: multipart/form-data" \
    -F model="&lt;path-to-model-file&gt;"
        </pre>

        <div>
            <h2>Try it out</h2>
            <form action=")" + sparams.request_path + sparams.inference_path + R"(" method="POST" enctype="multipart/form-data">
                <label for="file">Choose an audio file:</label>
                <input type="file" id="file" name="file" accept="audio/*" required><br>

                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" name="temperature" value="0.0" step="0.01" placeholder="e.g., 0.0"><br>

                <label for="response_format">Response Format:</label>
                <select id="response_format" name="response_format">
                    <option value="verbose_json">Verbose JSON</option>
                    <option value="json">JSON</option>
                    <option value="text">Text</option>
                    <option value="srt">SRT</option>
                    <option value="vtt">VTT</option>
                </select><br>

                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    )";

    // this is only called if no index.html is found in the public --path
    svr->Get(sparams.request_path + "/", [&](const Request &, Response &res){
        res.set_content(default_content, "text/html");
        return false;
    });

    svr->Options(sparams.request_path + sparams.inference_path, [&](const Request &, Response &){
    });

    svr->Post(sparams.request_path + sparams.inference_path, [&](const Request &req, Response &res){
        // acquire whisper model mutex lock
        std::lock_guard<std::mutex> lock(whisper_mutex);

        if (!ensure_context_loaded(res)) {
            return;
        }

        // first check user requested fields of the request
        if (!req.has_file("file"))
        {
            fprintf(stderr, "error: no 'file' field in the request\n");
            const std::string error_resp = "{\"error\":\"no 'file' field in the request\"}";
            res.status = 400;
            res.set_content(error_resp, "application/json");
            return;
        }
        auto audio_file = req.get_file_value("file");

        whisper_params params = default_params;
        get_req_parameters(req, params);

        std::string filename{audio_file.filename};
        printf("Received request: %s\n", filename.c_str());

        // audio arrays
        std::vector<float> pcmf32;               // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (sparams.ffmpeg_converter) {
            // if file is not wav, convert to wav
            // write to temporary file
            const std::string temp_filename = generate_temp_filename(sparams.tmp_dir, "whisper-server", ".wav");
            std::ofstream temp_file{temp_filename, std::ios::binary};
            temp_file << audio_file.content;
            temp_file.close();

            std::string error_resp = "{\"error\":\"Failed to execute ffmpeg command.\"}";
            const bool is_converted = convert_to_wav(temp_filename, error_resp, params.diarize);
            if (!is_converted) {
                res.status = 500;
                res.set_content(error_resp, "application/json");
                return;
            }

            // read audio content into pcmf32
            if (!::read_audio_data(temp_filename, pcmf32, pcmf32s, params.diarize))
            {
                fprintf(stderr, "error: failed to read WAV file '%s'\n", temp_filename.c_str());
                const std::string error_resp = "{\"error\":\"failed to read WAV file\"}";
                res.status = 400;
                res.set_content(error_resp, "application/json");
                std::remove(temp_filename.c_str());
                return;
            }
            // remove temp file
            std::remove(temp_filename.c_str());
        } else {
            if (!::read_audio_data(audio_file.content.data(), audio_file.content.size(), pcmf32, pcmf32s, params.diarize)) {
                fprintf(stderr, "error: failed to read audio data\n");
                const std::string error_resp = "{\"error\":\"failed to read audio data\"}";
                res.status = 400;
                res.set_content(error_resp, "application/json");
                return;
            }
        }

        printf("Successfully loaded %s\n", filename.c_str());

        // print system information
        {
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                    params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
        }

        // print some info about the processing
        {
            fprintf(stderr, "\n");
            if (!whisper_is_multilingual(ctx)) {
                if (params.language != "en" || params.translate) {
                    params.language = "en";
                    params.translate = false;
                    fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
                }
            }
            if (params.detect_language) {
                params.language = "auto";
            }
            fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, %stimestamps = %d ...\n",
                    __func__, filename.c_str(), int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                    params.n_threads, params.n_processors,
                    params.language.c_str(),
                    params.translate ? "translate" : "transcribe",
                    params.tinydiarize ? "tdrz = 1, " : "",
                    params.no_timestamps ? 0 : 1);

            fprintf(stderr, "\n");
        }

        // run the inference
        {
            printf("Running whisper.cpp inference on %s\n", filename.c_str());
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

            wparams.print_realtime   = false;
            wparams.print_progress   = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.print_special    = params.print_special;
            wparams.translate        = params.translate;
            wparams.language         = params.language.c_str();
            wparams.detect_language  = params.detect_language;
            wparams.n_threads        = params.n_threads;
            wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = params.offset_t_ms;
            wparams.duration_ms      = params.duration_ms;

            wparams.thold_pt         = params.word_thold;
            wparams.max_len          = params.max_len;
            wparams.split_on_word    = params.split_on_word;
            wparams.audio_ctx        = params.audio_ctx;

            wparams.debug_mode       = params.debug_mode;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            wparams.initial_prompt   = params.prompt.c_str();
            wparams.carry_initial_prompt = params.carry_initial_prompt;

            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.temperature      = params.temperature;
            wparams.no_speech_thold  = params.no_speech_thold;
            wparams.temperature_inc  = params.temperature_inc;
            wparams.entropy_thold    = params.entropy_thold;
            wparams.logprob_thold    = params.logprob_thold;

            wparams.no_timestamps    = params.no_timestamps;
            wparams.token_timestamps = params.token_timestamps;
            wparams.no_context       = params.no_context;

            wparams.suppress_nst     = params.suppress_nst;

            wparams.vad              = params.vad;
            wparams.vad_model_path   = params.vad_model.c_str();

            wparams.vad_params.threshold               = params.vad_threshold;
            wparams.vad_params.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
            wparams.vad_params.min_silence_duration_ms = params.vad_min_silence_duration_ms;
            wparams.vad_params.max_speech_duration_s   = params.vad_max_speech_duration_s;
            wparams.vad_params.speech_pad_ms           = params.vad_speech_pad_ms;
            wparams.vad_params.samples_overlap         = params.vad_samples_overlap;

            whisper_print_user_data user_data = { &params, &pcmf32s, 0 };

            // this callback is called on each new segment
            if (params.print_realtime) {
                wparams.new_segment_callback           = whisper_print_segment_callback;
                wparams.new_segment_callback_user_data = &user_data;
            }

            if (wparams.print_progress) {
                wparams.progress_callback           = whisper_print_progress_callback;
                wparams.progress_callback_user_data = &user_data;
            }

            // Avoid a socket syscall in the hot inference callback. Caddy owns
            // the public request timeout and inference is allowed to complete.
            wparams.abort_callback = [](void *) { return false; };
            wparams.abort_callback_user_data = nullptr;

            whisper_reset_timings(ctx);
            const auto inference_start = std::chrono::steady_clock::now();
            const int inference_result = whisper_full_parallel(
                ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors);
            const auto inference_end = std::chrono::steady_clock::now();
            fprintf(stderr, "request_timings: inference = %.3f ms\n",
                    std::chrono::duration<double, std::milli>(inference_end - inference_start).count());
            whisper_print_timings(ctx);

            if (inference_result != 0) {
                // handle failure or early abort
                if (req.is_connection_closed()) {
                    // log client disconnect
                    fprintf(stderr, "client disconnected, aborted processing\n");
                    res.status = 499; // Client Closed Request (nginx convention)
                    res.set_content("{\"error\":\"client disconnected\"}", "application/json");
                    return;
                }
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                res.status = 500; // Internal Server Error
                const std::string error_resp = "{\"error\":\"failed to process audio\"}";
                res.set_content(error_resp, "application/json");
                return;
            }
        }

        // return results to user
        if (params.response_format == text_format)
        {
            std::string results = output_str(ctx, params, pcmf32s);
            res.set_content(results.c_str(), "text/plain; charset=utf-8");
        }
        else if (params.response_format == srt_format)
        {
            std::stringstream ss;
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                std::string speaker = "";

                if (params.diarize && pcmf32s.size() == 2)
                {
                    speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
                }

                ss << i + 1 + params.offset_n << "\n";
                ss << to_timestamp(t0, true) << " --> " << to_timestamp(t1, true) << "\n";
                ss << speaker << text << "\n\n";
            }
            res.set_content(ss.str(), "application/x-subrip");
        } else if (params.response_format == vtt_format) {
            std::stringstream ss;

            ss << "WEBVTT\n\n";

            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                std::string speaker = "";

                if (params.diarize && pcmf32s.size() == 2)
                {
                    speaker = estimate_diarization_speaker(pcmf32s, t0, t1, true);
                    speaker.insert(0, "<v Speaker");
                    speaker.append(">");
                }

                ss << to_timestamp(t0) << " --> " << to_timestamp(t1) << "\n";
                ss << speaker << text << "\n\n";
            }
            res.set_content(ss.str(), "text/vtt");
        } else if (params.response_format == vjson_format) {
            /* try to match openai/whisper's Python format */
            std::string results = output_str(ctx, params, pcmf32s);
            json jres = json{
                {"task", params.translate ? "translate" : "transcribe"},
                {"language", whisper_lang_str_full(whisper_full_lang_id(ctx))},
                {"duration", float(pcmf32.size())/WHISPER_SAMPLE_RATE},
                {"text", results},
                {"segments", json::array()}
            };
            // Only compute language probabilities if requested (expensive operation)
            if (!params.no_language_probabilities) {
                std::vector<float> lang_probs(whisper_lang_max_id() + 1, 0.0f);
                const auto detected_lang_id = whisper_lang_auto_detect(ctx, 0, params.n_threads, lang_probs.data());
                jres["detected_language"] = whisper_lang_str_full(detected_lang_id);
                jres["detected_language_probability"] = lang_probs[detected_lang_id];
                jres["language_probabilities"] = json::object();
                // Add all language probabilities
                for (int i = 0; i <= whisper_lang_max_id(); ++i) {
                    if (lang_probs[i] > 0.001f) { // Only include non-negligible probabilities
                        jres["language_probabilities"][whisper_lang_str(i)] = lang_probs[i];
                    }
                }
            }
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i)
            {
                json segment = json{
                    {"id", i},
                    {"text", whisper_full_get_segment_text(ctx, i)},
                };

                if (!params.no_timestamps) {
                    segment["start"] = whisper_full_get_segment_t0(ctx, i) * 0.01;
                    segment["end"] = whisper_full_get_segment_t1(ctx, i) * 0.01;
                }

                if (params.diarize && pcmf32s.size() == 2) {
                    segment["speaker"] = estimate_diarization_speaker(
                        pcmf32s,
                        whisper_full_get_segment_t0(ctx, i),
                        whisper_full_get_segment_t1(ctx, i),
                        true);
                }

                float total_logprob = 0;
                const int n_tokens = whisper_full_n_tokens(ctx, i);
                for (int j = 0; j < n_tokens; ++j) {
                    whisper_token_data token = whisper_full_get_token_data(ctx, i, j);
                    if (token.id >= whisper_token_eot(ctx)) {
                        continue;
                    }

                    segment["tokens"].push_back(token.id);
                    std::string word_text = whisper_full_get_token_text(ctx, i, j);
                    int64_t word_t1 = token.t1;

                    while (j + 1 < n_tokens && utf8_trailing_bytes_needed(word_text) > 0) {
                        const whisper_token_data next_token = whisper_full_get_token_data(ctx, i, j + 1);
                        // Keep verbose_json tokens free of EOT ids, matching the pre-merge server behavior.
                        if (next_token.id >= whisper_token_eot(ctx)) {
                            break;
                        }

                        ++j;
                        segment["tokens"].push_back(next_token.id);
                        word_text += whisper_full_get_token_text(ctx, i, j);
                        if (next_token.t1 > -1) {
                            word_t1 = next_token.t1;
                        }
                        total_logprob += next_token.plog;
                    }

                    json word = json{{"word", word_text}};
                    if (!params.no_timestamps && params.token_timestamps) {
                        word["start"] = token.t0 * 0.01;
                        word["end"] = word_t1 * 0.01;
                        word["t_dtw"] = token.t_dtw;
                    }
                    word["probability"] = token.p;
                    total_logprob += token.plog;
                    segment["words"].push_back(word);
                }

                segment["temperature"] = params.temperature;
                segment["avg_logprob"] = total_logprob / n_tokens;

                // TODO compression_ratio and no_speech_prob are not implemented yet
                // segment["compression_ratio"] = 0;
                segment["no_speech_prob"] = whisper_full_get_segment_no_speech_prob(ctx, i);

                jres["segments"].push_back(segment);
            }
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        }
        // TODO add more output formats
        else
        {
            std::string results = output_str(ctx, params, pcmf32s);
            json jres = json{
                {"text", results}
            };
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        }

    });
    svr->Options(sparams.request_path + sparams.homan_inference_path, [&](const Request &, Response &){
    });

    svr->Post(sparams.request_path + sparams.homan_inference_path, [&](const Request & req, Response & res) {
        // External requests remain serialized. Only the items of this one
        // authenticated Homan batch may use multiple whisper_state instances.
        std::lock_guard<std::mutex> lock(whisper_mutex);
        const auto handler_started = std::chrono::steady_clock::now();

        auto fail = [&](int status, const std::string & message, const std::string & type = "invalid_request_error") {
            res.status = status;
            res.set_content(json{
                {"error", json{
                    {"message", message},
                    {"type", type},
                }},
            }.dump(), "application/json");
        };

        if (!ensure_context_loaded(res)) {
            return;
        }
        if (!req.has_file("manifest")) {
            fail(400, "missing multipart field 'manifest'");
            return;
        }

        json manifest;
        try {
            manifest = json::parse(req.get_file_value("manifest").content);
        } catch (const std::exception &) {
            fail(400, "manifest must be valid JSON");
            return;
        }

        try {
            if (!manifest.is_object()) {
                fail(400, "manifest must be a JSON object");
                return;
            }
            if (!manifest.contains("schema_version") ||
                    !manifest["schema_version"].is_number_integer() ||
                    manifest["schema_version"].get<int>() != 1) {
                fail(400, "unsupported or missing manifest schema_version");
                return;
            }
            if (!manifest.contains("request_id") ||
                    !manifest["request_id"].is_string() ||
                    manifest["request_id"].get<std::string>().empty() ||
                    manifest["request_id"].get<std::string>().size() > 128) {
                fail(400, "request_id must be a non-empty string up to 128 characters");
                return;
            }
            if (!manifest.contains("items") || !manifest["items"].is_array()) {
                fail(400, "manifest.items must be an array");
                return;
            }
        } catch (const std::exception &) {
            fail(400, "manifest has invalid field types");
            return;
        }

        const auto & manifest_items = manifest["items"];
        if (manifest_items.empty()) {
            fail(400, "manifest.items must not be empty");
            return;
        }
        if (manifest_items.size() > static_cast<size_t>(sparams.batch_max_items)) {
            fail(413, "batch contains too many audio items");
            return;
        }

        int requested_concurrency = 1;
        try {
            if (manifest.contains("options") && manifest["options"].is_object() &&
                    manifest["options"].contains("concurrency")) {
                requested_concurrency = manifest["options"]["concurrency"].get<int>();
            }
        } catch (const std::exception &) {
            fail(400, "options.concurrency must be an integer");
            return;
        }
        if (requested_concurrency < 1) {
            fail(400, "options.concurrency must be at least 1");
            return;
        }
        std::vector<homan_batch_item> items;
        items.reserve(manifest_items.size());
        std::unordered_set<std::string> ids;
        std::unordered_set<std::string> file_fields;
        constexpr size_t max_item_upload_bytes = 64ULL * 1024ULL * 1024ULL;
        constexpr double max_item_duration_seconds = 35.0;

        const auto decode_started = std::chrono::steady_clock::now();
        for (size_t index = 0; index < manifest_items.size(); ++index) {
            const auto & entry = manifest_items[index];
            try {
                if (!entry.is_object() ||
                        !entry.contains("id") || !entry["id"].is_string() ||
                        !entry.contains("source") || !entry["source"].is_string() ||
                        !entry.contains("file") || !entry["file"].is_string() ||
                        !entry.contains("start") || !entry["start"].is_number() ||
                        !entry.contains("end") || !entry["end"].is_number()) {
                    fail(400, "each item requires string id/source/file and numeric start/end");
                    return;
                }

                homan_batch_item item;
                item.id         = entry["id"].get<std::string>();
                item.source     = entry["source"].get<std::string>();
                item.file_field = entry["file"].get<std::string>();
                item.start      = entry["start"].get<double>();
                item.end        = entry["end"].get<double>();

                if (item.id.empty() || item.id.size() > 128 || !ids.insert(item.id).second) {
                    fail(400, "item ids must be unique non-empty strings up to 128 characters");
                    return;
                }
                if (item.source != "microphone" &&
                        item.source != "system" &&
                        item.source != "legacy_mixed") {
                    fail(400, "item.source must be microphone, system, or legacy_mixed");
                    return;
                }
                if (item.file_field.empty() || item.file_field.size() > 128 ||
                        !file_fields.insert(item.file_field).second) {
                    fail(400, "item file fields must be unique non-empty strings up to 128 characters");
                    return;
                }
                if (!std::isfinite(item.start) || !std::isfinite(item.end) ||
                        item.start < 0.0 || item.end <= item.start ||
                        item.end - item.start > max_item_duration_seconds) {
                    fail(400, "item start/end must describe a positive interval no longer than 35 seconds");
                    return;
                }
                if (!req.has_file(item.file_field)) {
                    fail(400, "missing multipart audio field '" + item.file_field + "'");
                    return;
                }

                const auto & upload = req.get_file_value(item.file_field);
                if (upload.content.empty() || upload.content.size() > max_item_upload_bytes) {
                    fail(413, "audio item '" + item.id + "' is empty or too large");
                    return;
                }
                item.filename = upload.filename.empty() ? item.file_field : upload.filename;
                item.encoded_audio = upload.content;
                items.push_back(std::move(item));
            } catch (const std::exception &) {
                fail(400, "manifest item " + std::to_string(index) + " has invalid field types");
                return;
            }
        }

        const int decode_concurrency = std::min(
            sparams.batch_decode_concurrency,
            static_cast<int>(items.size()));
        std::vector<std::string> decode_errors(items.size());
        std::atomic<size_t> next_decode_index{0};
        auto decode_worker = [&]() {
            while (true) {
                const size_t item_index = next_decode_index.fetch_add(1);
                if (item_index >= items.size()) {
                    return;
                }
                auto & item = items[item_index];
                try {
                    const bool decoded = decode_homan_audio(
                            item.encoded_audio,
                            item.filename,
                            sparams,
                            item.pcmf32,
                            decode_errors[item_index]);
                    if (decoded) {
                        const double decoded_duration =
                            static_cast<double>(item.pcmf32.size()) / WHISPER_SAMPLE_RATE;
                        if (decoded_duration < 0.1 ||
                                decoded_duration > max_item_duration_seconds + 0.5) {
                            decode_errors[item_index] = "decoded audio item '" + item.id +
                                "' must be between 0.1 and 35.5 seconds";
                        }
                    }
                } catch (const std::exception & exception) {
                    decode_errors[item_index] = "could not decode audio item '" +
                        item.id + "': " + exception.what();
                } catch (...) {
                    decode_errors[item_index] = "could not decode audio item '" +
                        item.id + "'";
                }
                std::string().swap(item.encoded_audio);
            }
        };
        std::vector<std::thread> decode_workers;
        decode_workers.reserve(static_cast<size_t>(std::max(0, decode_concurrency - 1)));
        for (int worker_index = 1; worker_index < decode_concurrency; ++worker_index) {
            decode_workers.emplace_back(decode_worker);
        }
        decode_worker();
        for (auto & thread : decode_workers) {
            thread.join();
        }
        for (const auto & decode_error : decode_errors) {
            if (!decode_error.empty()) {
                fail(400, decode_error);
                return;
            }
        }
        const auto decode_finished = std::chrono::steady_clock::now();

        std::vector<homan_batch_reel> reels = pack_homan_batch_reels(items);
        const int concurrency = std::min({
            requested_concurrency,
            sparams.batch_max_concurrency,
            static_cast<int>(reels.size()),
        });

        std::vector<whisper_state *> states(static_cast<size_t>(concurrency), nullptr);
        for (int index = 0; index < concurrency; ++index) {
            states[static_cast<size_t>(index)] = whisper_init_state(ctx);
            if (states[static_cast<size_t>(index)] == nullptr) {
                for (auto * state_ptr : states) {
                    if (state_ptr != nullptr) {
                        whisper_free_state(state_ptr);
                    }
                }
                fail(503, "could not allocate whisper state for requested batch concurrency", "server_capacity_error");
                return;
            }
        }

        whisper_params batch_params = default_params;
        batch_params.language = "auto";
        batch_params.detect_language = false;
        batch_params.translate = false;
        batch_params.diarize = false;
        batch_params.tinydiarize = false;
        batch_params.token_timestamps = false;
        batch_params.no_context = true;

        std::vector<json> item_results;
        item_results.reserve(items.size());
        for (const auto & item : items) {
            item_results.push_back(make_homan_batch_item_result(item));
        }
        std::vector<homan_batch_reel_result> reel_results(reels.size());
        std::vector<std::string> reel_errors(reels.size());
        std::atomic<size_t> next_index{0};
        auto worker = [&](int worker_index) {
            while (true) {
                const size_t reel_index = next_index.fetch_add(1);
                if (reel_index >= reels.size()) {
                    return;
                }
                try {
                    transcribe_homan_batch_reel(
                        ctx,
                        states[static_cast<size_t>(worker_index)],
                        batch_params,
                        reels[reel_index],
                        items,
                        item_results,
                        reel_results[reel_index],
                        reel_errors[reel_index]);
                } catch (const std::exception & exception) {
                    reel_errors[reel_index] =
                        "packed reel failed: " + std::string(exception.what());
                } catch (...) {
                    reel_errors[reel_index] =
                        "packed reel failed with an unknown error";
                }
            }
        };

        const auto inference_started = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(std::max(0, concurrency - 1)));
        for (int worker_index = 1; worker_index < concurrency; ++worker_index) {
            workers.emplace_back(worker, worker_index);
        }
        worker(0);
        for (auto & thread : workers) {
            thread.join();
        }

        for (size_t index = 0; index < reel_errors.size(); ++index) {
            if (!reel_errors[index].empty()) {
                for (auto * state_ptr : states) {
                    whisper_free_state(state_ptr);
                }
                fail(500, reel_errors[index], "server_error");
                return;
            }
        }

        std::unordered_set<size_t> fallback_item_set;
        for (const auto & reel_result : reel_results) {
            fallback_item_set.insert(
                reel_result.fallback_item_indices.begin(),
                reel_result.fallback_item_indices.end());
        }
        for (size_t item_index = 0; item_index < item_results.size(); ++item_index) {
            if (trim_transcript(
                    item_results[item_index]["text"].get<std::string>()).empty()) {
                fallback_item_set.insert(item_index);
            }
        }
        std::vector<size_t> fallback_item_indices(
            fallback_item_set.begin(), fallback_item_set.end());
        std::sort(fallback_item_indices.begin(), fallback_item_indices.end());

        std::vector<homan_batch_item> fallback_repack_items;
        std::vector<size_t> fallback_repack_original_indices;
        std::vector<size_t> isolated_fallback_item_indices;
        if (sparams.batch_fallback_repack_short) {
            for (const size_t item_index : fallback_item_indices) {
                const double item_duration = static_cast<double>(
                    items[item_index].pcmf32.size()) / WHISPER_SAMPLE_RATE;
                if (item_duration <= homan_fallback_repack_max_item_seconds) {
                    homan_batch_item repack_item = items[item_index];
                    repack_item.packing_language =
                        item_results[item_index]["language"].get<std::string>();
                    fallback_repack_items.push_back(std::move(repack_item));
                    fallback_repack_original_indices.push_back(item_index);
                } else {
                    isolated_fallback_item_indices.push_back(item_index);
                }
            }
        } else {
            isolated_fallback_item_indices = fallback_item_indices;
        }

        std::vector<homan_batch_reel> fallback_repack_reels;
        std::vector<homan_batch_reel_result> fallback_repack_reel_results;
        int fallback_repack_cross_item_segments = 0;
        size_t fallback_repack_residual_item_count = 0;
        if (!fallback_repack_items.empty()) {
            fallback_repack_reels = pack_homan_batch_reels(
                fallback_repack_items,
                homan_fallback_repack_separator_seconds);
            std::vector<json> fallback_repack_item_results;
            fallback_repack_item_results.reserve(fallback_repack_items.size());
            for (const auto & item : fallback_repack_items) {
                fallback_repack_item_results.push_back(
                    make_homan_batch_item_result(item));
            }
            fallback_repack_reel_results.resize(fallback_repack_reels.size());
            std::vector<std::string> fallback_repack_errors(
                fallback_repack_reels.size());
            std::atomic<size_t> next_fallback_reel_index{0};
            auto fallback_repack_worker = [&](int worker_index) {
                while (true) {
                    const size_t reel_index = next_fallback_reel_index.fetch_add(1);
                    if (reel_index >= fallback_repack_reels.size()) {
                        return;
                    }
                    try {
                        transcribe_homan_batch_reel(
                            ctx,
                            states[static_cast<size_t>(worker_index)],
                            batch_params,
                            fallback_repack_reels[reel_index],
                            fallback_repack_items,
                            fallback_repack_item_results,
                            fallback_repack_reel_results[reel_index],
                            fallback_repack_errors[reel_index]);
                    } catch (const std::exception & exception) {
                        fallback_repack_errors[reel_index] =
                            "fallback repack reel failed: " +
                            std::string(exception.what());
                    } catch (...) {
                        fallback_repack_errors[reel_index] =
                            "fallback repack reel failed with an unknown error";
                    }
                }
            };
            std::vector<std::thread> fallback_repack_workers;
            fallback_repack_workers.reserve(
                static_cast<size_t>(std::max(0, concurrency - 1)));
            for (int worker_index = 1; worker_index < concurrency; ++worker_index) {
                fallback_repack_workers.emplace_back(
                    fallback_repack_worker, worker_index);
            }
            fallback_repack_worker(0);
            for (auto & thread : fallback_repack_workers) {
                thread.join();
            }
            for (const auto & fallback_repack_error : fallback_repack_errors) {
                if (!fallback_repack_error.empty()) {
                    for (auto * state_ptr : states) {
                        whisper_free_state(state_ptr);
                    }
                    fail(500, fallback_repack_error, "server_error");
                    return;
                }
            }

            std::unordered_set<size_t> fallback_repack_residual_set;
            for (const auto & reel_result : fallback_repack_reel_results) {
                fallback_repack_cross_item_segments +=
                    reel_result.cross_item_segments;
                fallback_repack_residual_set.insert(
                    reel_result.fallback_item_indices.begin(),
                    reel_result.fallback_item_indices.end());
            }
            for (size_t item_index = 0;
                    item_index < fallback_repack_item_results.size();
                    ++item_index) {
                if (trim_transcript(
                        fallback_repack_item_results[item_index]["text"]
                            .get<std::string>()).empty()) {
                    fallback_repack_residual_set.insert(item_index);
                }
            }
            fallback_repack_residual_item_count =
                fallback_repack_residual_set.size();
            for (size_t item_index = 0;
                    item_index < fallback_repack_item_results.size();
                    ++item_index) {
                const size_t original_item_index =
                    fallback_repack_original_indices[item_index];
                if (fallback_repack_residual_set.count(item_index) != 0) {
                    isolated_fallback_item_indices.push_back(original_item_index);
                } else {
                    item_results[original_item_index] =
                        std::move(fallback_repack_item_results[item_index]);
                }
            }
        }

        std::sort(
            isolated_fallback_item_indices.begin(),
            isolated_fallback_item_indices.end());
        isolated_fallback_item_indices.erase(
            std::unique(
                isolated_fallback_item_indices.begin(),
                isolated_fallback_item_indices.end()),
            isolated_fallback_item_indices.end());
        std::vector<double> isolated_fallback_processing_ms(
            isolated_fallback_item_indices.size(), 0.0);
        std::vector<std::string> fallback_errors(
            isolated_fallback_item_indices.size());
        std::atomic<size_t> next_fallback_index{0};
        auto fallback_worker = [&](int worker_index) {
            while (true) {
                const size_t fallback_index = next_fallback_index.fetch_add(1);
                if (fallback_index >= isolated_fallback_item_indices.size()) {
                    return;
                }
                const size_t item_index =
                    isolated_fallback_item_indices[fallback_index];
                try {
                    whisper_params fallback_params = batch_params;
                    const std::string reel_language =
                        item_results[item_index]["language"].get<std::string>();
                    if (!reel_language.empty()) {
                        fallback_params.language = reel_language;
                    }
                    json fallback_result = transcribe_homan_batch_item(
                        ctx,
                        states[static_cast<size_t>(worker_index)],
                        fallback_params,
                        items[item_index],
                        fallback_errors[fallback_index]);
                    if (fallback_errors[fallback_index].empty()) {
                        isolated_fallback_processing_ms[fallback_index] =
                            fallback_result["processing_ms"].get<double>();
                        item_results[item_index] = std::move(fallback_result);
                    }
                } catch (const std::exception & exception) {
                    fallback_errors[fallback_index] =
                        "fallback item failed: " + std::string(exception.what());
                } catch (...) {
                    fallback_errors[fallback_index] =
                        "fallback item failed with an unknown error";
                }
            }
        };

        if (!isolated_fallback_item_indices.empty()) {
            std::vector<std::thread> fallback_workers;
            fallback_workers.reserve(static_cast<size_t>(std::max(0, concurrency - 1)));
            for (int worker_index = 1; worker_index < concurrency; ++worker_index) {
                fallback_workers.emplace_back(fallback_worker, worker_index);
            }
            fallback_worker(0);
            for (auto & thread : fallback_workers) {
                thread.join();
            }
        }
        const auto inference_finished = std::chrono::steady_clock::now();

        for (auto * state_ptr : states) {
            whisper_free_state(state_ptr);
        }

        for (size_t index = 0; index < fallback_errors.size(); ++index) {
            if (!fallback_errors[index].empty()) {
                fail(500, fallback_errors[index], "server_error");
                return;
            }
        }

        json response_items = json::array();
        std::string combined_text;
        double audio_duration = 0.0;
        for (size_t index = 0; index < item_results.size(); ++index) {
            if (!combined_text.empty() && !item_results[index]["text"].get<std::string>().empty()) {
                combined_text += "\n";
            }
            combined_text += item_results[index]["text"].get<std::string>();
            audio_duration += item_results[index]["audio_duration"].get<double>();
            response_items.push_back(std::move(item_results[index]));
        }
        double packed_audio_duration = 0.0;
        double summed_reel_processing_ms = 0.0;
        int cross_item_segments = 0;
        for (size_t index = 0; index < reels.size(); ++index) {
            packed_audio_duration += static_cast<double>(reels[index].pcmf32.size()) /
                WHISPER_SAMPLE_RATE;
            summed_reel_processing_ms += reel_results[index].processing_ms;
            cross_item_segments += reel_results[index].cross_item_segments;
        }
        double fallback_repack_packed_audio_duration = 0.0;
        double summed_fallback_repack_processing_ms = 0.0;
        for (size_t index = 0; index < fallback_repack_reels.size(); ++index) {
            fallback_repack_packed_audio_duration += static_cast<double>(
                fallback_repack_reels[index].pcmf32.size()) / WHISPER_SAMPLE_RATE;
            summed_fallback_repack_processing_ms +=
                fallback_repack_reel_results[index].processing_ms;
        }
        double summed_isolated_fallback_processing_ms = 0.0;
        for (const double processing_ms : isolated_fallback_processing_ms) {
            summed_isolated_fallback_processing_ms += processing_ms;
        }
        const double summed_fallback_processing_ms =
            summed_fallback_repack_processing_ms +
            summed_isolated_fallback_processing_ms;
        const double summed_processing_ms =
            summed_reel_processing_ms + summed_fallback_processing_ms;

        const auto handler_finished = std::chrono::steady_clock::now();
        const double decode_ms = std::chrono::duration<double, std::milli>(
            decode_finished - decode_started).count();
        const double inference_wall_ms = std::chrono::duration<double, std::milli>(
            inference_finished - inference_started).count();
        const double server_wall_ms = std::chrono::duration<double, std::milli>(
            handler_finished - handler_started).count();

        json response = {
            {"schema_version", 1},
            {"request_id", manifest["request_id"]},
            {"model", "large-v3-turbo"},
            {"language", "auto"},
            {"text", trim_transcript(combined_text)},
            {"audio_duration", audio_duration},
            {"concurrency_requested", requested_concurrency},
            {"concurrency_used", concurrency},
            {"decode_concurrency_used", decode_concurrency},
            {"decode_ms", decode_ms},
            {"inference_wall_ms", inference_wall_ms},
            {"summed_item_processing_ms", summed_processing_ms},
            {"summed_reel_processing_ms", summed_reel_processing_ms},
            {"server_wall_ms", server_wall_ms},
            {"packing", json{
                {"enabled", true},
                {"reel_count", reels.size()},
                {"max_reel_seconds", homan_reel_max_seconds},
                {"separator_seconds", homan_reel_separator_seconds},
                {"packed_audio_duration", packed_audio_duration},
                {"cross_item_segments", cross_item_segments},
                {"fallback_item_count", fallback_item_indices.size()},
                {"fallback_processing_ms", summed_fallback_processing_ms},
                {"fallback_repack_enabled", sparams.batch_fallback_repack_short},
                {"fallback_repack_item_count", fallback_repack_items.size()},
                {"fallback_repack_reel_count", fallback_repack_reels.size()},
                {"fallback_repack_separator_seconds",
                    homan_fallback_repack_separator_seconds},
                {"fallback_repack_packed_audio_duration",
                    fallback_repack_packed_audio_duration},
                {"fallback_repack_cross_item_segments",
                    fallback_repack_cross_item_segments},
                {"fallback_repack_residual_item_count",
                    fallback_repack_residual_item_count},
                {"fallback_repack_processing_ms",
                    summed_fallback_repack_processing_ms},
                {"fallback_isolated_item_count",
                    isolated_fallback_item_indices.size()},
                {"fallback_isolated_processing_ms",
                    summed_isolated_fallback_processing_ms},
            }},
            {"items", std::move(response_items)},
        };
        res.set_content(
            response.dump(-1, ' ', false, json::error_handler_t::replace),
            "application/json");
    });

    svr->Post(sparams.request_path + "/load", [&](const Request &req, Response &res){
        std::lock_guard<std::mutex> lock(whisper_mutex);
        state.store(SERVER_STATE_LOADING_MODEL);
        if (!req.has_file("model"))
        {
            fprintf(stderr, "error: no 'model' field in the request\n");
            const std::string error_resp = "{\"error\":\"no 'model' field in the request\"}";
            res.status = 400;
            res.set_content(error_resp, "application/json");
            return;
        }
        std::string model = req.get_file_value("model").content;
        if (!is_file_exist(model.c_str()))
        {
            fprintf(stderr, "error: 'model': %s not found!\n", model.c_str());
            const std::string error_resp = "{\"error\":\"model not found!\"}";
            res.status = 400;
            res.set_content(error_resp, "application/json");
            return;
        }

        // clean up
        if (ctx != nullptr) {
            whisper_free(ctx);
            ctx = nullptr;
        }

        // whisper init
        ctx = whisper_init_from_file_with_params(model.c_str(), cparams);

        // TODO perhaps load prior model here instead of exit
        if (ctx == nullptr) {
            fprintf(stderr, "error: model init  failed, no model loaded must exit\n");
            exit(1);
        }

        default_params.model = model;
        state.store(SERVER_STATE_READY);
        const std::string success = "Load was successful!";
        res.set_content(success, "application/text");

        // check if the model is in the file system
    });

    svr->Get(sparams.request_path + "/health", [&](const Request &, Response &res){
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_READY) {
            const std::string health_response = "{\"status\":\"ok\"}";
            res.set_content(health_response, "application/json");
        } else {
            // The server is live and deliberately loads the model on the first
            // inference worker request to preserve Vulkan thread affinity.
            res.set_content("{\"status\":\"ok\",\"model_loaded\":false}", "application/json");
        }
    });

    svr->set_exception_handler([](const Request &, Response &res, std::exception_ptr ep) {
        const char fmt[] = "500 Internal Server Error\n%s";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception &e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/plain");
        res.status = 500;
    });

    svr->set_error_handler([](const Request &req, Response &res) {
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("File Not Found (" + req.path + ")", "text/plain");
            res.status = 404;
        }
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout(sparams.read_timeout);
    svr->set_write_timeout(sparams.write_timeout);

    if (!svr->bind_to_port(sparams.hostname, sparams.port))
    {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n",
                sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    // Set the base directory for serving static files
    svr->set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    printf("\nwhisper server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    shutdown_handler = [&](int signal) {
        printf("\nCaught signal %d, shutting down gracefully...\n", signal);
        if (svr) {
            svr->stop();
        }
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // clean up function, to be called before exit
    auto clean_up = [&]() {
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
    };

    std::thread t([&] {
        if (!svr->listen_after_bind()) {
            fprintf(stderr, "error: server listen failed\n");
        }
    });

    svr->wait_until_ready();

    t.join();


    clean_up();

    return 0;
}
