use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use futures_util::StreamExt;
use kokoro_micro::TtsEngine;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tower_http::cors::CorsLayer;

const SAMPLE_RATE: u32 = 24000;
const HF_VOICES_URL: &str = "https://github.com/8b-is/kokoro-tiny/raw/main/models/0.bin";

#[derive(Clone)]
struct AppState {
    tts: Arc<RwLock<Option<TtsEngine>>>,
    download_state: Arc<Mutex<DownloadProgress>>,
    cache_dir: PathBuf,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct DownloadProgress {
    is_downloading: bool,
    current_file: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    percentage: f32,
    status_message: String,
    error: Option<String>,
}

#[derive(Serialize)]
struct SystemStatus {
    model_cached: bool,
    model_size_bytes: u64,
    voices_size_bytes: u64,
    cache_path: String,
    download: DownloadProgress,
    available_voices_count: usize,
    active_quantization: String,
    quantizations: Vec<QuantizationOption>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct QuantizationOption {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub description: String,
    pub size_mb: f32,
}

#[derive(Deserialize, Default)]
pub struct DownloadPayload {
    pub quantization: Option<String>,
}

#[derive(Deserialize)]
struct GenerateRequest {
    text: String,
    voice: Option<String>,
    speed: Option<f32>,
    pitch_semitones: Option<f32>,
    gain: Option<f32>,
}

#[derive(Serialize)]
struct GenerateResponse {
    success: bool,
    duration_seconds: f32,
    sample_count: usize,
    sample_rate: u32,
    voice: String,
    wav_base64: String,
    latency_ms: u64,
    error: Option<String>,
}

#[derive(Serialize, Clone)]
struct VoiceInfo {
    id: String,
    name: String,
    gender: String,
    language: String,
    accent: String,
    flag: String,
    tone: String,
    pace: String,
    recommended_speed: f32,
    traits: Vec<String>,
    description: String,
}

fn get_cache_directory() -> PathBuf {
    // ponytail: check for local models next to exe for portable distribution
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let local_models = exe_dir.join("models");
            if local_models.exists() {
                return local_models;
            }
            if exe_dir.join("0.onnx").exists() {
                return exe_dir.to_path_buf();
            }
            let local_cache = exe_dir.join(".cache").join("k");
            if local_cache.exists() {
                return local_cache;
            }
        }
    }

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("k")
}

async fn load_engine_from_cache(cache_dir: &Path) -> Result<TtsEngine, String> {
    let model_path = cache_dir.join("0.onnx");
    let voices_path = cache_dir.join("0.bin");
    if !model_path.exists() || !voices_path.exists() {
        return Err("Model or voices file missing in cache directory".to_string());
    }
    TtsEngine::with_paths(
        model_path.to_str().unwrap_or("0.onnx"),
        voices_path.to_str().unwrap_or("0.bin"),
    )
    .await
}

fn convert_samples_to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| format!("WAV create error: {e}"))?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let s16 = (clamped * 32767.0) as i16;
            writer
                .write_sample(s16)
                .map_err(|e| format!("WAV write sample error: {e}"))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("WAV finalize error: {e}"))?;
    }
    Ok(cursor.into_inner())
}

// ponytail: basic pitch shifting or bypass when semitones == 0
fn apply_pitch_shift(samples: &[f32], semitones: f32) -> Vec<f32> {
    if semitones.abs() < 0.05 {
        return samples.to_vec();
    }
    use pitch_shift::{Shifter, TOTAL_F32};
    let mut shifter = Shifter::new(Box::new([0.0f32; TOTAL_F32]));
    let mut out = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(128) {
        if chunk.len() == 128 {
            let shifted = shifter.shift(chunk, semitones, 128, SAMPLE_RATE as f32);
            out.extend_from_slice(shifted);
        } else {
            let mut padded = [0.0f32; 128];
            padded[..chunk.len()].copy_from_slice(chunk);
            let shifted = shifter.shift(&padded, semitones, 128, SAMPLE_RATE as f32);
            out.extend_from_slice(&shifted[..chunk.len()]);
        }
    }
    out
}

fn get_voice_catalog() -> Vec<VoiceInfo> {
    let v = |id: &str, name: &str, gender: &str, lang: &str, acc: &str, flag: &str, tone: &str, pace: &str, spd: f32, traits: &[&str], desc: &str| -> VoiceInfo {
        VoiceInfo {
            id: id.into(),
            name: name.into(),
            gender: gender.into(),
            language: lang.into(),
            accent: acc.into(),
            flag: flag.into(),
            tone: tone.into(),
            pace: pace.into(),
            recommended_speed: spd,
            traits: traits.iter().map(|&s| s.into()).collect(),
            description: desc.into(),
        }
    };

    vec![
        // American English - Female
        v("af_sky", "Sky", "Female", "English", "American", "🇺🇸", "Clear", "Standard", 1.00, &["clear", "expressive", "default"], "Expressive & clear (Default)"),
        v("af_bella", "Bella", "Female", "English", "American", "🇺🇸", "Warm", "Standard", 0.95, &["warm", "natural", "conversational"], "Warm & natural"),
        v("af_sarah", "Sarah", "Female", "English", "American", "🇺🇸", "News", "Brisk", 1.05, &["professional", "news", "articulate"], "Professional & articulate"),
        v("af_nicole", "Nicole", "Female", "English", "American", "🇺🇸", "Bright", "Standard", 1.00, &["crisp", "friendly", "casual"], "Crisp & friendly"),
        v("af_heart", "Heart", "Female", "English", "American", "🇺🇸", "Soft", "Relaxed", 0.90, &["soft", "gentle", "meditation"], "Soft & gentle"),
        v("af_alloy", "Alloy", "Female", "English", "American", "🇺🇸", "Clear", "Standard", 1.00, &["balanced", "modern", "tech"], "Balanced & modern"),
        v("af_aoede", "Aoede", "Female", "English", "American", "🇺🇸", "Calm", "Relaxed", 0.95, &["melodic", "smooth", "audiobook"], "Melodic & smooth"),
        v("af_jessica", "Jessica", "Female", "English", "American", "🇺🇸", "Bright", "Brisk", 1.05, &["bright", "energetic", "youthful"], "Bright & energetic"),
        v("af_kore", "Kore", "Female", "English", "American", "🇺🇸", "Calm", "Relaxed", 0.90, &["calm", "relaxed", "peaceful"], "Calm & relaxed"),
        v("af_nova", "Nova", "Female", "English", "American", "🇺🇸", "Bright", "Brisk", 1.05, &["dynamic", "lively", "commercial"], "Dynamic & lively"),
        v("af_river", "River", "Female", "English", "American", "🇺🇸", "Narrator", "Relaxed", 0.92, &["deep", "resonant", "narrator"], "Deep & resonant"),

        // American English - Male
        v("am_adam", "Adam", "Male", "English", "American", "🇺🇸", "Narrator", "Relaxed", 0.95, &["deep", "narrator", "documentary"], "Deep & narrator style"),
        v("am_echo", "Echo", "Male", "English", "American", "🇺🇸", "Clear", "Standard", 1.00, &["conversational", "modern", "podcast"], "Conversational & modern"),
        v("am_eric", "Eric", "Male", "English", "American", "🇺🇸", "Warm", "Standard", 1.00, &["friendly", "casual", "approachable"], "Friendly & casual"),
        v("am_liam", "Liam", "Male", "English", "American", "🇺🇸", "Bright", "Brisk", 1.05, &["youthful", "upbeat", "vlog"], "Youthful & upbeat"),
        v("am_michael", "Michael", "Male", "English", "American", "🇺🇸", "News", "Brisk", 1.05, &["authoritative", "news", "broadcast"], "Authoritative & news style"),
        v("am_onyx", "Onyx", "Male", "English", "American", "🇺🇸", "Narrator", "Relaxed", 0.92, &["deep", "warm", "cinematic"], "Deep & warm tone"),
        v("am_puck", "Puck", "Male", "English", "American", "🇺🇸", "Playful", "Standard", 1.00, &["playful", "dramatic", "character"], "Playful & dramatic"),
        v("am_fenrir", "Fenrir", "Male", "English", "American", "🇺🇸", "Narrator", "Relaxed", 0.90, &["rich", "cinematic", "trailer"], "Rich & cinematic"),
        v("am_santa", "Santa", "Male", "English", "American", "🇺🇸", "Playful", "Standard", 0.95, &["boisterous", "festive", "character"], "Boisterous & festive"),

        // British English - Female
        v("bf_alice", "Alice", "Female", "English", "British", "🇬🇧", "Clear", "Standard", 1.00, &["sophisticated", "clear", "corporate"], "Sophisticated & clear"),
        v("bf_emma", "Emma", "Female", "English", "British", "🇬🇧", "Narrator", "Relaxed", 0.95, &["classic", "rp", "refined", "audiobook"], "Classic RP & refined"),
        v("bf_isabella", "Isabella", "Female", "English", "British", "🇬🇧", "Warm", "Standard", 0.95, &["warm", "eloquent", "literature"], "Warm & eloquent"),
        v("bf_lily", "Lily", "Female", "English", "British", "🇬🇧", "Soft", "Relaxed", 0.90, &["gentle", "delicate", "intimate"], "Gentle & delicate"),

        // British English - Male
        v("bm_daniel", "Daniel", "Male", "English", "British", "🇬🇧", "News", "Brisk", 1.05, &["professional", "crisp", "presenter"], "Professional & crisp"),
        v("bm_fable", "Fable", "Male", "English", "British", "🇬🇧", "Narrator", "Relaxed", 0.92, &["storyteller", "engaging", "theatrical"], "Storyteller & engaging"),
        v("bm_george", "George", "Male", "English", "British", "🇬🇧", "Warm", "Standard", 1.00, &["warm", "conversational", "friendly"], "Warm & conversational"),
        v("bm_lewis", "Lewis", "Male", "English", "British", "🇬🇧", "Calm", "Relaxed", 0.95, &["measured", "calm", "documentary"], "Measured & calm"),

        // Japanese
        v("jf_alpha", "Alpha (JP)", "Female", "Japanese", "Standard", "🇯🇵", "Clear", "Standard", 1.00, &["natural", "clear", "standard"], "Natural & clear"),
        v("jf_gongitsune", "Gongitsune", "Female", "Japanese", "Narrative", "🇯🇵", "Narrator", "Relaxed", 0.95, &["storytelling", "traditional"], "Folklore storytelling"),
        v("jf_nezumi", "Nezumi", "Female", "Japanese", "Anime", "🇯🇵", "Playful", "Brisk", 1.05, &["high-pitched", "anime", "lively"], "High-pitched & lively"),
        v("jf_tebukuro", "Tebukuro", "Female", "Japanese", "Soft", "🇯🇵", "Soft", "Relaxed", 0.90, &["gentle", "calming", "bedtime"], "Gentle & calming"),
        v("jm_kumo", "Kumo", "Male", "Japanese", "Standard", "🇯🇵", "Calm", "Standard", 1.00, &["calm", "composed", "standard"], "Calm & composed"),

        // Mandarin Chinese
        v("zf_xiaobei", "Xiaobei", "Female", "Chinese", "Mandarin", "🇨🇳", "Clear", "Standard", 1.00, &["beijing", "crisp", "standard"], "Beijing accent & crisp"),
        v("zf_xiaoni", "Xiaoni", "Female", "Chinese", "Mandarin", "🇨🇳", "News", "Brisk", 1.05, &["broadcast", "standard", "news"], "Standard broadcast voice"),
        v("zf_xiaoxiao", "Xiaoxiao", "Female", "Chinese", "Mandarin", "🇨🇳", "Warm", "Standard", 0.95, &["warm", "gentle", "conversational"], "Warm & gentle"),
        v("zf_xiaoyi", "Xiaoyi", "Female", "Chinese", "Mandarin", "🇨🇳", "Bright", "Standard", 1.00, &["expressive", "modern"], "Expressive & modern"),
        v("zm_yunjian", "Yunjian", "Male", "Chinese", "Mandarin", "🇨🇳", "News", "Brisk", 1.05, &["articulate", "formal", "business"], "Articulate & formal"),
        v("zm_yunxi", "Yunxi", "Male", "Chinese", "Mandarin", "🇨🇳", "Warm", "Standard", 1.00, &["warm", "conversational"], "Warm conversational"),
        v("zm_yunxia", "Yunxia", "Male", "Chinese", "Mandarin", "🇨🇳", "Narrator", "Relaxed", 0.95, &["deep", "authoritative", "documentary"], "Deep & authoritative"),
        v("zm_yunyang", "Yunyang", "Male", "Chinese", "Mandarin", "🇨🇳", "Bright", "Brisk", 1.05, &["youthful", "energetic"], "Youthful & energetic"),

        // Spanish
        v("ef_dora", "Dora", "Female", "Spanish", "Castilian", "🇪🇸", "Clear", "Standard", 1.00, &["clear", "natural", "castilian"], "Clear & natural"),
        v("em_alex", "Alex", "Male", "Spanish", "Neutral", "🇪🇸", "Warm", "Standard", 1.00, &["warm", "articulate", "latin"], "Warm & articulate"),
        v("em_santa", "Santa (ES)", "Male", "Spanish", "Festive", "🇪🇸", "Playful", "Standard", 0.95, &["deep", "festive"], "Deep & festive"),

        // French
        v("ff_siwis", "Siwis", "Female", "French", "Standard", "🇫🇷", "Clear", "Standard", 1.00, &["elegant", "natural", "parisian"], "Elegant & natural"),

        // Italian
        v("if_sara", "Sara", "Female", "Italian", "Standard", "🇮🇹", "Warm", "Standard", 1.00, &["melodic", "warm", "expressive"], "Melodic & warm"),
        v("im_nicola", "Nicola", "Male", "Italian", "Standard", "🇮🇹", "Bright", "Standard", 1.00, &["clear", "engaging", "energetic"], "Clear & engaging"),

        // Portuguese
        v("pf_dora", "Dora (PT)", "Female", "Portuguese", "Brazilian", "🇧🇷", "Warm", "Standard", 1.00, &["smooth", "melodic", "brazilian"], "Smooth & melodic"),
        v("pm_alex", "Alex (PT)", "Male", "Portuguese", "Brazilian", "🇧🇷", "Clear", "Standard", 1.00, &["clear", "friendly"], "Clear & friendly"),
        v("pm_santa", "Santa (PT)", "Male", "Portuguese", "Festive", "🇧🇷", "Playful", "Standard", 0.95, &["deep", "festive"], "Deep tone"),

        // Hindi
        v("hf_alpha", "Alpha (HI)", "Female", "Hindi", "Standard", "🇮🇳", "Clear", "Standard", 1.00, &["articulate", "clear"], "Articulate & clear"),
        v("hf_beta", "Beta (HI)", "Female", "Hindi", "Standard", "🇮🇳", "Warm", "Standard", 0.95, &["warm", "melodic"], "Warm & melodic"),
        v("hm_omega", "Omega", "Male", "Hindi", "Standard", "🇮🇳", "Narrator", "Relaxed", 0.92, &["deep", "resonant"], "Deep & resonant"),
        v("hm_psi", "Psi", "Male", "Hindi", "Standard", "🇮🇳", "Narrator", "Standard", 0.95, &["strong", "narrative"], "Strong & narrative"),
    ]
}

fn get_quantization_catalog() -> Vec<QuantizationOption> {
    vec![
        QuantizationOption {
            id: "quantized".to_string(),
            name: "INT8 Quantized".to_string(),
            filename: "model_quantized.onnx".to_string(),
            description: "Fastest CPU inference, lowest RAM (~88 MB) - Recommended".to_string(),
            size_mb: 88.1,
        },
        QuantizationOption {
            id: "fp32".to_string(),
            name: "FP32 Full Precision".to_string(),
            filename: "model.onnx".to_string(),
            description: "Original unquantized floating point (~311 MB) - Highest Quality".to_string(),
            size_mb: 310.5,
        },
        QuantizationOption {
            id: "fp16".to_string(),
            name: "FP16 Half Precision".to_string(),
            filename: "model_fp16.onnx".to_string(),
            description: "16-bit floating point (~156 MB)".to_string(),
            size_mb: 155.7,
        },
        QuantizationOption {
            id: "q8f16".to_string(),
            name: "Q8_F16 Mixed Precision".to_string(),
            filename: "model_q8f16.onnx".to_string(),
            description: "8-bit weights, 16-bit activations (~82 MB) - Smallest footprint".to_string(),
            size_mb: 82.0,
        },
        QuantizationOption {
            id: "q4f16".to_string(),
            name: "Q4_F16 4-bit Mixed".to_string(),
            filename: "model_q4f16.onnx".to_string(),
            description: "4-bit weights, 16-bit activations (~147 MB)".to_string(),
            size_mb: 147.4,
        },
        QuantizationOption {
            id: "q4".to_string(),
            name: "Q4 Quantized".to_string(),
            filename: "model_q4.onnx".to_string(),
            description: "4-bit quantization (~291 MB)".to_string(),
            size_mb: 291.1,
        },
        QuantizationOption {
            id: "uint8".to_string(),
            name: "UINT8 Quantized".to_string(),
            filename: "model_uint8.onnx".to_string(),
            description: "Unsigned 8-bit quantization (~169 MB)".to_string(),
            size_mb: 169.2,
        },
    ]
}

fn get_active_quantization(cache_dir: &Path) -> String {
    let quant_file = cache_dir.join("quantization.txt");
    if let Ok(content) = fs::read_to_string(&quant_file) {
        let trimmed = content.trim().to_lowercase();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    let model_path = cache_dir.join("0.onnx");
    if let Ok(meta) = fs::metadata(&model_path) {
        let mb = meta.len() as f64 / 1_048_576.0;
        if mb > 200.0 {
            return "fp32".to_string();
        } else if mb < 100.0 {
            return "quantized".to_string();
        } else if mb < 165.0 {
            return "fp16".to_string();
        }
    }
    "quantized".to_string()
}

async fn handle_status(State(state): State<AppState>) -> Json<SystemStatus> {
    let model_path = state.cache_dir.join("0.onnx");
    let voices_path = state.cache_dir.join("0.bin");

    let model_size = fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
    let voices_size = fs::metadata(&voices_path).map(|m| m.len()).unwrap_or(0);
    let model_cached = model_path.exists() && voices_path.exists() && model_size > 50_000_000;

    let download = {
        let lock = state.download_state.lock().await;
        lock.clone()
    };

    let voices_count = get_voice_catalog().len();
    let active_quantization = get_active_quantization(&state.cache_dir);
    let quantizations = get_quantization_catalog();

    Json(SystemStatus {
        model_cached,
        model_size_bytes: model_size,
        voices_size_bytes: voices_size,
        cache_path: state.cache_dir.to_string_lossy().to_string(),
        download,
        available_voices_count: voices_count,
        active_quantization,
        quantizations,
    })
}

async fn handle_voices() -> Json<Vec<VoiceInfo>> {
    Json(get_voice_catalog())
}

async fn download_file_with_progress(
    client: &reqwest::Client,
    url: &str,
    target_path: &Path,
    file_label: &str,
    progress: Arc<Mutex<DownloadProgress>>,
    file_index: usize,
    total_files: usize,
) -> Result<(), String> {
    println!("Starting download of {} from {}", file_label, url);
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed for {}: {e}", file_label))?;

    if !resp.status().is_success() {
        return Err(format!("Download failed with status: {}", resp.status()));
    }

    let total_size = resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let temp_path = target_path.with_extension("tmp");
    let mut file = File::create(&temp_path)
        .map_err(|e| format!("Failed to create temporary file: {e}"))?;

    let mut stream = resp.bytes_stream();

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|e| format!("Stream error reading {}: {e}", file_label))?;
        file.write_all(&chunk)
            .map_err(|e| format!("File write error: {e}"))?;
        downloaded += chunk.len() as u64;

        let pct = if total_size > 0 {
            (downloaded as f32 / total_size as f32) * 100.0
        } else {
            0.0
        };

        let overall_pct = ((file_index as f32 - 1.0) / total_files as f32) * 100.0
            + (pct / total_files as f32);

        let mut lock = progress.lock().await;
        lock.current_file = file_label.to_string();
        lock.downloaded_bytes = downloaded;
        lock.total_bytes = total_size;
        lock.percentage = overall_pct;
        lock.status_message = format!(
            "Downloading {} [{}/{}]: {:.1} MB / {:.1} MB ({:.1}%)",
            file_label,
            file_index,
            total_files,
            downloaded as f64 / 1_048_576.0,
            total_size as f64 / 1_048_576.0,
            pct
        );
    }

    file.flush()
        .map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    if target_path.exists() {
        let _ = fs::remove_file(target_path);
    }
    fs::rename(&temp_path, target_path)
        .map_err(|e| format!("Failed to replace final file {}: {e}", target_path.display()))?;

    Ok(())
}

async fn handle_download(
    State(state): State<AppState>,
    payload: Option<Json<DownloadPayload>>,
) -> Json<serde_json::Value> {
    {
        let lock = state.download_state.lock().await;
        if lock.is_downloading {
            return Json(serde_json::json!({
                "success": false,
                "message": "Download is already in progress."
            }));
        }
    }

    let chosen_quant = payload
        .and_then(|p| p.0.quantization)
        .unwrap_or_else(|| get_active_quantization(&state.cache_dir));

    let catalog = get_quantization_catalog();
    let target_opt = catalog
        .iter()
        .find(|q| q.id == chosen_quant)
        .unwrap_or(&catalog[0])
        .clone();
    let opt_name = target_opt.name.clone();

    let progress_arc = state.download_state.clone();
    let cache_dir = state.cache_dir.clone();
    let tts_arc = state.tts.clone();

    tokio::spawn(async move {
        {
            let mut lock = progress_arc.lock().await;
            lock.is_downloading = true;
            lock.error = None;
            lock.percentage = 0.0;
            lock.status_message = format!("Initiating download for {}...", target_opt.name);
        }

        let _ = fs::create_dir_all(&cache_dir);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .unwrap_or_default();

        let model_path = cache_dir.join("0.onnx");
        let voices_path = cache_dir.join("0.bin");
        let model_url = format!(
            "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/{}",
            target_opt.filename
        );

        let voices_present = voices_path.exists()
            && fs::metadata(&voices_path).map(|m| m.len()).unwrap_or(0) > 10_000_000;
        let total_steps = if voices_present { 1 } else { 2 };

        let download_result = async {
            // 1. Download Model ONNX from Hugging Face
            download_file_with_progress(
                &client,
                &model_url,
                &model_path,
                &format!("{} ({})", target_opt.filename, target_opt.name),
                progress_arc.clone(),
                1,
                total_steps,
            )
            .await?;

            // 2. Download Voices pack if missing
            if !voices_present {
                download_file_with_progress(
                    &client,
                    HF_VOICES_URL,
                    &voices_path,
                    "voices.bin (54 Voice Styles)",
                    progress_arc.clone(),
                    2,
                    total_steps,
                )
                .await?;
            }

            Ok::<(), String>(())
        }
        .await;

        match download_result {
            Ok(_) => {
                let _ = fs::write(cache_dir.join("quantization.txt"), &target_opt.id);
                println!(
                    "Download completed successfully for {}. Initializing Kokoro engine...",
                    target_opt.name
                );
                {
                    let mut lock = progress_arc.lock().await;
                    lock.percentage = 100.0;
                    lock.status_message = "Loading Kokoro engine into memory...".to_string();
                }

                match load_engine_from_cache(&cache_dir).await {
                    Ok(engine) => {
                        let mut tts_lock = tts_arc.write().await;
                        *tts_lock = Some(engine);
                        println!("Kokoro engine ready ({})", target_opt.name);

                        let mut lock = progress_arc.lock().await;
                        lock.is_downloading = false;
                        lock.status_message = format!("Model ready: {}", target_opt.name);
                    }
                    Err(e) => {
                        eprintln!("Failed to load Kokoro engine: {e}");
                        let mut lock = progress_arc.lock().await;
                        lock.is_downloading = false;
                        lock.error = Some(format!("Failed to load engine: {e}"));
                    }
                }
            }
            Err(e) => {
                eprintln!("Download failed: {e}");
                let mut lock = progress_arc.lock().await;
                lock.is_downloading = false;
                lock.error = Some(e);
            }
        }
    });

    Json(serde_json::json!({
        "success": true,
        "message": format!("Download started for {}.", opt_name)
    }))
}

fn normalize_text(text: &str) -> String {
    text.replace(['\u{2018}', '\u{2019}', '`', '´'], "'")
        .replace(['\u{201C}', '\u{201D}', '«', '»'], "\"")
        .replace(['\u{2014}', '\u{2013}'], " - ")
        .replace('\u{2026}', "...")
        .replace('\u{00A0}', " ")
        .replace(['\u{200B}', '\u{FEFF}'], "")
}

async fn handle_generate(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let text = payload.text.trim().to_string();
    if text.is_empty() {
        return Json(GenerateResponse {
            success: false,
            duration_seconds: 0.0,
            sample_count: 0,
            sample_rate: SAMPLE_RATE,
            voice: "".into(),
            wav_base64: "".into(),
            latency_ms: 0,
            error: Some("Input text cannot be empty.".into()),
        });
    }

    let voice = payload.voice.unwrap_or_else(|| "af_sky".to_string());
    let speed = payload.speed.unwrap_or(1.0).clamp(0.4, 2.5);
    let pitch = payload.pitch_semitones.unwrap_or(0.0).clamp(-12.0, 12.0);
    let gain = payload.gain.unwrap_or(1.0).clamp(0.2, 2.0);

    let start = Instant::now();

    // Check if engine is loaded, if not try initializing
    let mut engine_guard = state.tts.write().await;
    if engine_guard.is_none() {
        println!("TTS engine not yet initialized. Attempting load...");
        match load_engine_from_cache(&state.cache_dir).await {
            Ok(eng) => {
                *engine_guard = Some(eng);
            }
            Err(e) => {
                return Json(GenerateResponse {
                    success: false,
                    duration_seconds: 0.0,
                    sample_count: 0,
                    sample_rate: SAMPLE_RATE,
                    voice,
                    wav_base64: "".into(),
                    latency_ms: 0,
                    error: Some(format!(
                        "TTS engine not initialized. Please click 'Download Model' first. ({e})"
                    )),
                });
            }
        }
    }

    let normalized_text = normalize_text(&text);
    let engine = engine_guard.as_mut().unwrap();

    let synth_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.synthesize_with_options(&normalized_text, Some(&voice), speed, gain, None)
    }));

    let synth_res = match synth_res {
        Ok(res) => res,
        Err(_) => Err("Inference panic caught safely (check text input characters)".to_string()),
    };

    match synth_res {
        Ok(samples) => {
            let processed_samples = apply_pitch_shift(&samples, pitch);
            let sample_count = processed_samples.len();
            let duration = sample_count as f32 / SAMPLE_RATE as f32;

            match convert_samples_to_wav(&processed_samples, SAMPLE_RATE) {
                Ok(wav_bytes) => {
                    use base64::Engine;
                    let base64_str =
                        base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
                    let latency = start.elapsed().as_millis() as u64;

                    Json(GenerateResponse {
                        success: true,
                        duration_seconds: duration,
                        sample_count,
                        sample_rate: SAMPLE_RATE,
                        voice,
                        wav_base64: base64_str,
                        latency_ms: latency,
                        error: None,
                    })
                }
                Err(e) => Json(GenerateResponse {
                    success: false,
                    duration_seconds: 0.0,
                    sample_count: 0,
                    sample_rate: SAMPLE_RATE,
                    voice,
                    wav_base64: "".into(),
                    latency_ms: 0,
                    error: Some(format!("Failed to encode WAV audio: {e}")),
                }),
            }
        }
        Err(e) => Json(GenerateResponse {
            success: false,
            duration_seconds: 0.0,
            sample_count: 0,
            sample_rate: SAMPLE_RATE,
            voice,
            wav_base64: "".into(),
            latency_ms: 0,
            error: Some(format!("Speech synthesis failed: {e}")),
        }),
    }
}

async fn handle_index() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}

async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = get_cache_directory();
    let _ = fs::create_dir_all(&cache_dir);

    println!("==================================================");
    println!("     Kokoro TTS Studio (Rust + ONNX Runtime)      ");
    println!("==================================================");
    println!("Cache directory: {}", cache_dir.display());

    let initial_engine = match load_engine_from_cache(&cache_dir).await {
        Ok(engine) => {
            println!("Kokoro 82M TTS engine loaded successfully.");
            Some(engine)
        }
        Err(e) => {
            println!("Notice: Kokoro model not found or not initialized yet: {e}");
            println!("You can download it directly from Hugging Face via the web interface.");
            None
        }
    };

    let state = AppState {
        tts: Arc::new(RwLock::new(initial_engine)),
        download_state: Arc::new(Mutex::new(DownloadProgress::default())),
        cache_dir,
    };

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/api/status", get(handle_status))
        .route("/api/voices", get(handle_voices))
        .route("/api/download", post(handle_download))
        .route("/api/generate", post(handle_generate))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let start_port = 7860;
    let mut bound_listener = None;
    let mut bound_port = start_port;

    // ponytail: scan 7860..7880 and detect if already running
    for port in start_port..start_port + 20 {
        let addr = format!("127.0.0.1:{port}");
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(l) => {
                bound_listener = Some(l);
                bound_port = port;
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                if port == start_port {
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_millis(600))
                        .build()
                        .unwrap_or_default();
                    if let Ok(resp) = client.get(format!("http://127.0.0.1:{port}/api/status")).send().await {
                        if resp.status().is_success() {
                            let url = format!("http://localhost:{port}");
                            println!("\nKokoro TTS server is already active on {url}");
                            println!("Opening browser interface...\n");
                            let _ = open::that(&url);
                            println!("Press Enter to close launcher...");
                            let mut buf = String::new();
                            let _ = std::io::stdin().read_line(&mut buf);
                            return Ok(());
                        }
                    }
                }
                println!("Port {port} in use, trying port {}...", port + 1);
            }
            Err(e) => return Err(e.into()),
        }
    }

    let listener = bound_listener
        .ok_or_else(|| format!("Could not bind to any port in range {start_port}..{}", start_port + 20))?;

    let url = format!("http://localhost:{bound_port}");
    println!("\nServer listening on {}", url);
    println!("Opening browser interface...\n");

    // Open browser automatically
    let _ = open::that(&url);

    axum::serve(listener, app).await?;

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run_server().await {
        eprintln!("\n[Error] Server failed to start: {e}\n");
        eprintln!("Press Enter to exit...");
        let mut buf = String::new();
        let _ = std::io::stdin().read_line(&mut buf);
        std::process::exit(1);
    }
}
