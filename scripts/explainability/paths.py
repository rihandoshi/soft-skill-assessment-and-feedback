"""Single source of truth for where the explainability pipeline reads and writes.

results/
  README.md                          index of everything below
  audio_score_models/                earlier score-prediction experiments (scripts/legacy/)
    handcrafted_improvements/  ssl_layer_sweep/  vad_pause/  repeated_cv_v1/
  explainability/
    README.md                        stage index: what each folder answers, where to start reading
    stage0_confound_audit/           report.md, notes.md, tables/, plots/
    stage1_descriptors/              extraction_report.md, verification_report.md, descriptor_registry.json,
                                     tables/, verification/, listening_check/, qc_plots/
    stage2_reference/                report.md, percentile_tables.json, tables/, evidence_examples/
    stage3_recruitview_validation/   report.md, notes.md, decisions.csv/.json, bands.json, tables/, plots/
    stage4_asr/                      report.md, words.csv, descriptor_registry.json, tables/, listening_check/, qc_plots/
    stage5_ser/                      report.md, notes.md, model_agreement.md, descriptor_registry.json, tables/
    stage6_quality/                  report.md, quality_scores.csv, quality_gate.json, tables/
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Earlier score-prediction experiments (inputs to the explainability work)
SCORE_MODELS = RESULTS / "audio_score_models"
HANDCRAFTED = SCORE_MODELS / "handcrafted_improvements"
HANDCRAFTED_FEATURES = HANDCRAFTED / "features"
HANDCRAFTED_DIMS = HANDCRAFTED / "feature_dimensions.csv"
VAD_FEATURES = SCORE_MODELS / "vad_pause" / "features"
REPEATED_CV = SCORE_MODELS / "repeated_cv_v1"
CV_SPLITS = REPEATED_CV / "cv_splits.json"

EXPL = RESULTS / "explainability"

# Stage 0
S0 = EXPL / "stage0_confound_audit"
S0_TABLES = S0 / "tables"
S0_PLOTS = S0 / "plots"

# Stage 1
S1 = EXPL / "stage1_descriptors"
S1_TABLES = S1 / "tables"
S1_SENTENCES = S1_TABLES / "sentence_descriptors.csv"
S1_SEGMENTS = S1_TABLES / "segment_descriptors.csv"
S1_PAUSES = S1_TABLES / "pause_events.csv"
PITCH_SETTINGS = S1_TABLES / "speaker_pitch_settings.csv"
S1_REGISTRY = S1 / "descriptor_registry.json"
EXTRACTION_REPORT = S1 / "extraction_report.md"
VERIFICATION_REPORT = S1 / "verification_report.md"
S1_VERIFICATION = S1 / "verification"
LISTENING = S1 / "listening_check"
QC_PLOTS = S1 / "qc_plots"

# Stage 2
S2 = EXPL / "stage2_reference"
S2_TABLES = S2 / "tables"
PERCENTILE_TABLES = S2 / "percentile_tables.json"
S2_REPORT = S2 / "report.md"
EVIDENCE_EXAMPLES = S2 / "evidence_examples"

# Stage 3
S3 = EXPL / "stage3_recruitview_validation"
S3_TABLES = S3 / "tables"
S3_PLOTS = S3 / "plots"
S3_DECISIONS = S3 / "decisions.json"
S3_BANDS = S3 / "bands.json"

# Stage 4
S4 = EXPL / "stage4_asr"
S4_TABLES = S4 / "tables"
WORDS = S4 / "words.csv"
S4_SENTENCES = S4_TABLES / "sentence_descriptors.csv"
S4_SEGMENTS = S4_TABLES / "segment_descriptors.csv"
S4_PAUSES = S4_TABLES / "pause_events.csv"
S4_REGISTRY = S4 / "descriptor_registry.json"
S4_REPORT = S4 / "report.md"
S4_LISTENING = S4 / "listening_check"
S4_QC_PLOTS = S4 / "qc_plots"

# Stage 5 (SER): answer-level columns are merged into the descriptor tables by descriptor_tables.py
S5 = EXPL / "stage5_ser"
S5_TABLES = S5 / "tables"
S5_SEGMENTS = S5_TABLES / "segment_ser.csv"
S5_REGISTRY = S5 / "descriptor_registry.json"
S5_REPORT = S5 / "report.md"
CACHE_SER = ROOT / "outputs" / "audio" / "cache" / "stage5_ser"


def s5_windows(model: str = "audeering") -> Path:
    return S5_TABLES / f"windows_{model}.csv"


# Stage 6 (recording-quality gate)
S6 = EXPL / "stage6_quality"
S6_TABLES = S6 / "tables"
QUALITY_SCORES = S6 / "quality_scores.csv"
QUALITY_GATE = S6 / "quality_gate.json"
S6_REPORT = S6 / "report.md"
CACHE_QUALITY = ROOT / "outputs" / "audio" / "cache" / "stage6_quality"

# Stage 7 (MIT Interview Dataset: 138 interviews, 69 students, pre / post coaching)
MIT = ROOT / "Datasets" / "MIT" / "MIT_INTERVIEW_DATASET"
MIT_AUDIO = MIT / "Audio"
MIT_TRANSCRIPTS = MIT / "Labels" / "interview_transcripts_by_turkers.csv"
MIT_SCORES = MIT / "Labels" / "turker_scores_full_interview.csv"
MIT_PROSODY = MIT / "Prosody" / "prosodic_features.csv"
S7 = EXPL / "stage7_mit_validation"
S7_TABLES = S7 / "tables"
S7_REPORT = S7 / "report.md"
S7_RELIABILITY = S7_TABLES / "label_reliability.csv"
CACHE_MIT = ROOT / "outputs" / "audio" / "cache" / "stage7_mit"
MIT_CLIPS = CACHE_MIT / "answers"          # 16-kHz mono answer clips cut from the interviews

# Stage 8: per-answer detectors (hesitation, confidence) -> levels vs other people's answers
S8 = EXPL / "stage8_detectors"
S8_TABLES = S8 / "tables"
S8_REPORT = S8 / "report.md"
CONFIDENCE_PRED = S8_TABLES / "confidence_predictions.csv"      # RecruitView + MIT, one row per answer
DISFLUENCY_WINDOWS = S8_TABLES / "disfluency_windows.csv"       # per 3-s window class probabilities
DISFLUENCY_ANSWERS = S8_TABLES / "disfluency_answers.csv"       # per answer shares
CRISPER_WORDS = S8_TABLES / "crisper_words.csv"                 # CrisperWhisper verbatim words (RecruitView + MIT)
CRISPER_ANSWERS = S8_TABLES / "crisper_answers.csv"             # per answer filler / repetition counts
ANSWER_LEVELS = S8_TABLES / "answer_levels.csv"
LEVEL_CUTS = S8 / "level_cuts.json"
CACHE_S8 = ROOT / "outputs" / "audio" / "cache" / "stage8_detectors"
HF_CRISPER = ROOT / "outputs" / "audio" / "cache" / "hf_crisperwhisper"      # weights only (safetensors + json)
HF_DISFLUENCY = ROOT / "outputs" / "audio" / "cache" / "hf_vocametrix_stutter"

# Stage 9: evidence JSON v2 (per-answer levels, no numbers in the wording) -> LLM feedback
S9 = EXPL / "stage9_feedback"
EVIDENCE_V2 = S9 / "evidence_v2.jsonl"
EVIDENCE_V2_EXAMPLES = S9 / "evidence_v2_examples"


# Which descriptor tables Stages 2, 3 and the evidence JSON read:
#   "asr"        Stage 4 tables: real sentences + word counts from verbatim ASR (default once Stage 4 exists)
#   "transcript" Stage 1 tables: RecruitView transcript lines (cleaned text, 1-s stamps)
DESCRIPTOR_SOURCE = "asr" if S4_SEGMENTS.exists() else "transcript"
_TABLES = {"asr": (S4_SEGMENTS, S4_SENTENCES, S4_PAUSES, S4_REGISTRY),
           "transcript": (S1_SEGMENTS, S1_SENTENCES, S1_PAUSES, S1_REGISTRY)}
SEGMENTS, SENTENCES, PAUSES, REGISTRY = _TABLES[DESCRIPTOR_SOURCE]

# Gitignored caches
CACHE_ASR = ROOT / "outputs" / "audio" / "cache" / "stage4_asr"
CACHE_DESCRIPTORS = ROOT / "outputs" / "audio" / "cache" / "delivery_descriptors"
CACHE_WHISPER = ROOT / "outputs" / "audio" / "cache" / "stage1_verification" / "whisper"
WAVLM_OOF = ROOT / "outputs" / "explainability" / "repeated_cv_v1_oof_predictions.csv"
