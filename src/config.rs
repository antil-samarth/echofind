// Audio Processing Parameters
pub const TARGET_SAMPLE_RATE: u32 = 8192;
pub const WINDOW_SIZE: usize = 1024;
pub const HOP_SIZE: usize = 64;
pub const FREQ_BANDS: &[usize] = &[5, 80, 160, 320, 640, 1280, 4096];

// Peak Finding Parameters
pub const PEAK_THRESHOLD_FACTOR: f64 = 0.85;
pub const MIN_PEAK_TIME_DISTANCE: usize = 3;
pub const MIN_PEAK_FREQ_DISTANCE: usize = 3;

// Hashing Parameters
pub const MIN_TIME_DELTA: usize = 15;
pub const MAX_TIME_DELTA: usize = 45;
pub const NUM_TARGETS_IN_HASH: usize = 4;
pub const FUZ_FACTOR: usize = 4;

// Matching Parameters
pub const MIN_MATCH_COUNT: u32 = 3;
pub const HIGH_CONFIDENCE_THRESHOLD: f64 = 15.0;
pub const MEDIUM_CONFIDENCE_THRESHOLD: f64 = 5.0;