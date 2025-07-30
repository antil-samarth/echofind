use std::collections::HashMap;
use crate::config::{MIN_MATCH_COUNT, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD};

pub fn find_best_matches(
    histogram: &HashMap<i64, HashMap<i64, u32>>,
    hash_count: usize,
) -> Vec<(i64, u32, f64)> {
    let mut matches: Vec<_> = histogram
        .iter()
        .map(|(&song_id, offsets)| {
            let max_count_for_song = offsets.values().max().cloned().unwrap_or(0);
            (song_id, max_count_for_song)
        })
        .filter(|&(_, count)| count >= MIN_MATCH_COUNT)
        .collect();

    matches.sort_by(|a, b| b.1.cmp(&a.1));

    matches
        .into_iter()
        .map(|(song_id, match_count)| {
            let confidence = (match_count as f64 / hash_count as f64) * 100.0;
            let confidence = confidence.min(100.0);
            (song_id, match_count, confidence)
        })
        .collect()
}

pub fn get_match_quality(confidence: f64) -> &'static str {
    if confidence > HIGH_CONFIDENCE_THRESHOLD {
        "High"
    } else if confidence > MEDIUM_CONFIDENCE_THRESHOLD {
        "Medium"
    } else {
        "Low"
    }
}
