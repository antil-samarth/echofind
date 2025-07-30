use std::collections::HashMap;

pub fn find_best_matches(
    histogram: &HashMap<i64, HashMap<i64, u32>>,
    snippet_hash_count: usize,
) -> Vec<(i64, u32, f64)> {
    // (song_id, peak_count, confidence)
    //println!("\nAnalyzing histograms...");

    let mut matches: Vec<(i64, u32)> = histogram // Vec of (song_id, max_offset_count)
        .iter()
        .filter_map(|(&song_id, offsets_map)| {
            // Find the maximum count within this song's offset histogram
            offsets_map
                .values()
                .max()
                .map(|&max_count| (song_id, max_count))
        })
        .filter(|&(_, count)| count > 2) 
        .collect();

    // Sort by the peak offset count (match strength) in descending order
    matches.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // Calculate confidence
    let results_with_confidence: Vec<(i64, u32, f64)> = matches
        .into_iter()
        .map(|(song_id, peak_count)| {
            let confidence = if snippet_hash_count > 0 {
                (peak_count as f64 / snippet_hash_count as f64) * 100.0
            } else {
                0.0 // Avoid division by zero
            };
            (song_id, peak_count, confidence.min(100.0)) // Clamp confidence
        })
        .collect();

    results_with_confidence
}