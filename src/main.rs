use hound;
use image;
use rusqlite::Connection;
use rustfft::FftPlanner;
use std::fs::read_dir;

mod audio;
mod config;
mod db;
mod hashing;

use audio::{
    compute_spectrogram, extract_significant_peaks, hamming_window, hz_to_bin,
    load_and_prepare_audio,
};
use config::{
    FREQ_BANDS, HOP_SIZE, MAX_TIME_DELTA, MIN_PEAK_FREQ_DISTANCE, MIN_PEAK_TIME_DISTANCE,
    MIN_TIME_DELTA, PEAK_THRESHOLD_FACTOR, TARGET_FANOUT, TARGET_SAMPLE_RATE, WINDOW_SIZE,
};
use db::{
    check_song_exists, get_song_filepath, insert_fingerprints, insert_song_record, query_matches,
    setup_database,
};
use hashing::generate_hashes;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // constants
    const DB_PATH: &str = "src/media/audio_fingerprints.db";
    const AUDIO_DIR: &str = "src/media/wav/";
    const INPUT_FILE: &str = "src/media/recording/recording1.wav";

    const MODE: &str = "train"; // "test" or "train"

    if MODE == "train" {
        let mut conn = setup_database(DB_PATH)?;

        println!("\nScanning directory: {}", AUDIO_DIR);
        let mut audio_files = Vec::new();
        for entry in read_dir(AUDIO_DIR)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "wav") {
                if hound::WavReader::open(&path).is_ok() {
                    audio_files.push(path.to_string_lossy().to_string());
                    println!("Found valid audio file: {}", path.display());
                } else {
                    println!(
                        "Skipping invalid or corrupted audio file: {}",
                        path.display()
                    );
                }
            }
        }
        println!("Found {} audio files.", audio_files.len());

        let mut audios_processed = 0;

        for filepath in &audio_files {
            println!("\n--- Processing: {} ---", filepath);

            let exists_check: Result<i64, rusqlite::Error> = check_song_exists(&conn, &filepath);
            match exists_check {
                Ok(existing_id) => {
                    println!(
                        "Song already exists in the database with ID: {}",
                        existing_id
                    );
                    continue;
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => {
                    println!("Song not found in the database, proceeding to insert.");
                }
                Err(e) => {
                    eprintln!("Error checking DB for song existence: {}", e);
                    return Err(e.into());
                }
            }

            let song_id = match insert_song_record(&mut conn, &filepath) {
                Ok(id) => {
                    println!("Inserted new song record with ID: {}", id);
                    id
                }
                Err(e) => {
                    eprintln!("Error inserting song record: {}", e);
                    continue;
                }
            };
            println!("Using song_id: {}", song_id);

            let hashes = match process_audio_file(
                &filepath,
                FREQ_BANDS,
                WINDOW_SIZE,
                HOP_SIZE,
                TARGET_SAMPLE_RATE,
                MIN_TIME_DELTA,
                MAX_TIME_DELTA,
                TARGET_FANOUT,
            ) {
                Ok(hashes) => {
                    println!("Generated {} hashes for song ID: {}", hashes.len(), song_id);
                    hashes
                }
                Err(e) => {
                    eprintln!("Error processing audio file: {}", e);
                    continue;
                }
            };

            match insert_fingerprints(&mut conn, &hashes, song_id) {
                Ok(_) => {
                    println!(
                        "Inserted {} fingerprints for song ID: {}",
                        hashes.len(),
                        song_id
                    );
                }
                Err(e) => {
                    eprintln!("Error inserting fingerprints: {}", e);
                    continue;
                }
            }
            println!("{} Fingerprints inserted successfully.", hashes.len());

            audios_processed += 1;
        }
        println!(
            "\nProcessed {} audio files, skipped {} files.",
            audios_processed,
            audio_files.len() - audios_processed
        );
        println!("All done!");
    } else if MODE == "test" {
        println!("\n ------ TESTING ------ ");

        let snippet_hashes = process_audio_file(
            INPUT_FILE,
            FREQ_BANDS,
            WINDOW_SIZE,
            HOP_SIZE,
            TARGET_SAMPLE_RATE,
            MAX_TIME_DELTA,
            MIN_TIME_DELTA,
            TARGET_FANOUT,
        )?;
        if snippet_hashes.is_empty() {
            println!("No hashes generated for the test file.");
            return Ok(());
        }
        println!("Generated {} hashes for test file.", snippet_hashes.len());
        let hash_count = snippet_hashes.len() as u32;
        let conn = Connection::open(DB_PATH)?;
        println!("Connected to database.");

        let histogram = query_matches(&conn, &snippet_hashes)?;

        println!("\nAnalyzing matches...");
        let mut matches: Vec<_> = histogram
            .iter()
            .map(|(&song_id, offsets)| {
                let max_count_for_song = offsets.values().max().cloned().unwrap_or(0);
                (song_id, max_count_for_song)
            })
            .filter(|&(_, count)| count > 2)
            .collect();

        matches.sort_by(|a, b| b.1.cmp(&a.1));

        if !matches.is_empty() {
            println!("\nTop matches found:");

            for (i, &(song_id, match_count)) in matches.iter().take(3).enumerate() {
                println!("\nMatch #{}", i + 1);
                println!("Song ID: {}", song_id);
                println!("Match strength: {} matching points", match_count);

                let confidence = (match_count as f64 / hash_count as f64) * 100.0;
                let confidence = confidence.min(100.0);
                println!("Confidence: {:.1}%", confidence);

                let filepath = get_song_filepath(&conn, song_id)?;
                println!("Matched Filepath: {}", filepath);

                let quality = if confidence > 15.0 {
                    "High"
                } else if confidence > 5.0 {
                    "Medium"
                } else {
                    "Low"
                };
                println!("Match quality: {}", quality);
            }
        } else {
            println!("\nNo matches found.");
        }
    } else {
        eprintln!("Invalid mode specified. Use 'index' or 'match'.");
    }

    Ok(())
}

fn process_audio_file(
    filepath: &str,
    freq_bands: &[usize],
    window_size: usize,
    hop_size: usize,
    target_sample_rate: u32,
    max_time_delta: usize,
    min_time_delta: usize,
    target_fanout: usize,
) -> Result<Vec<(u64, usize)>, Box<dyn std::error::Error>> {
    println!("\nLoading and preparing audio file: {}", filepath);
    let samples_vec = load_and_prepare_audio(filepath, target_sample_rate)?;
    let window_coefficients = hamming_window(WINDOW_SIZE);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    let spectrogram =
        compute_spectrogram(samples_vec, window_coefficients, fft, hop_size, window_size);

    let bin_ranges = freq_bands
        .windows(2)
        .map(|band| {
            (
                hz_to_bin(band[0], window_size, target_sample_rate),
                hz_to_bin(band[1], window_size, target_sample_rate),
            )
        })
        .collect::<Vec<_>>();

    /* let bin_ranges: Vec<(usize, usize)> = vec![(0, 5), (5, 80), (80, 160), (160, 320), (320, 640), (640, 1280), (1280, 4096)]; */
    let mut peaks: Vec<(usize, usize)> = Vec::new();

    for (time_slice_index, magnitudes) in spectrogram.iter().enumerate() {
        for &(low_bin, high_bin) in &bin_ranges {
            if high_bin > magnitudes.len() {
                continue;
            }
            let band_slice = &magnitudes[low_bin..high_bin];

            if let Some((max_index_in_band, _)) = band_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                let absolute_bin_index = low_bin + max_index_in_band;
                peaks.push((time_slice_index, absolute_bin_index));
            }
        }
    }

    println!("Found {} peaks.", peaks.len());

    let filtered_peaks = extract_significant_peaks(
        &spectrogram,
        &bin_ranges,
        MIN_PEAK_FREQ_DISTANCE,
        MIN_PEAK_TIME_DISTANCE,
        PEAK_THRESHOLD_FACTOR,
    );

    let hashes = generate_hashes(
        &filtered_peaks,
        max_time_delta,
        min_time_delta,
        target_fanout,
    );
    Ok(hashes)
}

fn _spectrogram_to_image(spectrogram: &Vec<Vec<f64>>, output_path: &str) {
    if spectrogram.is_empty() || spectrogram[0].is_empty() {
        eprintln!("Spectrogram does not exist. Cannot visualize");
        return;
    }

    let width = spectrogram.len();
    let height = spectrogram[0].len();

    let mut min_log_mag = f64::MAX; // goes to min
    let mut max_log_mag = f64::MIN; // goes to max

    for time_slice in spectrogram {
        for &magnitude in time_slice {
            if magnitude < 1e-10 {
                continue;
            }

            let log_mag = (magnitude - 1e-6).log10();
            min_log_mag = min_log_mag.min(log_mag);
            max_log_mag = max_log_mag.max(log_mag);
        }
    }

    let log_mag_range = max_log_mag - min_log_mag;
    let log_mag_range = if log_mag_range < 1e-6 {
        1.0
    } else {
        log_mag_range
    };

    let mut img_buf = image::GrayImage::new(width as u32, height as u32);

    for (t, time_slice) in spectrogram.iter().enumerate() {
        for (f, magnitude) in time_slice.iter().enumerate() {
            let log_mag = (magnitude + 1e-6).log10();
            let scaled_val = (log_mag - min_log_mag) / log_mag_range;
            let intensity = (scaled_val.clamp(0.0, 1.0) * 255.0).round() as u8;

            let x = t as u32;
            let y = (height - 1 - f) as u32;

            img_buf.put_pixel(x, y, image::Luma([intensity]));
        }
    }

    match img_buf.save(output_path) {
        Ok(_) => println!("Spectrogram saved to {}", output_path),
        Err(e) => eprintln!("Error saving spectrogram: {}", e),
    }
}
