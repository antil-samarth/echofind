use clap::Parser;
use hound;
use rusqlite::Connection;
use rustfft::FftPlanner;
use std::fs;
use std::path::PathBuf;

mod audio;
mod config;
mod db;
mod hashing;
mod matching;

use audio::{
    compute_spectrogram, extract_significant_peaks, hamming_window, hz_to_bin,
    load_and_prepare_audio,
};
use config::*;
use db::{
    check_song_exists, get_song_filepath, insert_fingerprints, insert_song_record, query_matches,
    setup_database,
};
use hashing::generate_constellation_hashes;
use matching::{find_best_matches, get_match_quality};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Add songs to the database
    Train {
        /// Path to the directory containing WAV files to train on
        #[arg(short, long)]
        audio_dir: PathBuf,

        /// Path to the database file
        #[arg(short, long, default_value = "src/media/audio_fingerprints.db")]
        db_path: PathBuf,
    },
    /// Match a song from a recording
    Test {
        /// Path to the audio file to test
        #[arg(short, long)]
        input_file: PathBuf,

        /// Path to the database file
        #[arg(short, long, default_value = "src/media/audio_fingerprints.db")]
        db_path: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Commands::Train { audio_dir, db_path } => {
            let mut conn = setup_database(&db_path.to_string_lossy())?;

            println!("\nScanning directory: {}", audio_dir.display());
            let mut audio_files = Vec::new();
            for entry in fs::read_dir(audio_dir)? {
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

                if let Ok(existing_id) = check_song_exists(&conn, &filepath) {
                    println!(
                        "Song already exists in the database with ID: {}",
                        existing_id
                    );
                    continue;
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

                let hashes = match process_audio_file(&filepath) {
                    Ok(hashes) => {
                        println!("Generated {} hashes for song ID: {}", hashes.len(), song_id);
                        hashes
                    }
                    Err(e) => {
                        eprintln!("Error processing audio file: {}", e);
                        continue;
                    }
                };

                if let Err(e) = insert_fingerprints(&mut conn, &hashes, song_id) {
                    eprintln!("Error inserting fingerprints: {}", e);
                    continue;
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
        }
        Commands::Test { input_file, db_path } => {
            println!("\n ------ TESTING ------ ");

            let snippet_hashes = process_audio_file(&input_file.to_string_lossy())?;
            if snippet_hashes.is_empty() {
                println!("No hashes generated for the test file.");
                return Ok(());
            }
            println!("Generated {} hashes for test file.", snippet_hashes.len());
            let hash_count = snippet_hashes.len();
            let conn = Connection::open(&db_path)?;
            println!("Connected to database.");

            let histogram = query_matches(&conn, &snippet_hashes)?;

            let matches_with_confidence = find_best_matches(&histogram, hash_count);

            if !matches_with_confidence.is_empty() {
                println!("\nTop matches found:");

                for (i, &(song_id, match_count, confidence)) in
                    matches_with_confidence.iter().take(3).enumerate()
                {
                    println!("\nMatch #{}", i + 1);
                    println!("Song ID: {}", song_id);
                    println!("Match strength: {} matching points", match_count);
                    println!("Confidence: {:.1}%", confidence);

                    let filepath = get_song_filepath(&conn, song_id)?;
                    println!("Matched Filepath: {}", filepath);

                    let quality = get_match_quality(confidence);
                    println!("Match quality: {}", quality);
                }
            } else {
                println!("\nNo matches found.");
            }
        }
    }

    Ok(())
}

fn process_audio_file(
    filepath: &str,
) -> Result<Vec<(u64, usize)>, Box<dyn std::error::Error>> {
    println!("\nLoading and preparing audio file: {}", filepath);
    let samples_vec = load_and_prepare_audio(filepath, TARGET_SAMPLE_RATE)?;
    let window_coefficients = hamming_window(WINDOW_SIZE);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    let spectrogram =
        compute_spectrogram(samples_vec, window_coefficients, fft, HOP_SIZE, WINDOW_SIZE);

    let bin_ranges = FREQ_BANDS
        .windows(2)
        .map(|band| {
            (
                hz_to_bin(band[0], WINDOW_SIZE, TARGET_SAMPLE_RATE),
                hz_to_bin(band[1], WINDOW_SIZE, TARGET_SAMPLE_RATE),
            )
        })
        .collect::<Vec<_>>();

    let filtered_peaks = extract_significant_peaks(
        &spectrogram,
        &bin_ranges,
        MIN_PEAK_FREQ_DISTANCE,
        MIN_PEAK_TIME_DISTANCE,
        PEAK_THRESHOLD_FACTOR,
    );

    let hashes = generate_constellation_hashes(
        &filtered_peaks,
        MIN_TIME_DELTA,
        MAX_TIME_DELTA,
    );
    Ok(hashes)
}