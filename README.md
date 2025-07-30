# EchoFind: Music Recognition in Rust

[![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Database: SQLite](https://img.shields.io/badge/database-SQLite-blue.svg)](https://sqlite.org/index.html)

A project implementing the core principles of audio fingerprinting and recognition, similar to Shazam or Google Pixel's "Now Playing" feature, built using Rust and SQLite.

## Motivation & Background

This project was inspired by the impressive "Now Playing" feature on my Google Pixel phone. Its ability to quickly and accurately identify songs playing in the background, seemingly offline, and without needing long samples, sparked curiosity about the underlying technology.

The primary goals for building EchoFind are:

1.  **Explore Audio Fingerprinting:** To understand and implement the algorithms that make robust song recognition possible.
2.  **Showcase Language Versatility:** To dive into **Rust**, a modern, performant systems programming language known for its safety, speed, and growing ecosystem. This project serves as a practical application to learn and demonstrate proficiency in Rust.
3.  **Utilize Widely-Used Technologies:** To gain hands-on experience with **SQLite3**, a lightweight, embedded, and extensively used database engine, managing the storage and retrieval of audio fingerprints.

## Features

*   **Audio Loading:** Reads standard WAV audio files.
*   **Signal Processing:** Converts stereo to mono and downsamples audio for efficiency.
*   **Spectrogram Generation:** Computes the Short-Time Fourier Transform (STFT) using `rustfft` to analyze frequency content over time.
*   **Peak Finding:** Identifies prominent time-frequency peaks in the spectrogram, which serve as audio landmarks.
*   **Constellation Hashing:** Implements a robust hashing strategy based on "constellations" of an anchor peak and multiple target peaks.
*   **Database Storage:** Stores generated fingerprints in an SQLite database using `rusqlite`.
*   **Song Matching:** Compares fingerprints from an audio snippet against the database to find the most likely match.
*   **Command-Line Interface:** A user-friendly CLI built with `clap` for training the database and matching songs.

## Technology Stack

*   **Language:** [Rust](https://www.rust-lang.org/) (Stable toolchain)
*   **Database:** [SQLite](https://sqlite.org/index.html)
*   **Core Crates:**
    *   `hound`: For reading WAV audio files.
    *   `rustfft`: For Fast Fourier Transform calculations.
    *   `rusqlite`: For SQLite database interaction.
    *   `clap`: For command-line argument parsing.

## Algorithm Overview

1.  **Preprocessing:** Audio is loaded, converted to mono, and downsampled (e.g., to 8192 Hz) to reduce computational load while retaining essential frequencies.
2.  **STFT:** The audio is divided into short, overlapping windows. The Fast Fourier Transform (FFT) is applied to each window to get the frequency spectrum for that time segment. This creates a spectrogram.
3.  **Peak Finding:** For each time slice in the spectrogram, the algorithm identifies frequency peaks with the highest energy within predefined frequency bands. These `(time, frequency)` points form a "constellation map."
4.  **Constellation Hashing:** For each "anchor" peak, the algorithm finds a set of subsequent "target" peaks within a defined time window. A hash is then generated that encodes the anchor's frequency and the frequencies and time deltas of the targets. This creates a robust and specific fingerprint.
5.  **Storage:** Each generated `(hash, anchor_time_offset, song_id)` tuple is stored in an indexed SQLite database.
6.  **Matching:** Hashes are generated for an unknown audio snippet and queried against the database. The final identification relies on finding a statistically significant number of matching hashes with consistent time offsets.

## Usage

### Training

To add songs to the fingerprint database, use the `train` command:

```bash
cargo run -- train --audio-dir path/to/your/audio
```

This will scan the specified directory for `.wav` files, generate fingerprints for them, and store them in the database.

### Matching

To identify a song from a recording, use the `test` command:

```bash
cargo run -- test --input-file path/to/your/recording.wav
```

This will generate fingerprints for the input file and compare them against the database to find a match.

## Project Structure

```
echofind/
├── mp3_to_wav.bat   # Script to convert MP3 audio files to WAV
├── Cargo.toml       # Project manifest and dependencies
├── README.md        # This file
├── src/
│   ├── main.rs      # Main application logic and CLI
│   ├── audio.rs     # Audio processing functions
│   ├── config.rs    # Configuration constants
│   ├── db.rs        # Database interaction functions
│   ├── hashing.rs   # Fingerprint hashing functions
│   ├── matching.rs  # Song matching logic
│   └── media/       # Sample audio files and database
└── target/          # Build artifacts
```

## Future Enhancements

*   Implement real-time microphone input using `cpal`.
*   Support for more audio formats (e.g., using `symphonia`).
*   Explore more advanced/robust peak finding algorithms.
*   Performance optimizations for fingerprinting and matching.
*   Add basic song metadata handling (reading tags, storing title/artist).
*   Add a graphical user interface (GUI).