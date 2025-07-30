use image::{Rgb, RgbImage};
use std::f64;

fn spectrogram_to_image(spectrogram: &Vec<Vec<f64>>, output_path: &str) {
    if spectrogram.is_empty() || spectrogram[0].is_empty() {
        println!("Spectrogram is empty, cannot visualize.");
        return;
    }

    let width = spectrogram.len();
    let height = spectrogram[0].len();

    let mut min_log_mag = f64::MAX;
    let mut max_log_mag = f64::MIN;
    for time_slice in spectrogram {
        for &magnitude in time_slice {
            if magnitude < 1e-10 {
                continue;
            };
            let log_mag = (magnitude + 1e-6).log10();
            min_log_mag = min_log_mag.min(log_mag);
            max_log_mag = max_log_mag.max(log_mag);
        }
    }
    println!("Log Mag Range: {} to {}", min_log_mag, max_log_mag);
    let log_mag_range = max_log_mag - min_log_mag;
    let log_mag_range = if log_mag_range < 1e-6 {
        1.0
    } else {
        log_mag_range
    };

    let mut imgbuf = image::GrayImage::new(width as u32, height as u32);

    for (t, time_slice) in spectrogram.iter().enumerate() {
        for (f, &magnitude) in time_slice.iter().enumerate() {
            let log_mag = (magnitude + 1e-6).log10();
            let scaled_val = (log_mag - min_log_mag) / log_mag_range;
            let intensity = (scaled_val.clamp(0.0, 1.0) * 255.0).round() as u8;

            let x = t as u32;
            let y = (height - 1 - f) as u32;

            imgbuf.put_pixel(x, y, image::Luma([intensity]));
        }
    }

    match imgbuf.save(output_path) {
        Ok(_) => println!("Spectrogram saved to {}", output_path),
        Err(e) => eprintln!("Error saving spectrogram: {}", e),
    }
}

fn visualize_spectrogram_with_peaks(
    spectrogram: &Vec<Vec<f64>>,
    peaks: &[(usize, usize)],
    bin_ranges: &[(usize, usize)], // <-- Pass bin_ranges
    output_path: &str,
) {
    const BAND_COLORS: &[Rgb<u8>] = &[
        Rgb([0, 0, 255]),   // Band 0: 5-80 Hz (Blue)
        Rgb([0, 255, 255]), // Band 1: 80-160 Hz (Cyan)
        Rgb([0, 255, 0]),   // Band 2: 160-320 Hz (Green)
        Rgb([255, 255, 0]), // Band 3: 320-640 Hz (Yellow)
        Rgb([255, 165, 0]), // Band 4: 640-1280 Hz (Orange)
        Rgb([255, 0, 0]),   // Band 5: 1280-4096 Hz (Red)
    ];
    if spectrogram.is_empty() || spectrogram[0].is_empty() {
        println!("Spectrogram is empty, cannot visualize.");
        return;
    }

    let width = spectrogram.len();
    let height = spectrogram[0].len();

    // --- Find min/max log magnitude for scaling (same as before) ---
    let mut min_log_mag = f64::MAX;
    let mut max_log_mag = f64::MIN;
    for time_slice in spectrogram {
        for &magnitude in time_slice {
            if magnitude < 1e-10 {
                continue;
            };
            let log_mag = (magnitude + 1e-6).log10();
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

    // --- Create an RGB image ---
    let mut imgbuf = RgbImage::new(width as u32, height as u32);

    // --- Draw the grayscale spectrogram background ---
    for (t, time_slice) in spectrogram.iter().enumerate() {
        for (f, &magnitude) in time_slice.iter().enumerate() {
            let log_mag = (magnitude + 1e-6).log10();
            let scaled_val = (log_mag - min_log_mag) / log_mag_range;
            let intensity = (scaled_val.clamp(0.0, 1.0) * 255.0).round() as u8;
            let x = t as u32;
            let y = (height - 1 - f) as u32;
            imgbuf.put_pixel(x, y, Rgb([intensity, intensity, intensity]));
        }
    }

    // --- Draw the peaks on top using band-specific colors ---
    let default_peak_color = Rgb([255u8, 255u8, 255u8]); // White for fallback

    for &(time_idx, freq_idx) in peaks {
        // Find which band this peak's frequency index belongs to
        let mut band_index_for_peak = None;
        for (band_idx, &(low_bin, high_bin)) in bin_ranges.iter().enumerate() {
            if freq_idx >= low_bin && freq_idx < high_bin {
                band_index_for_peak = Some(band_idx);
                break; // Found the band
            }
        }

        // Select the color based on the band index
        let peak_color = match band_index_for_peak {
            // Get color from the provided list, fallback to default if index is out of bounds
            Some(idx) => *BAND_COLORS.get(idx).unwrap_or(&default_peak_color),
            None => default_peak_color, // Use default if peak didn't fall in any range
        };

        // Draw the pixel (with bounds check)
        if time_idx < width && freq_idx < height {
            let x = time_idx as u32;
            let y = (height - 1 - freq_idx) as u32; // Invert y-axis
            imgbuf.put_pixel(x, y, peak_color);
        }
    }

    // --- Save the image ---
    match imgbuf.save(output_path) {
        Ok(_) => println!("Spectrogram with peaks saved to {}", output_path),
        Err(e) => eprintln!("Error saving spectrogram with peaks: {}", e),
    }
}
