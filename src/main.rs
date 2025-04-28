use hound;
/* This is a project about audio fingerprinting */
fn main() {
    println!("Hello, world!");
    let reader = hound::WavReader::open("src/media/wav/01_Genesis.wav").unwrap();
    println!("file name - 01_Genesis.wav");
    let spec = reader.spec();
    println!("Sample rate: {}", spec.sample_rate);
    println!("Channels: {}", spec.channels);
    println!("Bits per sample: {}", spec.bits_per_sample);
    println!("Sample format: {:?}", spec.sample_format);
    println!("Total samples: {}", reader.len());
    println!(
        "Duration: {} seconds",
        reader.duration() as f32 / spec.sample_rate as f32
    );
}
