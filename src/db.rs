use rusqlite::{Connection, Error as RusqliteError, params};
use std::{collections::HashMap, error::Error};

pub fn setup_database(db_path: &str) -> Result<Connection, Box<dyn Error>> {
    //println!("Connecting to SQLite database at {}", db_path);
    let conn = Connection::open(db_path)?;
    //println!("Connected to database");
    //println!("Ensuring database schema exists...");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS songs (
            song_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath    TEXT NOT NULL UNIQUE
        )",
        [],
    )?;
    //println!("'songs' table ready.");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS fingerprints (
            hash        INTEGER NOT NULL,
            time_offset INTEGER NOT NULL,
            song_id     INTEGER NOT NULL,
            FOREIGN KEY (song_id) REFERENCES songs (song_id)
        )",
        [],
    )?;
    //println!("'fingerprints' table ready.");

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints (hash)",
        [],
    )?;
    //println!("Index 'idx_hash' ready.");
    //println!("Database schema ready.");
    Ok(conn)
}

pub fn check_song_exists(conn: &Connection, filepath: &str) -> Result<i64, RusqliteError> {
    let mut stmt = conn.prepare_cached("SELECT song_id FROM songs WHERE filepath = ?1")?;
    stmt.query_row(params![filepath], |row| row.get(0))
}

pub fn insert_song_record(conn: &Connection, filepath: &str) -> Result<i64, rusqlite::Error> {
    conn.execute(
        "INSERT INTO songs (filepath) VALUES (?1)",
        params![filepath],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn insert_fingerprints(
    conn: &mut Connection,
    hashes: &[(u64, usize)],
    song_id: i64,
) -> Result<(), rusqlite::Error> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx
            .prepare("INSERT INTO fingerprints (hash, time_offset, song_id) VALUES (?1, ?2, ?3)")?;
        for (hash, time_offset) in hashes {
            stmt.execute(params![*hash as i64, *time_offset as i64, song_id])?;
        }
    }
    tx.commit()?;
    println!("Fingerprints inserted successfully.");
    Ok(())
}

pub fn query_matches(
    conn: &Connection,
    snippet_hashes: &[(u64, usize)],
) -> Result<HashMap<i64, HashMap<i64, u32>>, Box<dyn Error>> {
    //println!("Querying database and building histogram...");

    let mut histogram: HashMap<i64, HashMap<i64, u32>> = HashMap::new();
    let mut stmt =
        conn.prepare_cached("SELECT song_id, time_offset FROM fingerprints WHERE hash = ?1")?;

    let mut _queried_count = 0;
    let mut _total_db_matches = 0;

    for (snippet_hash, snippet_anchor_time) in snippet_hashes.iter() {
        let db_matches = stmt.query_map(params![*snippet_hash as i64], |row| {
            Ok((row.get::<usize, i64>(0)?, row.get::<usize, i64>(1)?)) // (song_id, db_time_offset)
        })?;

        for result in db_matches {
            match result {
                Ok((db_song_id, db_time_offset)) => {
                    _total_db_matches += 1;
                    let offset_diff = db_time_offset - (*snippet_anchor_time as i64);
                    *histogram
                        .entry(db_song_id)
                        .or_default()
                        .entry(offset_diff)
                        .or_default() += 1;
                }
                Err(e) => {
                    eprintln!("Error processing row: {}", e);
                }
            }
        }
        _queried_count += 1;
    }
    //println!("Queried {} snippet hashes, found {} total DB matches.", _queried_count, _total_db_matches);
    Ok(histogram)
}

pub fn get_song_filepath(conn: &Connection, song_id: i64) -> Result<String, RusqliteError> {
    let mut stmt = conn.prepare_cached("SELECT filepath FROM songs WHERE song_id = ?1")?;
    stmt.query_row(params![song_id], |row| row.get(0))
}
