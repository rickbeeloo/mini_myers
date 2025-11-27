use ::std::os::raw::c_char;
use edlib_rs::edlib_sys::*;
use edlib_rs::*;
use once_cell::sync::Lazy;
use sassy::profiles::*;

// from sassy bench

pub mod sim_data {
    #[derive(PartialEq, Clone, Copy)]
    pub enum Alphabet {
        Dna,
        Iupac,
    }
}

static EQUALITY_PAIRS: Lazy<Vec<EdlibEqualityPairRs>> = Lazy::new(build_equality_pairs);

pub fn get_edlib_config(k: i32, alphabet: &sim_data::Alphabet) -> EdlibAlignConfigRs<'static> {
    let mut config = EdlibAlignConfigRs::default();
    config.mode = EdlibAlignModeRs::EDLIB_MODE_HW;
    if alphabet == &sim_data::Alphabet::Iupac {
        println!("[EDLIB] Added iupac alphabet");
        config.additionalequalities = &EQUALITY_PAIRS;
    }
    config.k = k;
    config.task = EdlibAlignTaskRs::EDLIB_TASK_PATH;
    config
}

fn build_equality_pairs() -> Vec<EdlibEqualityPairRs> {
    let codes = b"ACGTURYSWKMBDHVNX";
    let mut pairs = Vec::new();
    for &a in codes.iter() {
        for &b in codes.iter() {
            if Iupac::is_match(a, b) {
                // both upper
                pairs.push(EdlibEqualityPairRs {
                    first: a as c_char,
                    second: b as c_char,
                });
                // both lower
                pairs.push(EdlibEqualityPairRs {
                    first: a.to_ascii_lowercase() as c_char,
                    second: b.to_ascii_lowercase() as c_char,
                });
                // first upper, second lower
                pairs.push(EdlibEqualityPairRs {
                    first: a.to_ascii_lowercase() as c_char,
                    second: b as c_char,
                });
                // first lower, second upper
                pairs.push(EdlibEqualityPairRs {
                    first: a as c_char,
                    second: b.to_ascii_lowercase() as c_char,
                });
            }
        }
    }
    pairs
}

pub fn run_edlib(
    query: &[u8],
    target: &[u8],
    edlib_config: &EdlibAlignConfigRs,
) -> EdlibAlignResultRs {
    let edlib_result = edlibAlignRs(query, target, edlib_config);
    assert_eq!(edlib_result.status, EDLIB_STATUS_OK);
    edlib_result
}
