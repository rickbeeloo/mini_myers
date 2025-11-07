use crate::constant::INVALID_IUPAC;

#[inline(always)]
pub fn get_encoded(c: u8) -> u8 {
    IUPAC_CODE[(c & 0x1F) as usize]
}

// Based on sassy: https://github.com/RagnarGrootKoerkamp/sassy/blob/master/src/profiles/iupac.rs#L258
// todo: add some tests for this table
#[rustfmt::skip]
const IUPAC_CODE: [u8; 32] = {
    // Every char *not* being in the table will be set to invalid IUPAC (255 u8 value)
    let mut t = [INVALID_IUPAC; 32];
    const A: u8 = 1 << 0;
    const C: u8 = 1 << 1;
    const T: u8 = 1 << 2;
    const G: u8 = 1 << 3;

    t[b'A' as usize & 0x1F] = A;
    t[b'C' as usize & 0x1F] = C;
    t[b'T' as usize & 0x1F] = T;
    t[b'U' as usize & 0x1F] = T;
    t[b'G' as usize & 0x1F] = G;
    t[b'N' as usize & 0x1F] = A | C | T | G;

    t[b'R' as usize & 0x1F] = A | G;
    t[b'Y' as usize & 0x1F] = C | T;
    t[b'S' as usize & 0x1F] = G | C;
    t[b'W' as usize & 0x1F] = A | T;
    t[b'K' as usize & 0x1F] = G | T;
    t[b'M' as usize & 0x1F] = A | C;
    t[b'B' as usize & 0x1F] = C | G | T;
    t[b'D' as usize & 0x1F] = A | G | T;
    t[b'H' as usize & 0x1F] = A | C | T;
    t[b'V' as usize & 0x1F] = A | C | G;

    t[b'X' as usize & 0x1F] = 0;

    t
};

#[inline(always)]
pub(crate) fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&c| RC[c as usize]).collect()
}

const RC: [u8; 256] = {
    let mut rc = [0; 256];
    let mut i = 0;
    while i < 256 {
        rc[i] = i as u8;
        i += 1;
    }
    // Standard bases
    rc[b'A' as usize] = b'T';
    rc[b'C' as usize] = b'G';
    rc[b'T' as usize] = b'A';
    rc[b'G' as usize] = b'C';
    rc[b'a' as usize] = b't';
    rc[b'c' as usize] = b'g';
    rc[b't' as usize] = b'a';
    rc[b'g' as usize] = b'c';
    // IUPAC ambiguity codes
    rc[b'R' as usize] = b'Y'; // A|G -> T|C
    rc[b'Y' as usize] = b'R'; // C|T -> G|A
    rc[b'S' as usize] = b'S'; // G|C -> C|G
    rc[b'W' as usize] = b'W'; // A|T -> T|A
    rc[b'K' as usize] = b'M'; // G|T -> C|A
    rc[b'M' as usize] = b'K'; // A|C -> T|G
    rc[b'B' as usize] = b'V'; // C|G|T -> G|C|A
    rc[b'D' as usize] = b'H'; // A|G|T -> T|C|A
    rc[b'H' as usize] = b'D'; // A|C|T -> T|G|A
    rc[b'V' as usize] = b'B'; // A|C|G -> T|G|C
    rc[b'N' as usize] = b'N'; // A|C|G|T -> T|G|C|A
    rc[b'X' as usize] = b'X';
    // Lowercase versions
    rc[b'r' as usize] = b'y';
    rc[b'y' as usize] = b'r';
    rc[b's' as usize] = b's';
    rc[b'w' as usize] = b'w';
    rc[b'k' as usize] = b'm';
    rc[b'm' as usize] = b'k';
    rc[b'b' as usize] = b'v';
    rc[b'd' as usize] = b'h';
    rc[b'h' as usize] = b'd';
    rc[b'v' as usize] = b'b';
    rc[b'n' as usize] = b'n';
    rc[b'x' as usize] = b'x';
    rc
};
