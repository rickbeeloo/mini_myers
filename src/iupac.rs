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
