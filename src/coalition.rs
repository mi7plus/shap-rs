pub fn all(n: usize) -> impl Iterator<Item = u64> {
    0..if n < 63 { 1u64 << n } else { 0 }
}
pub fn members(mask: u64, n: usize) -> Vec<bool> {
    (0..n).map(|i| mask & (1 << i) != 0).collect()
}
pub fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1, |v, i| v * (n - i) as u128 / (i + 1) as u128)
}
pub fn kernel_weight(m: usize, s: usize) -> f64 {
    if m == 0 || s > m {
        return 0.0;
    }
    if s == 0 || s == m {
        1e6
    } else {
        (m - 1) as f64 / (binomial(m, s) as f64 * s as f64 * (m - s) as f64)
    }
}

/// Derives a stable RNG seed from a base seed and one sample's exact bits.
pub(crate) fn sample_seed(seed: u64, sample: ndarray::ArrayView1<'_, f64>) -> u64 {
    fn mix(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
    sample
        .iter()
        .fold(mix(seed), |state, value| mix(state ^ mix(value.to_bits())))
}
