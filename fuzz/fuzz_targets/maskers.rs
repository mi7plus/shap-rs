#![no_main]
use libfuzzer_sys::fuzz_target;
use shap_rs::{FixedMasker, ImageMasker, TextMasker};

fuzz_target!(|data: &[u8]| {
    if let Ok(value) = serde_json::from_slice::<FixedMasker>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<ImageMasker>(data) { let _ = value.validate(); }
    if let Ok(value) = serde_json::from_slice::<TextMasker>(data) { let _ = value.validate(); }
});
