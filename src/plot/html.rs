//! Dependency-free interactive HTML wrappers for SVG plots.

use super::svg::{self, SvgOptions};
use crate::{Explanation, Result, ShapError};

/// Renders global importance as a standalone interactive HTML document.
pub fn global_bar(explanation: &Explanation, options: &SvgOptions) -> Result<String> {
    document(
        "Global SHAP importance",
        &svg::global_bar(explanation, options)?,
    )
}

/// Renders a local waterfall as a standalone interactive HTML document.
pub fn waterfall(
    explanation: &Explanation,
    sample: usize,
    output: usize,
    options: &SvgOptions,
) -> Result<String> {
    document(
        "SHAP waterfall",
        &svg::waterfall(explanation, sample, output, options)?,
    )
}

/// Wraps a trusted `shap-rs` SVG in a self-contained interactive document.
pub fn document(title: &str, svg: &str) -> Result<String> {
    if title.trim().is_empty() || !svg.trim_start().starts_with("<svg") || !svg.ends_with("</svg>")
    {
        return Err(ShapError::InvalidConfiguration(
            "interactive HTML requires a title and complete SVG".into(),
        ));
    }
    let title = escape(title);
    Ok(format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>body{{font-family:system-ui,sans-serif;margin:1rem;background:#fff;color:#222}}.controls{{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}}button{{padding:.45rem .7rem}}figure{{margin:1rem 0;overflow:auto}}svg [data-tooltip],svg rect,svg circle,svg polygon{{cursor:help}}#tooltip{{position:fixed;display:none;pointer-events:none;background:#111;color:#fff;padding:.35rem .5rem;border-radius:.25rem;font-size:.85rem}}.contrast{{filter:contrast(1.4) saturate(1.2)}}.legend span{{margin-right:1rem}}.positive::before,.negative::before{{content:"";display:inline-block;width:.8rem;height:.8rem;margin-right:.3rem}}.positive::before{{background:#e53935}}.negative::before{{background:#1e88e5}}</style></head><body><main><h1>{title}</h1><div class="controls" aria-label="Plot controls"><button id="contrast" type="button" aria-pressed="false">High contrast</button><button id="download" type="button">Download SVG</button><span id="status" role="status" aria-live="polite"></span></div><div class="legend" aria-label="Contribution legend"><span class="positive">Positive contribution</span><span class="negative">Negative contribution</span></div><figure aria-labelledby="caption">{svg}<figcaption id="caption">{title}. Focus controls with the keyboard; hover plot marks for element details.</figcaption></figure><div id="tooltip" role="tooltip"></div></main><script>(()=>{{const s=document.querySelector('svg'),t=document.querySelector('#tooltip'),status=document.querySelector('#status'),contrast=document.querySelector('#contrast');contrast.onclick=()=>{{const on=s.classList.toggle('contrast');contrast.setAttribute('aria-pressed',on);status.textContent=on?'High contrast enabled':'High contrast disabled'}};document.querySelector('#download').onclick=()=>{{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s.outerHTML],{{type:'image/svg+xml'}}));a.download='shap.svg';a.click();URL.revokeObjectURL(a.href);status.textContent='SVG download prepared'}};s.addEventListener('pointermove',e=>{{const mark=e.target.closest('rect,circle,polygon');if(!mark)return;t.textContent=mark.dataset.tooltip||mark.getAttribute('aria-label')||mark.tagName.toLowerCase()+' plot mark';t.style.display='block';t.style.left=e.clientX+12+'px';t.style.top=e.clientY+12+'px'}});s.addEventListener('pointerleave',()=>t.style.display='none')}})();</script></body></html>"#
    ))
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    #[test]
    fn creates_accessible_standalone_interactive_output() {
        let explanation = Explanation::new(
            Array3::from_shape_vec((1, 2, 1), vec![1.0, -2.0]).unwrap(),
            array![[0.0]],
            array![[3.0, 4.0]],
        )
        .unwrap();
        let html = global_bar(&explanation, &SvgOptions::default()).unwrap();
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("aria-live=\"polite\""));
        assert!(html.contains("Download SVG"));
        assert!(html.contains("<svg"));
    }
}
