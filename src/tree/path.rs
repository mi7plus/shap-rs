#[derive(Clone, Copy, Debug)]
pub(crate) struct PathElement {
    pub feature: usize,
    pub zero: f64,
    pub one: f64,
    pub weight: f64,
}
pub(crate) fn extend(path: &mut Vec<PathElement>, feature: usize, zero: f64, one: f64) {
    let depth = path.len();
    path.push(PathElement {
        feature,
        zero,
        one,
        weight: if depth == 0 { 1.0 } else { 0.0 },
    });
    if depth > 0 {
        for i in (0..depth).rev() {
            path[i + 1].weight += one * (i + 1) as f64 / (depth + 1) as f64 * path[i].weight;
            path[i].weight *= zero * (depth - i) as f64 / (depth + 1) as f64;
        }
    }
}
pub(crate) fn unwind(path: &mut Vec<PathElement>, index: usize) {
    let depth = path.len() - 1;
    let one = path[index].one;
    let zero = path[index].zero;
    let mut next = path[depth].weight;
    if one != 0.0 {
        for i in (0..depth).rev() {
            let tmp = path[i].weight;
            path[i].weight = next * (depth + 1) as f64 / ((i + 1) as f64 * one);
            next = tmp - path[i].weight * zero * (depth - i) as f64 / (depth + 1) as f64;
        }
    } else if zero != 0.0 {
        for (i, item) in path.iter_mut().enumerate().take(depth) {
            item.weight *= (depth + 1) as f64 / (zero * (depth - i) as f64);
        }
    }
    // Unwinding removes the split metadata from the path, but the pweights
    // above were recomputed for their current depths and must not be shifted.
    for i in index..depth {
        path[i].feature = path[i + 1].feature;
        path[i].zero = path[i + 1].zero;
        path[i].one = path[i + 1].one;
    }
    path.pop();
}
pub(crate) fn unwound_sum(path: &[PathElement], index: usize) -> f64 {
    let depth = path.len() - 1;
    let one = path[index].one;
    let zero = path[index].zero;
    let mut next = path[depth].weight;
    let mut total = 0.0;
    if one != 0.0 {
        for i in (0..depth).rev() {
            let tmp = next * (depth + 1) as f64 / ((i + 1) as f64 * one);
            total += tmp;
            next = path[i].weight - tmp * zero * (depth - i) as f64 / (depth + 1) as f64;
        }
    } else if zero != 0.0 {
        for (i, item) in path.iter().enumerate().take(depth) {
            total += item.weight * (depth + 1) as f64 / (zero * (depth - i) as f64);
        }
    }
    total
}
