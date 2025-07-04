// src/lib.rs
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule};
use pyo3::Bound;
use rand::Rng;

// --- Global Distance Matrix ---
static DISTANCE_MATRIX: OnceCell<Vec<Vec<f64>>> = OnceCell::new();

// --- Existing PyFunctions (init, calculate_fitness_rust) ---
#[pyfunction]
#[allow(unsafe_op_in_unsafe_fn)]
fn init_distance_matrix(_py: Python<'_>, matrix_py: &Bound<'_, PyAny>) -> PyResult<()> {
    let matrix: Vec<Vec<f64>> = matrix_py.extract()?;
    match DISTANCE_MATRIX.set(matrix) {
        Ok(()) => Ok(()),
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Distance matrix has already been initialized for this process.",
        )),
    }
}

#[pyfunction]
#[allow(unsafe_op_in_unsafe_fn)]
fn calculate_fitness_rust(route_py: &Bound<'_, PyList>) -> PyResult<f64> {
    let matrix = DISTANCE_MATRIX.get().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Distance matrix not initialized. Call init_distance_matrix first.",
        )
    })?;
    let route: Vec<usize> = route_py.extract()?;
    calculate_fitness_core(&route, matrix) // Call the core logic
        .map_err(|e_str| PyErr::new::<pyo3::exceptions::PyIndexError, _>(e_str))
}

// --- Core Fitness Calculation Logic (for internal Rust use) ---
fn calculate_fitness_core(route: &[usize], matrix: &Vec<Vec<f64>>) -> Result<f64, String> {
    let num_points_in_route: usize = route.len();
    if num_points_in_route == 0 {
        return Ok(f64::INFINITY);
    }
    if num_points_in_route == 1 {
        return Ok(0.0);
    }

    let mut total_distance: f64 = 0.0;
    for i in 0..num_points_in_route {
        let from_point_idx: usize = route[i];
        let to_point_idx: usize = route[(i + 1) % num_points_in_route];

        if let Some(row) = matrix.get(from_point_idx) {
            if let Some(&distance_val) = row.get(to_point_idx) {
                if distance_val.is_infinite() || distance_val.is_nan() {
                    return Ok(f64::INFINITY);
                }
                total_distance += distance_val;
            } else {
                return Err(format!(
                    "to_point_idx {} out of bounds for matrix row {} (len {})",
                    to_point_idx,
                    from_point_idx,
                    row.len()
                ));
            }
        } else {
            return Err(format!(
                "from_point_idx {} out of bounds for matrix (len {})",
                from_point_idx,
                matrix.len()
            ));
        }
    }
    Ok(total_distance)
}

// Internal helper for 2-opt and 3-opt closures
fn fitness_adapter_for_opt(route_slice: &[usize]) -> f64 {
    if let Some(matrix_ref) = DISTANCE_MATRIX.get() {
        calculate_fitness_core(route_slice, matrix_ref).unwrap_or(f64::MAX)
    } else {
        eprintln!("Critical Error: DISTANCE_MATRIX not initialized when fitness_adapter_for_opt was called.");
        f64::MAX
    }
}

// --- 2-Opt Functionality ---
fn two_opt_swap_rs(route: &[usize], i: usize, k: usize) -> Vec<usize> {
    if i > k || k >= route.len() {
        panic!(
            "Invalid indices for two_opt_swap_rs (0-based): i={}, k={}, route_len={}. Ensure 0 <= i <= k < route.len().",
            i, k, route.len()
        );
    }
    let mut new_route = Vec::with_capacity(route.len());
    new_route.extend_from_slice(&route[0..i]);
    new_route.extend(route[i..=k].iter().rev());
    new_route.extend_from_slice(&route[k + 1..]);
    new_route
}

fn apply_two_opt_internal_rs<F>(
    initial_route: &[usize],
    full_run: bool,
    calculate_fitness_closure: &F,
) -> Vec<usize>
where
    F: Fn(&[usize]) -> f64,
{
    let num_nodes = initial_route.len();
    if num_nodes < 4 {
        return initial_route.to_vec();
    }
    let mut best_route = initial_route.to_vec();
    let mut best_distance = calculate_fitness_closure(&best_route);

    if !full_run {
        let total_iterations_quick_check = 2;
        if num_nodes <= 1 {
            return best_route;
        }
        let py_start_idx = rand::thread_rng().gen_range(1..num_nodes);

        for iter_offset in 0..total_iterations_quick_check {
            let i = py_start_idx + iter_offset;
            if i >= num_nodes - 1 {
                break;
            }
            for k in (i + 1)..num_nodes {
                let new_route = two_opt_swap_rs(&best_route, i, k);
                let new_distance = calculate_fitness_closure(&new_route);
                if new_distance < best_distance {
                    best_route = new_route;
                    best_distance = new_distance;
                }
            }
        }
    } else {
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..(num_nodes - 1) {
                for k in (i + 1)..num_nodes {
                    if k == i {
                        continue;
                    }
                    if i == 0 && k == num_nodes - 1 {
                        continue;
                    }
                    let new_route = two_opt_swap_rs(&best_route, i, k);
                    let new_distance = calculate_fitness_closure(&new_route);
                    if new_distance < best_distance {
                        best_route = new_route;
                        best_distance = new_distance;
                        improved = true;
                    }
                }
            }
        }
    }
    best_route
}

#[pyfunction]
#[allow(unsafe_op_in_unsafe_fn)]
fn apply_two_opt_from_python(
    py: Python<'_>,
    initial_route_py: &Bound<'_, PyList>,
    full_run: bool,
) -> PyResult<Py<PyList>> {
    let initial_route: Vec<usize> = initial_route_py.extract()?;
    DISTANCE_MATRIX.get().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Distance matrix not initialized. Call init_distance_matrix first.",
        )
    })?;

    let improved_route_vec =
        apply_two_opt_internal_rs(&initial_route, full_run, &fitness_adapter_for_opt);

    Ok(PyList::new(py, &improved_route_vec)?.into())
}

// --- 3-Opt Functionality ---

/// Helper to get distance between two nodes.
#[inline]
fn get_dist_internal(node1: usize, node2: usize) -> f64 {
    DISTANCE_MATRIX
        .get()
        .and_then(|matrix| matrix.get(node1))
        .and_then(|row| row.get(node2))
        .cloned()
        .unwrap_or(f64::MAX)
}

/// Reconstructs the route for a given 3-opt case.
/// i_idx, j_idx, k_idx are indices of the *first node* of each of the 3 original edges to be broken.
/// Edges are (route[i_idx], route[i_idx+1]), (route[j_idx], route[j_idx+1]), (route[k_idx], route[k_idx+1])
/// The segments are defined by these breaks.
fn reconstruct_three_opt_route(
    route: &[usize],
    i_idx: usize,
    j_idx: usize,
    k_idx: usize, // Indices of the start nodes of the 3 edges
    case_type: u8,
) -> Vec<usize> {
    let n = route.len();
    let mut new_route = Vec::with_capacity(n);

    // Define segments based on the nodes *after* the cuts.
    // Segment 1 (S1): from (k_idx + 1) % n up to i_idx (inclusive)
    // Segment 2 (S2): from (i_idx + 1) % n up to j_idx (inclusive)
    // Segment 3 (S3): from (j_idx + 1) % n up to k_idx (inclusive)

    let mut s1 = Vec::new();
    let mut current = (k_idx + 1) % n;
    loop {
        s1.push(route[current]);
        if current == i_idx {
            break;
        }
        current = (current + 1) % n;
    }

    let mut s2 = Vec::new();
    current = (i_idx + 1) % n;
    loop {
        s2.push(route[current]);
        if current == j_idx {
            break;
        }
        current = (current + 1) % n;
    }

    let mut s3 = Vec::new();
    current = (j_idx + 1) % n;
    loop {
        s3.push(route[current]);
        if current == k_idx {
            break;
        }
        current = (current + 1) % n;
    }

    // The 7 cases correspond to different ways of joining S1, S2, S3,
    // with some segments possibly reversed.
    match case_type {
        1 => {
            // (n1,n2), (n3,n5), (n4,n6) => S1 S2 S3(R)
            new_route.extend(&s1);
            new_route.extend(&s2);
            s3.reverse();
            new_route.extend(&s3);
        }
        2 => {
            // (n1,n3), (n2,n4), (n5,n6) => S1 S2(R) S3
            new_route.extend(&s1);
            s2.reverse();
            new_route.extend(&s2);
            new_route.extend(&s3);
        }
        3 => {
            // (n1,n5), (n3,n4), (n2,n6) => S1(R) S2 S3
            s1.reverse();
            new_route.extend(&s1);
            new_route.extend(&s2);
            new_route.extend(&s3);
        }
        4 => {
            // (n1,n3), (n2,n5), (n4,n6) => S1 S3(R) S2
            new_route.extend(&s1);
            s3.reverse();
            new_route.extend(&s3);
            new_route.extend(&s2);
        }
        5 => {
            // (n1,n4), (n3,n5), (n2,n6) => S1 S2(R) S3(R)
            new_route.extend(&s1);
            s2.reverse();
            new_route.extend(&s2);
            s3.reverse();
            new_route.extend(&s3);
        }
        6 => {
            // (n1,n4), (n2,n5), (n3,n6) => S1 S3 S2(R)
            new_route.extend(&s1);
            new_route.extend(&s3);
            s2.reverse();
            new_route.extend(&s2);
        }
        7 => {
            // (n1,n5), (n2,n4), (n3,n6) => S1(R) S3(R) S2
            s1.reverse();
            new_route.extend(&s1);
            s3.reverse();
            new_route.extend(&s3);
            new_route.extend(&s2);
        }
        _ => return route.to_vec(), // Should not happen if case_type is 1-7
    }
    new_route
}

fn apply_three_opt_internal_rs_complete<F>(
    initial_route: &[usize],
    calculate_fitness: &F,
) -> Vec<usize>
where
    F: Fn(&[usize]) -> f64,
{
    let n = initial_route.len();
    if n < 6 {
        return initial_route.to_vec();
    }

    let mut best_route = initial_route.to_vec();
    let mut best_cost = calculate_fitness(&best_route);

    let mut improved = true;
    while improved {
        improved = false;
        'outer: for i_idx in 0..n {
            for j_idx_offset in 2..(n - 1) {
                let j_idx = (i_idx + j_idx_offset) % n;
                if j_idx == i_idx {
                    continue;
                }

                for k_idx_offset in 2..(n - 1) {
                    let k_idx = (j_idx + k_idx_offset) % n;
                    if k_idx == i_idx || k_idx == j_idx {
                        continue;
                    }

                    let n1 = best_route[i_idx];
                    let n2 = best_route[(i_idx + 1) % n];
                    let n3 = best_route[j_idx];
                    let n4 = best_route[(j_idx + 1) % n];
                    let n5 = best_route[k_idx];
                    let n6 = best_route[(k_idx + 1) % n];

                    if n2 == n3
                        || n2 == n4
                        || n2 == n5
                        || n2 == n6
                        || n4 == n5
                        || n4 == n6
                        || n4 == n1
                        || n6 == n1
                        || n6 == n2
                        || n6 == n3
                    {
                        continue;
                    }

                    let d = |u, v| get_dist_internal(u, v);
                    let original_sum = d(n1, n2) + d(n3, n4) + d(n5, n6);

                    let candidates = [
                        (d(n1, n2) + d(n3, n5) + d(n4, n6), 1u8),
                        (d(n1, n3) + d(n2, n4) + d(n5, n6), 2u8),
                        (d(n1, n5) + d(n3, n4) + d(n2, n6), 3u8),
                        (d(n1, n3) + d(n2, n5) + d(n4, n6), 4u8),
                        (d(n1, n4) + d(n3, n5) + d(n2, n6), 5u8),
                        (d(n1, n4) + d(n2, n5) + d(n3, n6), 6u8),
                        (d(n1, n5) + d(n2, n4) + d(n3, n6), 7u8),
                    ];

                    for (new_sum, case_type) in candidates.iter() {
                        if *new_sum < original_sum - 1e-9 {
                            let temp_route = reconstruct_three_opt_route(
                                &best_route,
                                i_idx,
                                j_idx,
                                k_idx,
                                *case_type,
                            );

                            if temp_route.len() == n {
                                let new_cost = calculate_fitness(&temp_route);
                                if new_cost < best_cost - 1e-9 {
                                    best_route = temp_route;
                                    best_cost = new_cost;
                                    improved = true;
                                    continue 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    best_route
}

#[pyfunction]
#[allow(unsafe_op_in_unsafe_fn)]
fn apply_three_opt_from_python(
    py: Python<'_>,
    initial_route_py: &Bound<'_, PyList>,
) -> PyResult<Py<PyList>> {
    let initial_route: Vec<usize> = initial_route_py.extract()?;
    DISTANCE_MATRIX.get().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Distance matrix not initialized. Call init_distance_matrix first.",
        )
    })?;

    let improved_route_vec =
        apply_three_opt_internal_rs_complete(&initial_route, &fitness_adapter_for_opt);

    Ok(PyList::new(py, &improved_route_vec)?.into())
}

// --- PyModule Definition ---
#[pymodule]
#[pyo3(name = "ga_helpers_rs")]
fn module_ga_helpers_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_fitness_rust, m)?)?;
    m.add_function(wrap_pyfunction!(apply_two_opt_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(apply_three_opt_from_python, m)?)?;
    Ok(())
}
