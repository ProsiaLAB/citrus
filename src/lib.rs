#![doc(
    html_logo_url = "https://raw.githubusercontent.com/PlanetesLAB/planeteslab.github.io/refs/heads/main/images/planeteslab.jpeg"
)]
//! This is the documentation for the `citrus`.
//!
//! is an excitation and radiation transfer code that can be used to predict line and continuum radiation
//! from an astronomical source. The code uses unstructured 3D Delaunay grids for photon transport and
//! accelerated Lambda Iteration for population calculations.
//!
//! For a detailed theoretical description of the code, please read the [book](https://planeteslab.github.io/books/citrus/).

#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::similar_names)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::many_single_char_names)]

pub mod collparts;
pub mod config;
pub mod constants;
pub mod engine;
pub mod grid;
pub mod io;
pub mod lines;
pub mod models;
pub mod pops;
pub mod raytrace;
pub mod solver;
pub mod source;
pub mod tree;
pub mod utils;
