use std::path::PathBuf;

use pyo3::{PyResult, Python, Py, PyErr};
use pyo3::types::PyModule;
use pyo3::types::PyAnyMethods;

use crate::model::traits::PhysicsModel;

pub struct PythonModel {
    py_module: Py<PyModule>,
}

impl PythonModel {
    /// Creates a new PythonModel by loading a Python file.
    ///
    /// The `file_name` can be just the name (e.g., "model.py") if the file is
    /// in the current working directory or a directory already on Python's sys.path.
    /// It can also be a relative or absolute path.
    pub fn new(file_name: &str) -> PyResult<Self> {
        Python::with_gil(|py| {
            let full_path = PathBuf::from(file_name);

            // Extract the module name (file name without extension)
            let module_name_str = full_path.file_stem()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid model file name: '{}' has no file name component.", file_name)
                ))?
                .to_str()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid model file name: '{}' is not valid UTF-8.", file_name)
                ))?;

            // Get the directory containing the file.
            // If `file_name` is just "model.py", parent() will be Some("").
            // If `file_name` is "subdir/model.py", parent() will be Some("subdir").
            let parent_dir_str = full_path.parent()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid model file name: '{}' has no parent directory component.", file_name)
                ))?
                .to_str()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid model file path: parent directory of '{}' is not valid UTF-8.", file_name)
                ))?;

            // Add the directory to Python's sys.path so the module can be found.
            // An empty string for parent_dir_str correctly represents the current working directory.
            if !parent_dir_str.is_empty() {
                let sys = py.import("sys")?;
                let sys_path = sys.getattr("path")?;
                // Insert at the beginning to prioritize our custom path
                sys_path.call_method1("insert", (0, parent_dir_str))?;
            }
            // If parent_dir_str is empty (""), it implies the current directory,
            // which is usually already on sys.path, so no explicit insert is needed.

            // Import the module by its name
            let module = PyModule::import(py, module_name_str)?;

            Ok(PythonModel {
                py_module: module.into(),
            })
        })
    }

    fn call_f64(&self, name: &str, x: f64, y: f64, z: f64, fallback: f64) -> f64 {
        Python::with_gil(|py| {
            let m = self.py_module.bind(py);
            match m.getattr(name) {
                Ok(f) => f.call1((x, y, z))
                          // Use a closure to call .extract() on the Bound<'py, PyAny> result
                          .and_then(|result_bound_pyany| result_bound_pyany.extract())
                          .unwrap_or_else(|e| {
                              eprintln!("Error extracting f64 or calling Python function '{}': {}", name, e);
                              fallback
                          }),
                Err(_) => fallback,
            }
        })
    }
}

impl PhysicsModel for PythonModel {
    fn density(&self, x: f64, y: f64, z: f64) -> f64 {
        self.call_f64("density", x, y, z, 0.0)
    }
    fn temperature(&self, x: f64, y: f64, z: f64) -> f64 {
        self.call_f64("temperature", x, y, z, 10.0)
    }
    fn abundance(&self, x: f64, y: f64, z: f64) -> f64 {
        self.call_f64("abundance", x, y, z, 1e-4)
    }
    fn mol_num_density(&self, x: f64, y: f64, z: f64) -> f64 {
        self.call_f64("mol_num_density", x, y, z, 0.0)
    }
    fn doppler(&self, x: f64, y: f64, z: f64) -> f64 {
        self.call_f64("doppler", x, y, z, 1e5)
    }
    fn velocity(&self, _x: f64, _y: f64, _z: f64) -> [f64; 3] {
        [0.0, 0.0, 0.0] 
    }
    fn magfield(&self, _x: f64, _y: f64, _z: f64) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
    fn gas_to_dust(&self, x: f64, y: f64, z: f64) -> f64 {
        self.call_f64("gas_to_dust", x, y, z, 100.0)
    }
}