#include <torch/extension.h>
#include "euclimix/distance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Mixed-precision Euclidean distances with the Original reliability test";
    pybind11::register_local_exception<c10::ValueError>(m, "_InvalidInputError", PyExc_ValueError);
    m.def("distance", &euclimix::paper_distance, pybind11::arg("X"), pybind11::arg("Y"),
          pybind11::arg("precision"), pybind11::arg("correction"), pybind11::arg("rho"),
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("euclidean", &euclimix::paper_euclidean_distance, pybind11::arg("X"), pybind11::arg("Y"),
          pybind11::arg("precision"), pybind11::arg("correction"), pybind11::arg("rho"),
          pybind11::call_guard<pybind11::gil_scoped_release>());
}

