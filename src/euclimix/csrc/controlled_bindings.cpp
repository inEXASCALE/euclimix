#include <torch/extension.h>
#include "euclimix/distance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Controlled storage/threshold distances and direct references";
    m.def("distance", &euclimix::controlled_distance,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("direct", &euclimix::direct_distance,
          pybind11::call_guard<pybind11::gil_scoped_release>());
}

