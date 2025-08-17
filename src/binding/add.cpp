#include <pybind11/pybind11.h>
#include "../core/kernels/CPU/_add.h"

namespace py = pybind11;

PYBIND11_MODULE(add, m) {
    m.doc() = "Add  something";

    m.def("add", &add, "A function that adds tow numbers");
}