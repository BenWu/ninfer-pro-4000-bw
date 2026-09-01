#include <cuda_runtime.h>
// RDC anchor: forces the device-link step for C++-only test executables that link
// RDC static libraries (cross-TU device symbols otherwise stay unresolved).
extern "C" int ninfer_test_rdc_anchor() { return 0; }
