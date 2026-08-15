// C-linkage shim around Triangle's C++-mangled API so Rust can link against
// stable, unmangled symbol names.
#include "triangle.h"

extern "C" {

void weka_triangulate(char *triswitches, struct triangulateio *in,
                       struct triangulateio *out, struct triangulateio *vorout) {
  triangulate(triswitches, in, out, vorout);
}

void weka_trifree(int *memptr) { trifree(memptr); }

} // extern "C"
