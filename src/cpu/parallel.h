#pragma once
#include <cstddef>

namespace rokoko::cpu {
// Synchronous work on the existing BLAS workers, including the calling thread.
// Small and nested jobs run inline. Exceptions are rethrown on the caller.
void parallel_for(size_t count, size_t grain,
                  void (*function)(void *, size_t, size_t), void *data);
bool in_parallel();

template <class F> void parallel(size_t count, size_t grain, F function) {
    parallel_for(count, grain,
                 [](void *p, size_t begin, size_t end) {
                     (*static_cast<F *>(p))(begin, end);
                 },
                 &function);
}
} // namespace rokoko::cpu
