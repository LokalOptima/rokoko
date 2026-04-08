// verbose.cpp — Definition of g_verbose (declared extern in weights.h)
// Weak: allows main.cu to provide a strong definition for CLI builds.
__attribute__((weak)) bool g_verbose = false;
