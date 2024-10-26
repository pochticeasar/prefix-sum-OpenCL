#pragma once

#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>
#endif

#include <stdlib.h>

int calculate1(cl_device_id device_id, cl_float* a, size_t n);
