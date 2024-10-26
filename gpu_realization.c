//
// Created by Файзиева Юлия on 08.09.2024.
//
#include <CL/cl_platform.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

#include <stdio.h>

#include "gpu_realization.h"

#define CONCAT_(a) "-DLOCAL2=" #a
#define CONCAT(a) CONCAT_(a)

#ifndef LOCAL2
#define LOCAL2 64
#endif

cl_uint round_to(cl_uint n, cl_uint m) {
  cl_uint del = n / m + (n % m == 0 ? 0 : 1);
  return del * m;
}

cl_uint round_to_div(cl_uint n, cl_uint m) { return round_to(n, m) / m; }

cl_int calculate_host(cl_context context, cl_command_queue command_queue,
                      cl_kernel up_and_down_sweep, cl_kernel sum_kernel,
                      cl_mem a_mem, size_t n, cl_event *events,
                      cl_uint *event_i) {
  cl_int error = CL_SUCCESS;
  size_t rounded_n = round_to(n, LOCAL2 * 2);
  cl_mem sums_mem = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(cl_float) * (rounded_n / (LOCAL2 * 2)),
      NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "ClCreateBuffer is not successful: %d\n", error);
    return 1;
  }
  size_t global_work_size[] = {round_to_div(rounded_n, 2)};
  size_t local_work_size[] = {LOCAL2};

  clSetKernelArg(up_and_down_sweep, 0, sizeof(cl_mem), &a_mem);
  clSetKernelArg(up_and_down_sweep, 1, sizeof(cl_mem), &sums_mem);
  clSetKernelArg(up_and_down_sweep, 2, sizeof(cl_uint), &n);

  error = clEnqueueNDRangeKernel(command_queue, up_and_down_sweep, 1, NULL,
                                 global_work_size, local_work_size, 0, NULL,
                                 &events[(*event_i)++]);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "clEnqueueNDRangeKernel is not successful: %d\n", error);
    clReleaseMemObject(sums_mem);
    return 1;
  }

  if (rounded_n > LOCAL2 * 2) {
    error = calculate_host(
        context, command_queue, up_and_down_sweep, sum_kernel, sums_mem,
        round_to_div(rounded_n, LOCAL2 * 2), events, event_i);
    if (error != CL_SUCCESS) {
      clReleaseMemObject(sums_mem);
      return 1;
    };

    clSetKernelArg(sum_kernel, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(sum_kernel, 1, sizeof(cl_mem), &sums_mem);
    clSetKernelArg(sum_kernel, 2, sizeof(cl_uint), &n);

    error = clEnqueueNDRangeKernel(command_queue, sum_kernel, 1, NULL,
                                   global_work_size, local_work_size, 0, NULL,
                                   &events[(*event_i)++]);

    if (error != CL_SUCCESS) {
      fprintf(stderr, "clEnqueueNDRangeKernel is not successful: %d\n", error);
      clReleaseMemObject(sums_mem);
      return 1;
    }
  }

  clReleaseMemObject(sums_mem);
  return CL_SUCCESS;
};

int calculate1(cl_device_id device_id, cl_float *a, size_t n) {
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
  FILE *file = fopen("pref_sum.cl", "rb");
  if (file == NULL) {
    fprintf(stderr, "Error while reading kernel\n");
    return 1;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fprintf(stderr, "Error while reading kernel\n");
    fclose(file);
    return 1;
  }
  size_t size = ftell(file);
  if (fseek(file, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Error while reading kernel\n");
    fclose(file);
    return 1;
  }

  char *source = malloc(size + 1);
  if (source == NULL) {
    fprintf(stderr, "Error while allocating memory for kernel\n");
    fclose(file);
    return 1;
  }
  if (fread(source, 1, size, file) != size) {
    fprintf(stderr, "Error while reading kernel\n");
    fclose(file);
    free(source);
    return 1;
  }
  if (fclose(file) == EOF) {
    fprintf(stderr, "Error while reading kernel\n");
    free(source);
    return 1;
  }
  source[size] = '\0';
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source, (const size_t *)&size, NULL);

  free(source);
  cl_int build_err =
      clBuildProgram(program, 1, &device_id, CONCAT(LOCAL2), NULL, NULL);
  if (build_err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = malloc(log_size);
    if (log == NULL) {
      fprintf(stderr, "Failed to allocate memory for build log\n");
      clReleaseProgram(program);
      return 1;
    };
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);
    clReleaseProgram(program);
    fprintf(stderr, "%s\n", log);
    free(log);
    return 1;
  }
  cl_int error = CL_SUCCESS;
  cl_command_queue command_queue = clCreateCommandQueue(
      context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "Error while creating command queue\n");
    clReleaseProgram(program);
    clReleaseContext(context);
    free(source);
    return 1;
  }

  cl_kernel up_and_down_sweep =
      clCreateKernel(program, "up_and_down_sweep", &error);
  cl_kernel sum_kernel = clCreateKernel(program, "sum", &error);

  // biggest possible n is 18446744073709551616
  // lowest possble blocksize is 2
  // log2(2 ** 64) == 64
  // 1 write + 64 kernels + 1 read = 66 events
  cl_event events[66];
  cl_uint event_i = 0;
  cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * n, NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "ClCreateBuffer is not successful: %d\n", error);
    clReleaseKernel(up_and_down_sweep);
    clReleaseKernel(sum_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }
  error = clEnqueueWriteBuffer(command_queue, a_mem, CL_TRUE, 0, sizeof(cl_float) * n,
                       a, 0, NULL, &events[event_i++]);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "ClEnqueueWriteBuffer is not successful: %d\n", error);
    clReleaseMemObject(a_mem);
    clReleaseKernel(up_and_down_sweep);
    clReleaseKernel(sum_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }
  error = calculate_host(context, command_queue, up_and_down_sweep, sum_kernel,
                         a_mem, n, events, &event_i);
  if (error != CL_SUCCESS) {
    clReleaseMemObject(a_mem);
    clReleaseKernel(up_and_down_sweep);
    clReleaseKernel(sum_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }

  error = clEnqueueReadBuffer(command_queue, a_mem, CL_TRUE, 0, sizeof(cl_float) * n, a,
                      0, NULL, &events[event_i++]);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "ClEnqueueReadBuffer is not successful: %d\n", error);
    clReleaseMemObject(a_mem);
    clReleaseKernel(up_and_down_sweep);
    clReleaseKernel(sum_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 1;
  }
  clFinish(command_queue);

  cl_ulong time_start, time_end;
  cl_double time_with_memory = 0;
  cl_double time_without_memory = 0;
  for (int i = 0; i < event_i; i++) {
    clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                            sizeof(time_end), &time_end, NULL);
    time_with_memory += (double)(time_end - time_start) / 1000000.0;
    if (i != 0 && i != event_i - 1) {
      time_without_memory += (double)(time_end - time_start) / 1000000.0;
    }
  }
  printf("Time: %g\t%g\n", time_without_memory, time_with_memory);
  printf("LOCAL_WORK_SIZE [%i, %i]\n", LOCAL2, 1);
  clReleaseMemObject(a_mem);
  clReleaseKernel(up_and_down_sweep);
  clReleaseKernel(sum_kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseContext(context);
  return 0;
}
