kernel void up_and_down_sweep(global float *a, global float *sums,
                              const uint n) {
  uint x = get_global_id(0);
  uint t = get_local_id(0);
  uint g = get_group_id(0);
  uint offset = 1;
  uint local_i = 2 * t;
  uint global_i = 2 * x;
  local float b[LOCAL2 * 2];

  b[local_i] = global_i >= n ? 0.0f : a[global_i];
  b[local_i + 1] = global_i + 1 >= n ? 0.0f : a[global_i + 1];

  for (uint i = LOCAL2; i > 0; i >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t < i) {
      int ai = offset * (local_i + 1) - 1;
      int bi = offset * (local_i + 2) - 1;

      b[bi] += b[ai];
    }
    offset <<= 1;
  }

  local float last;
  if (t == 0) {
    last = b[LOCAL2 * 2 - 1];
    sums[g] = last;
    b[LOCAL2 * 2 - 1] = 0;
  }

  for (uint i = 1; i < LOCAL2 * 2; i <<= 1) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t < i) {
      int ai = offset * (local_i + 1) - 1;
      int bi = offset * (local_i + 2) - 1;

      float t = b[ai];
      b[ai] = b[bi];
      b[bi] += t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (global_i < n) {
    a[global_i] = b[local_i + 1];
    if (global_i + 1 < n) {
      if (local_i + 2 < LOCAL2 * 2) {
        a[global_i + 1] = b[local_i + 2];
      } else {
        a[global_i + 1] = last;
      }
    }
  }
}
/*
kernel void sum(global float *a, global float *sums, const uint n) {
  uint x = get_global_id(0);
  uint g = get_group_id(0);

  uint global_i = 2 * x;
  local float block_sum;
  if (g > 0) {
    block_sum = sums[g - 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_i < n) {
      a[global_i] += block_sum;
      if (global_i + 1 < n) {
        a[global_i + 1] += block_sum;
      }
    }
  }
}
*/

kernel void sum(global float *a, global float *sums, const uint n) {
  uint x = get_global_id(0);
  uint g = get_group_id(0);

  uint global_i = 2 * x;
  if (g > 0) {
    float block_sum = sums[g - 1];

    if (global_i < n) {
      a[global_i] += block_sum;
      if (global_i + 1 < n) {
        a[global_i + 1] += block_sum;
      }
    }
  }
}
