//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/sort/sort.h>

#include <vector>

#include <nccl_test_common.h>
#include <testing.cuh>

#include "sort_common.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/vector.h>

namespace
{
using sort_test_util::abs_less;
using sort_test_util::make_value;
using sort_test_util::sort_types;

// Run the whole world's sort through the range overload and check the result against a host-side
// `cuda::std::sort` of the same elements. Every test in this file differs only in how the inputs
// are shaped, so all of them funnel through here.
template <class T, class Compare>
void check_sort_case(
  cuda::std::span<cudax::nccl_communicator_ref> comms, const std::vector<c2h::host_vector<T>>& host_inputs, Compare cmp)
{
  REQUIRE(host_inputs.size() == comms.size());

  const auto expected = sort_test_util::sorted_reference(host_inputs, cmp);
  auto streams        = nccl_test_util::make_streams();
  auto environments   = std::vector<cuda::stream_ref>{streams.begin(), streams.end()};
  auto device_vec     = sort_test_util::make_device_inputs(comms, host_inputs);

  cudax::sort(cudax::distributed, comms, environments, device_vec, cmp);

  // Since we are using c2h vectors instead of cuda buffers (which remember what stream they
  // are on), we need to sync here before doing the checks because the internal copy stream
  // wont know to wait on the data.
  for (auto& stream : streams)
  {
    stream.sync();
  }

  sort_test_util::check_rank_sizes(comms, device_vec, host_inputs);

  const auto output = sort_test_util::gather_outputs(comms, device_vec);

  REQUIRE(cuda::std::is_sorted(output.begin(), output.end(), cmp));
  REQUIRE_THAT(output, Equals(expected));
}

// Every input shape is worth exercising under both orderings: an ascending-only test would not
// catch a comparator that is applied with its arguments swapped somewhere in the pipeline.
template <class T>
void check_sort_case_sections(cuda::std::span<cudax::nccl_communicator_ref> comms,
                              const std::vector<c2h::host_vector<T>>& host_inputs)
{
  SECTION("ascending comparator")
  {
    check_sort_case(comms, host_inputs, cuda::std::less<>{});
  }

  SECTION("descending comparator")
  {
    check_sort_case(comms, host_inputs, cuda::std::greater<>{});
  }
}
} // namespace

MULTI_GPU_TEST("sort documentation example", c2h::type_list<int>)
{
  auto comms = this->communicators();

  if (comms.size() < 2)
  {
    SKIP("The sort documentation example requires at least two local GPUs");
  }

  auto streams_owned = nccl_test_util::make_streams();
  // Convert to stream_ref directly, cuda::stream on their own cant be passed directly to CUB
  auto streams = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};

  //! [sort]
  // Rank 0 holds {3, 1} and rank 1 holds {4, 2}, so the global sequence is {3, 1, 4, 2}. Each
  // input range must be resizable, since the sort re-partitions the keys across the ranks before
  // restoring the original per-rank sizes.
  std::vector<thrust::device_vector<int>> inputs;

  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(comms[i].logical_device().underlying_device().get()));
    inputs.emplace_back(i == 0 ? thrust::device_vector<int>{3, 1} : thrust::device_vector<int>{4, 2});
  }

  cudax::sort(cudax::distributed,
              comms,
              // Passing streams as the environment directly
              streams,
              inputs);

  // The sort is in place and each rank keeps its original element count, so the globally sorted
  // sequence {1, 2, 3, 4} is split back into two elements per rank, in ascending rank order.
  const thrust::device_vector<int> expected_rank_0{1, 2};
  const thrust::device_vector<int> expected_rank_1{3, 4};
  //! [sort]

  for (auto& stream : streams)
  {
    stream.sync();
  }

  REQUIRE(inputs[0] == expected_rank_0);
  REQUIRE(inputs[1] == expected_rank_1);
}

MULTI_GPU_TEST("sort, random inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (auto& local : input)
  {
    sort_test_util::fill_random(local, 100, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, uneven rank sizes", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    sort_test_util::fill_random(input[rank], (rank * 10) + 1, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, inputs with some empty ranks", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 1; rank < input.size(); rank += 2)
  {
    sort_test_util::fill_random(input[rank], 1'000, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, no communicators", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  const auto comms = cuda::std::span<cudax::nccl_communicator_ref>{};
  std::vector<c2h::host_vector<T>> input(comms.size());

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, all ranks empty", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, a single global item", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  if (!input.empty())
  {
    input[0].push_back(make_value<T>(1, 1));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, one item per rank", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    const auto key = static_cast<cuda::std::int64_t>(input.size() - rank);
    input[rank].push_back(make_value<T>(key, key));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, all equal inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (auto& local : input)
  {
    local.assign(100, make_value<T>(1, 1));
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, inputs with many equal keys", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(item % 2);
      local[item]    = make_value<T>(key, static_cast<cuda::std::int64_t>(rank * local.size() + item));
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, presorted inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(rank * local.size() + item);
      local[item]    = make_value<T>(key, key);
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, reverse-sorted inputs", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  std::vector<c2h::host_vector<T>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto key = static_cast<cuda::std::int64_t>(input.size() * local.size() - (rank * local.size() + item));
      local[item]    = make_value<T>(key, key);
    }
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, skewed rank sizes", sort_types)
{
  using T = typename c2h::get<0, TestType>;

  auto comms = this->communicators();
  auto rng   = sort_test_util::make_rng(C2H_SEED(2));

  std::vector<c2h::host_vector<T>> input(comms.size());
  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    sort_test_util::fill_random(input[rank], rank == 0 ? 1'000 : 1, rng);
  }

  check_sort_case_sections(comms, input);
}

MULTI_GPU_TEST("sort, nonstandard comparator", )
{
  auto comms = this->communicators();
  std::vector<c2h::host_vector<int>> input(comms.size());

  for (cuda::std::size_t rank = 0; rank < input.size(); ++rank)
  {
    auto& local = input[rank];
    local.resize(1'000);

    for (cuda::std::size_t item = 0; item < local.size(); ++item)
    {
      const auto magnitude = static_cast<int>((rank + item) % 5);
      local[item]          = item % 2 == 0 ? magnitude : -magnitude;
    }
  }

  check_sort_case(comms, input, abs_less<int>{});
}
