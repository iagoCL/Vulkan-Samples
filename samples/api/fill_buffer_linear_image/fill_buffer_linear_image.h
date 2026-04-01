/* Copyright (c) 2026, The Khronos Group
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <limits>
#include <utility>

#include "common/helpers.h"

#include "api_vulkan_sample.h"

class FillBufferLinearImage : public ApiVulkanSample
{
  public:
	struct ColorPixelData
	{
		uint8_t a = 0xFF;
		uint8_t b = 0;
		uint8_t g = 0;
		uint8_t r = 0;

		void init_float(float r_, float g_, float b_);
		void init_vec3(const glm::vec3 &value)
		{
			init_float(value.r, value.g, value.b);
		}
	};

	struct BufferImageData
	{
		uint32_t                    width  = 0;
		uint32_t                    height = 0;
		std::vector<ColorPixelData> data{};

		const ColorPixelData get_pixel(uint32_t id) const
		{
			return data.at(static_cast<size_t>(id));
		}
		const ColorPixelData get_pixel(uint32_t x, uint32_t y) const
		{
			return get_pixel(y * width + x);
		}
		ColorPixelData &get_pixel_non_const(uint32_t id)
		{
			return data.at(static_cast<size_t>(id));
		}
		ColorPixelData &get_pixel_non_const(uint32_t x, uint32_t y)
		{
			return get_pixel_non_const(y * width + x);
		}
	};

	FillBufferLinearImage();
	~FillBufferLinearImage() override;

	bool prepare(const vkb::ApplicationOptions &options) override;
	bool resize(const uint32_t width, const uint32_t height) override;
	void render(float delta_time) override;
	void build_command_buffers() override;

  private:
	enum class PatternType
	{
		Gradient = 0,
		CheckPattern
	};

	struct FillCommand
	{
		VkDeviceSize offset = 0;
		VkDeviceSize size   = 0;
		uint32_t     word   = 0;
	};

	struct CompressedFillCommand
	{
		uint32_t offset = 0;
		uint32_t size   = 0;
		uint32_t word   = 0;
	};

	struct FillCommandsCacheEntry
	{
		// Reusable fill plans for one logical image. The hash is derived from the generation inputs plus the fill
		// optimization mode, so a cache hit means we can skip re-running the CPU-side fill command search.
		uint64_t                           fill_commands_hash = 0;
		std::vector<CompressedFillCommand> commands{};
	};

	template <typename Entry>
	struct FixedRingCache
	{
		static constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();

		std::vector<Entry> entries{};
		uint32_t           next_slot = 0;

		void clear()
		{
			entries.clear();
			next_slot = 0;
		}

		bool empty() const
		{
			return entries.empty();
		}

		size_t size() const
		{
			return entries.size();
		}

		Entry &at(uint32_t index)
		{
			return entries.at(index);
		}

		const Entry &at(uint32_t index) const
		{
			return entries.at(index);
		}

		template <typename Predicate>
		uint32_t find_index(Predicate &&predicate) const
		{
			for (uint32_t index = 0; index < entries.size(); ++index)
			{
				if (predicate(entries[index]))
				{
					return index;
				}
			}
			return kInvalidIndex;
		}

		Entry &insert_or_assign(uint32_t capacity, Entry entry, uint32_t *assigned_index = nullptr)
		{
			assert(capacity > 0);

			if (entries.size() < capacity)
			{
				entries.push_back(std::move(entry));
				const uint32_t index = static_cast<uint32_t>(entries.size() - 1);
				if (assigned_index != nullptr)
				{
					*assigned_index = index;
				}
				return entries.back();
			}

			const uint32_t index = next_slot;
			next_slot            = (next_slot + 1) % capacity;
			entries[index]       = std::move(entry);
			if (assigned_index != nullptr)
			{
				*assigned_index = index;
			}
			return entries[index];
		}
	};

	struct CachedCpuImageEntry
	{
		// The CPU Image Cache stores the stable creation-parameter key for one logical image and a hash of the
		// generated CPU pixels.
		uint64_t creation_parameters_hash = 0;
		uint64_t cpu_image_pixels_hash    = 0;
	};

	struct CpuImageFillCommandsCacheEntry
	{
		uint64_t cpu_image_fill_commands_hash = 0;
		uint64_t fill_commands_hash           = 0;
	};

	struct FrameImageCreationParameters
	{
		// Declarative creation parameters for one logical frame image. This is the stable cache key input: if these fields do
		// not change, the generated image contents do not change either.
		uint64_t       creation_parameters_hash = 0;
		float          normalized_position      = 0.0f;
		PatternType    pattern_type{};
		ColorPixelData gradient_top_left{};
		ColorPixelData gradient_top_right{};
		ColorPixelData gradient_bottom_left{};
		ColorPixelData gradient_bottom_right{};
		ColorPixelData check_color_a{};
		ColorPixelData check_color_b{};
		uint32_t       check_size_x = 0;
		uint32_t       check_size_y = 0;
	};

	struct LinearImageAlias
	{
		VkImage        image        = VK_NULL_HANDLE;
		VkBuffer       buffer       = VK_NULL_HANDLE;
		VkDeviceMemory memory       = VK_NULL_HANDLE;
		VkDeviceSize   image_offset = 0;
		VkDeviceSize   row_pitch    = 0;
		VkDeviceSize   total_size   = 0;
		VkFormat       format       = VK_FORMAT_UNDEFINED;
	};

	struct RecordedCommandBufferEntry
	{
		VkCommandBuffer command_buffer               = VK_NULL_HANDLE;
		uint64_t        recorded_command_buffer_hash = 0;
		uint32_t        fill_command_count           = 0;
	};

	struct RecordedCommandBufferCache
	{
		FixedRingCache<RecordedCommandBufferEntry> entries{};

		void clear()
		{
			entries.clear();
		}
	};

	struct FrameInterpolationPattern
	{
		ColorPixelData gradient_top_left{};
		ColorPixelData gradient_top_right{};
		ColorPixelData gradient_bottom_left{};
		ColorPixelData gradient_bottom_right{};
		ColorPixelData check_color_a{};
		ColorPixelData check_color_b{};
	};
	struct PatternState
	{
		PatternType               pattern_type              = PatternType::CheckPattern;
		int32_t                   frame_image_count         = 15;
		bool                      combine_continuous_pixels = true;
		bool                      combine_most_common       = false;
		bool                      combine_both              = false;
		FrameInterpolationPattern start;
		FrameInterpolationPattern mid;
		uint32_t                  check_size_x = 70;
		uint32_t                  check_size_y = 55;
	};

  private:
	static constexpr uint32_t cpu_image_cache_size               = 512;
	static constexpr uint32_t cpu_image_fill_commands_cache_size = 1024;
	static constexpr uint32_t fill_commands_cache_size           = 1024;
	static constexpr uint32_t command_buffer_cache_size          = 512;

	void                                destroy_resources();
	void                                destroy_command_buffer_cache();
	void                                destroy_alias_resources();
	void                                create_resources();
	void                                create_alias_resources();
	void                                rebuild_sample();
	void                                build_frame_image_creation_parameters();
	void                                clear_cpu_images();
	const FrameImageCreationParameters &get_frame_image_creation_parameters(uint32_t frame_image_index);
	BufferImageData                     generate_cpu_image(const FrameImageCreationParameters &image_creation_parameters) const;
	uint32_t                            get_or_create_command_buffer_for_current_frame(uint32_t swapchain_image_index, uint32_t frame_image_index);
	void                                record_alias_fill_and_copy(VkCommandBuffer command_buffer, uint32_t swapchain_image_index, uint32_t frame_image_index, const FrameImageCreationParameters &image_creation_parameters, uint32_t cache_index);
	void                                build_command_buffer_cache();
	FillCommandsCacheEntry &            find_or_create_fill_commands_cache_entry(uint32_t image_index, const FrameImageCreationParameters &image_creation_parameters);
	std::vector<FillCommand>            decompress_fill_commands(const std::vector<CompressedFillCommand> &compressed_commands) const;
	std::vector<CompressedFillCommand>  compress_fill_commands(const std::vector<FillCommand> &commands) const;
	std::vector<FillCommand>            calculate_optimized_fill_cmd(const BufferImageData &image_data) const;
	std::vector<FillCommand>            normalize_fill_commands(std::vector<FillCommand> commands) const;
	std::vector<FillCommand>            build_row_fill_commands(const BufferImageData &image_data, const std::vector<bool> *skip_mask = nullptr) const;
	std::vector<FillCommand>            build_common_color_fill_commands(const BufferImageData &image_data, std::vector<bool> *covered_mask = nullptr) const;
	ColorPixelData                      interpolate_color(ColorPixelData a, ColorPixelData b, float proportion) const;
	uint64_t                            calculate_color_hash(const ColorPixelData &value) const;
	uint64_t                            calculate_creation_parameters_hash(const FrameImageCreationParameters &image_creation_parameters) const;
	uint64_t                            calculate_cpu_image_pixels_hash(const BufferImageData &image_data) const;
	uint64_t                            calculate_cpu_image_fill_commands_hash(uint64_t cpu_image_pixels_hash) const;
	uint64_t                            calculate_command_buffer_hash(uint32_t swapchain_image_index, VkImageLayout previous_swapchain_layout, uint64_t fill_commands_hash) const;
	uint64_t                            calculate_commands_hash(const std::vector<FillBufferLinearImage::CompressedFillCommand> &commands) const;
	BufferImageData                     init_gradient(ColorPixelData top_left, ColorPixelData top_right, ColorPixelData bottom_left, ColorPixelData bottom_right) const;
	BufferImageData                     init_check_pattern(ColorPixelData color_a, ColorPixelData color_b, uint32_t size_x, uint32_t size_y) const;
	uint32_t                            pack_color_word(const ColorPixelData &value) const;

  private:
	LinearImageAlias                               linear_alias{};
	VkCommandPool                                  cached_cmd_pool = VK_NULL_HANDLE;
	FixedRingCache<CachedCpuImageEntry>            cpu_image_cache{};
	FixedRingCache<CpuImageFillCommandsCacheEntry> cpu_image_fill_commands_cache{};
	std::vector<FrameImageCreationParameters>      frame_image_creation_parameters{};
	RecordedCommandBufferCache                     recorded_command_buffer_cache{};
	FixedRingCache<FillCommandsCacheEntry>         fill_commands_cache{};
	std::vector<VkImageLayout>                     swapchain_image_layouts{};
	PatternState                                   pattern_state{};
	uint64_t                                       frame_id        = 0;
	bool                                           resources_ready = false;
	bool                                           supported       = false;
};

std::unique_ptr<vkb::VulkanSampleC> create_fill_buffer_linear_image();
