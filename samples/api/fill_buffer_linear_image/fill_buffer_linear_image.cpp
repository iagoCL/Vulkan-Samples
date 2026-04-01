/* Copyright (c) 2026, Iago Calvo Lista
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

#include "fill_buffer_linear_image.h"

namespace
{
constexpr uint32_t kBytesPerPixel          = sizeof(uint32_t);
constexpr uint32_t kMinimumRepeatedOffsets = 10;
constexpr int32_t  kMinimumFrameImageCount = 1;
constexpr int32_t  kMaximumFrameImageCount = 60;

uint8_t float_to_byte(float value)
{
	const float clamped = std::clamp(value, 0.0f, 1.0f);
	return static_cast<uint8_t>(std::lround(clamped * 255.0f));
}

float byte_to_float(uint8_t value)
{
	return static_cast<float>(value) / 255.0f;
}

const auto lerp_byte(uint8_t a, uint8_t b, float t)
{
	return std::clamp(byte_to_float(a) + (byte_to_float(b) - byte_to_float(a)) * t, 0.0f, 1.0f);
};

VkDeviceSize align_down_to_fill_size(VkDeviceSize value)
{
	return value - (value % sizeof(uint32_t));
}

}        // namespace

void FillBufferLinearImage::ColorPixelData::init_float(float r_, float g_, float b_)
{
	a = 0xFF;
	b = float_to_byte(b_);
	g = float_to_byte(g_);
	r = float_to_byte(r_);
}

FillBufferLinearImage::FillBufferLinearImage()
{
	title = "Fill buffer into a linear image";

	pattern_state.start.gradient_top_left.init_float(1.0f, 0.0f, 0.0f);
	pattern_state.start.gradient_top_right.init_float(0.0f, 1.0f, 0.0f);
	pattern_state.start.gradient_bottom_left.init_float(1.0f, 0.0f, 1.0f);
	pattern_state.start.gradient_bottom_right.init_float(0.0f, 0.0f, 1.0f);
	pattern_state.start.check_color_a.init_float(0.0f, 1.0f, 1.0f);
	pattern_state.start.check_color_b.init_float(0.0f, 0.0f, 1.0f);

	pattern_state.mid.gradient_top_left.init_float(0.0f, 0.8f, 0.8f);
	pattern_state.mid.gradient_top_right.init_float(0.8f, 0.0f, 0.0f);
	pattern_state.mid.gradient_bottom_left.init_float(0.0f, 0.8f, 0.0f);
	pattern_state.mid.gradient_bottom_right.init_float(0.8f, 0.8f, 0.0f);
	pattern_state.mid.check_color_a.init_float(1.0f, 0.0f, 0.0f);
	pattern_state.mid.check_color_b.init_float(0.0f, 1.0f, 0.0f);
}

FillBufferLinearImage::~FillBufferLinearImage()
{
	destroy_resources();
	if (!has_device() || cached_cmd_pool == VK_NULL_HANDLE)
	{
		cached_cmd_pool = VK_NULL_HANDLE;
		return;
	}

	vkDestroyCommandPool(get_device().get_handle(), cached_cmd_pool, nullptr);
	cached_cmd_pool = VK_NULL_HANDLE;
}

uint32_t FillBufferLinearImage::pack_color_word(const ColorPixelData &value) const
{
	if (linear_alias.format == VK_FORMAT_R8G8B8A8_UNORM || linear_alias.format == VK_FORMAT_R8G8B8A8_SRGB)
	{
		// Pack as RGBA8
		return (static_cast<uint32_t>(value.a) << 24) | (static_cast<uint32_t>(value.b) << 16) | (static_cast<uint32_t>(value.g) << 8) | static_cast<uint32_t>(value.r);
	}

	if (linear_alias.format == VK_FORMAT_B8G8R8A8_UNORM || linear_alias.format == VK_FORMAT_B8G8R8A8_SRGB)
	{
		//Pack as BGRA8
		return (static_cast<uint32_t>(value.a) << 24) | (static_cast<uint32_t>(value.r) << 16) | (static_cast<uint32_t>(value.g) << 8) | static_cast<uint32_t>(value.b);
	}

	throw std::runtime_error("fill_buffer_linear_image: cannot pack texels for unsupported linear alias format " + vkb::to_string(linear_alias.format));
}

FillBufferLinearImage::BufferImageData FillBufferLinearImage::init_gradient(ColorPixelData top_left,
                                                                            ColorPixelData top_right,
                                                                            ColorPixelData bottom_left,
                                                                            ColorPixelData bottom_right) const
{
	BufferImageData image_data{
	    .width  = width,
	    .height = height,
	    .data   = std::vector<ColorPixelData>(width * height),
	};
	LOGI("Creating gradient: size {}x{} parameters ({},{},{})x({},{},{})x({}x{}x{})", width, height,
	     top_left.r, top_left.g, top_left.b,
	     top_right.r, top_right.g, top_right.b,
	     bottom_left.r, bottom_left.g, bottom_left.b,
	     bottom_right.r, bottom_right.g, bottom_right.b);

	const auto lerp       = [](float a, float b, float t) { return a + (b - a) * t; };
	const auto lerp_pixel = [](ColorPixelData a, ColorPixelData b, float t) {
		return glm::vec3(
		    lerp_byte(a.r, b.r, t),
		    lerp_byte(a.g, b.g, t),
		    lerp_byte(a.b, b.b, t));
	};
	const uint32_t width_denominator  = (image_data.width > 1u) ? image_data.width - 1u : 1u;
	const uint32_t height_denominator = (image_data.height > 1u) ? image_data.height - 1u : 1u;

	for (uint32_t x = 0; x < image_data.width; ++x)
	{
		const float fx = static_cast<float>(x) / static_cast<float>(width_denominator);

		const glm::vec3 top{lerp_pixel(top_left, top_right, fx)};
		const glm::vec3 bottom{lerp_pixel(bottom_left, bottom_right, fx)};
		for (uint32_t y = 0; y < image_data.height; ++y)
		{
			const float fy = static_cast<float>(y) / static_cast<float>(height_denominator);

			ColorPixelData pixel;
			pixel.init_float(lerp(top.r, bottom.r, fy), lerp(top.g, bottom.g, fy), lerp(top.b, bottom.b, fy));
			image_data.get_pixel_non_const(x, y) = pixel;
		}
	}

	return image_data;
}

FillBufferLinearImage::BufferImageData FillBufferLinearImage::init_check_pattern(ColorPixelData color_a,
                                                                                 ColorPixelData color_b,
                                                                                 uint32_t       size_x,
                                                                                 uint32_t       size_y) const
{
	LOGI("Creating check pattern: size {}x{} parameters ({},{},{})x({},{},{}) at {}x{}", width, height,
	     color_a.r, color_a.g, color_a.b,
	     color_b.r, color_b.g, color_b.b, size_x, size_y);
	BufferImageData image_data{
	    .width  = width,
	    .height = height,
	    .data   = std::vector<ColorPixelData>(width * height),
	};

	size_x = (size_x > 1u) ? size_x : 1u;
	size_y = (size_y > 1u) ? size_y : 1u;

	for (uint32_t y = 0; y < image_data.height; ++y)
	{
		for (uint32_t x = 0; x < image_data.width; ++x)
		{
			const bool use_color_a               = (((x / size_x) + (y / size_y)) % 2u) == 0u;
			image_data.get_pixel_non_const(x, y) = use_color_a ? color_a : color_b;
		}
	}

	return image_data;
}

void FillBufferLinearImage::destroy_alias_resources()
{
	if (!has_device())
	{
		return;
	}

	if (linear_alias.image != VK_NULL_HANDLE)
	{
		vkDestroyImage(get_device().get_handle(), linear_alias.image, nullptr);
		linear_alias.image = VK_NULL_HANDLE;
	}
	if (linear_alias.buffer != VK_NULL_HANDLE)
	{
		vkDestroyBuffer(get_device().get_handle(), linear_alias.buffer, nullptr);
		linear_alias.buffer = VK_NULL_HANDLE;
	}
	if (linear_alias.memory != VK_NULL_HANDLE)
	{
		vkFreeMemory(get_device().get_handle(), linear_alias.memory, nullptr);
		linear_alias.memory = VK_NULL_HANDLE;
	}

	linear_alias.image_offset = 0;
	linear_alias.row_pitch    = 0;
	linear_alias.total_size   = 0;
}

void FillBufferLinearImage::destroy_command_buffer_cache()
{
	if (!recorded_command_buffer_cache.entries.empty())
	{
		LOGI("fill_buffer_linear_image: invalidating command buffer cache ({} cached command buffers)", recorded_command_buffer_cache.entries.size());
	}

	if (!has_device() || recorded_command_buffer_cache.entries.empty())
	{
		recorded_command_buffer_cache.clear();
		return;
	}

	if (!recorded_command_buffer_cache.entries.empty())
	{
		std::vector<VkCommandBuffer> command_buffers{};
		command_buffers.reserve(recorded_command_buffer_cache.entries.size());
		for (const RecordedCommandBufferEntry &entry : recorded_command_buffer_cache.entries.entries)
		{
			command_buffers.push_back(entry.command_buffer);
		}

		vkFreeCommandBuffers(get_device().get_handle(),
		                     cached_cmd_pool,
		                     static_cast<uint32_t>(command_buffers.size()),
		                     command_buffers.data());
	}

	recorded_command_buffer_cache.clear();
}

void FillBufferLinearImage::destroy_resources()
{
	destroy_command_buffer_cache();
	fill_commands_cache.clear();
	cpu_image_fill_commands_cache.clear();
	destroy_alias_resources();
	clear_cpu_images();
	swapchain_image_layouts.clear();
	resources_ready = false;
}

void FillBufferLinearImage::create_alias_resources()
{
	// Keep the alias image format identical to the swapchain format so the final vkCmdCopyImage is a straight format-compatible copy.
	linear_alias.format = get_render_context().get_swapchain().get_format();
	{
		VkFormatProperties format_properties;
		vkGetPhysicalDeviceFormatProperties(get_device().get_gpu().get_handle(), linear_alias.format, &format_properties);

		if (linear_alias.format != VK_FORMAT_R8G8B8A8_UNORM &&
		    linear_alias.format != VK_FORMAT_B8G8R8A8_UNORM &&
		    linear_alias.format != VK_FORMAT_R8G8B8A8_SRGB &&
		    linear_alias.format != VK_FORMAT_B8G8R8A8_SRGB)
		{
			throw std::runtime_error("fill_buffer_linear_image: unsupported swapchain format " + vkb::to_string(linear_alias.format) + ". Expected VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, or VK_FORMAT_B8G8R8A8_SRGB.");
		}

		if ((format_properties.linearTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) == 0)
		{
			throw std::runtime_error("fill_buffer_linear_image: swapchain format " + vkb::to_string(linear_alias.format) + " is missing VK_FORMAT_FEATURE_TRANSFER_SRC_BIT for linear tiling.");
		}
	}

	// The aliased image is linear-tiled on purpose. Only linear images expose a defined byte layout through
	// vkGetImageSubresourceLayout, which lets the sample compute the exact buffer offsets that correspond to texels.
	VkImageCreateInfo image_create_info{
	    .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
	    .imageType     = VK_IMAGE_TYPE_2D,
	    .format        = linear_alias.format,
	    .extent        = {width, height, 1},
	    .mipLevels     = 1,
	    .arrayLayers   = 1,
	    .samples       = VK_SAMPLE_COUNT_1_BIT,
	    .tiling        = VK_IMAGE_TILING_LINEAR,
	    .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
	    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	VK_CHECK(vkCreateImage(get_device().get_handle(), &image_create_info, nullptr, &linear_alias.image));

	VkImageSubresource  subresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
	VkSubresourceLayout layout;
	// The aliased buffer must cover the full byte span used by the linear image,
	// including any implementation-defined leading offset and row padding.
	// A tightly packed width * height * kBytesPerPixel buffer is not sufficient here.
	vkGetImageSubresourceLayout(get_device().get_handle(), linear_alias.image, &subresource, &layout);
	linear_alias.image_offset = layout.offset;
	linear_alias.row_pitch    = layout.rowPitch;
	linear_alias.total_size   = layout.size;

	// Buffer should have the same size as the image, since it is aliased.
	const VkDeviceSize aliased_buffer_size = linear_alias.image_offset + linear_alias.total_size;

	// vkCmdFillBuffer can only target a VkBuffer,
	// so we create a buffer view of the same memory and later bind both resources to one VkDeviceMemory allocation.
	VkBufferCreateInfo buffer_create_info{
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size  = aliased_buffer_size,
	    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	VK_CHECK(vkCreateBuffer(get_device().get_handle(), &buffer_create_info, nullptr, &linear_alias.buffer));

	VkMemoryRequirements image_requirements;
	vkGetImageMemoryRequirements(get_device().get_handle(), linear_alias.image, &image_requirements);
	VkMemoryRequirements buffer_requirements;
	vkGetBufferMemoryRequirements(get_device().get_handle(), linear_alias.buffer, &buffer_requirements);

	const uint32_t compatible_memory_types = image_requirements.memoryTypeBits & buffer_requirements.memoryTypeBits;
	if (compatible_memory_types == 0)
	{
		throw std::runtime_error("The linear image and buffer do not share a compatible memory type for aliasing");
	}

	VkBool32 memory_type_found = VK_FALSE;
	uint32_t memory_type_index = get_device().get_gpu().get_memory_type(compatible_memory_types, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory_type_found);
	if (memory_type_found == VK_FALSE)
	{
		// Aliasing only requires one shared memory type.
		// Prefer device-local memory when available, then fall back to any compatible type.
		memory_type_index = get_device().get_gpu().get_memory_type(compatible_memory_types, 0, &memory_type_found);
		if (memory_type_found == VK_FALSE)
		{
			throw std::runtime_error("Unable to find valid memory");
		}

		LOGW("DEVICE_LOCAL memory does not support aliasing a buffer and image.");
	}

	// Aliasing only requires one allocation large enough for the stricter of the two requirement sets.
	const VkDeviceSize allocation_size = std::max(image_requirements.size, buffer_requirements.size);
	if (image_requirements.size != buffer_requirements.size)
	{
		LOGW("Aliased buffer does not have the same size as the image.");
	}
	VkMemoryAllocateInfo allocate_info{
	    .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize  = allocation_size,
	    .memoryTypeIndex = memory_type_index,
	};
	VK_CHECK(vkAllocateMemory(get_device().get_handle(), &allocate_info, nullptr, &linear_alias.memory));

	// Binding the image and buffer to the same buffer.
	// This is key, now image and buffer are two interpretations of the same physical memory.
	VK_CHECK(vkBindImageMemory(get_device().get_handle(), linear_alias.image, linear_alias.memory, 0));
	VK_CHECK(vkBindBufferMemory(get_device().get_handle(), linear_alias.buffer, linear_alias.memory, 0));

	// Establish the image's tracked layout once so the first recorded frame can legally keep it in GENERAL.
	const auto &                  queue          = get_device().get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0);
	VkCommandBuffer               command_buffer = get_device().create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
	const VkImageSubresourceRange color_range{
	    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
	    .baseMipLevel   = 0,
	    .levelCount     = 1,
	    .baseArrayLayer = 0,
	    .layerCount     = 1,
	};
	vkb::image_layout_transition(command_buffer,
	                             linear_alias.image,
	                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                             VK_PIPELINE_STAGE_TRANSFER_BIT,
	                             0,
	                             VK_ACCESS_TRANSFER_WRITE_BIT,
	                             VK_IMAGE_LAYOUT_UNDEFINED,
	                             VK_IMAGE_LAYOUT_GENERAL,
	                             color_range);
	get_device().flush_command_buffer(command_buffer, queue.get_handle());
}

void FillBufferLinearImage::create_resources()
{
	destroy_resources();
	//Init parameters
	{
		constexpr uint32_t kDefaultCheckPatternSizeX = 70;
		constexpr uint32_t kDefaultCheckPatternSizeY = 55;

		pattern_state.check_size_x = kDefaultCheckPatternSizeX;
		pattern_state.check_size_y = kDefaultCheckPatternSizeY;

		assert(pattern_state.pattern_type == PatternType::Gradient || pattern_state.pattern_type == PatternType::CheckPattern);
		assert(pattern_state.frame_image_count >= 1);

		if (pattern_state.combine_both)
		{
			pattern_state.combine_continuous_pixels = true;
			pattern_state.combine_most_common       = true;
		}
	}
	create_alias_resources();
	swapchain_image_layouts.assign(get_render_context().get_swapchain().get_images().size(), VK_IMAGE_LAYOUT_UNDEFINED);
	resources_ready = true;
}

FillBufferLinearImage::ColorPixelData FillBufferLinearImage::interpolate_color(ColorPixelData a, ColorPixelData b, float proportion) const
{
	return ColorPixelData{
	    .a = (a.a == b.a) ? a.a : float_to_byte(lerp_byte(a.a, b.a, proportion)),
	    .b = (a.b == b.b) ? a.b : float_to_byte(lerp_byte(a.b, b.b, proportion)),
	    .g = (a.g == b.g) ? a.g : float_to_byte(lerp_byte(a.g, b.g, proportion)),
	    .r = (a.r == b.r) ? a.r : float_to_byte(lerp_byte(a.r, b.r, proportion)),
	};
}

uint64_t FillBufferLinearImage::calculate_color_hash(const ColorPixelData &value) const
{
	return (static_cast<uint64_t>(value.a) << 24) |
	       (static_cast<uint64_t>(value.b) << 16) |
	       (static_cast<uint64_t>(value.g) << 8) |
	       static_cast<uint64_t>(value.r);
}

void FillBufferLinearImage::build_frame_image_creation_parameters()
{
	// Populate the Frame Image Creation Parameter Cache.
	// Every later cache layer keys off these creation-parameter hashes.
	// This lets the sample reuse work across rebuilds when the generation inputs match.
	LOGI("fill_buffer_linear_image: frame_image_creation_parameter_cache rebuild logical_image_count {}", pattern_state.frame_image_count);
	frame_image_creation_parameters.clear();
	frame_image_creation_parameters.reserve(static_cast<size_t>(pattern_state.frame_image_count));

	const int32_t last_image_index = (pattern_state.frame_image_count > 2) ? (pattern_state.frame_image_count - 1) : 1;
	for (int32_t image_index = 0; image_index < pattern_state.frame_image_count; ++image_index)
	{
		const float normalized_position = static_cast<float>(image_index) / static_cast<float>(last_image_index);

		// We loop images:
		// From 0.0 to 0.5 we go from start to mid
		// From 0.5 to 1.0 we go from mid to start
		// We could reuse images directly instead of relaying on the cache. But this will help for video or random patterns
		float proportion = 2.0f * normalized_position;
		if (proportion > 1.0f)
		{
			proportion = 2.0f - proportion;
		}

		FrameImageCreationParameters image_creation_parameters{};
		if (pattern_state.pattern_type == PatternType::Gradient)
		{
			image_creation_parameters.gradient_top_left     = interpolate_color(pattern_state.start.gradient_top_left, pattern_state.mid.gradient_top_left, proportion);
			image_creation_parameters.gradient_top_right    = interpolate_color(pattern_state.start.gradient_top_right, pattern_state.mid.gradient_top_right, proportion);
			image_creation_parameters.gradient_bottom_left  = interpolate_color(pattern_state.start.gradient_bottom_left, pattern_state.mid.gradient_bottom_left, proportion);
			image_creation_parameters.gradient_bottom_right = interpolate_color(pattern_state.start.gradient_bottom_right, pattern_state.mid.gradient_bottom_right, proportion);
		}
		else
		{
			image_creation_parameters.check_color_a = interpolate_color(pattern_state.start.check_color_a, pattern_state.mid.check_color_a, proportion);
			image_creation_parameters.check_color_b = interpolate_color(pattern_state.start.check_color_b, pattern_state.mid.check_color_b, proportion);
			image_creation_parameters.check_size_x  = pattern_state.check_size_x;
			image_creation_parameters.check_size_y  = pattern_state.check_size_y;
		}
		image_creation_parameters.normalized_position      = normalized_position;
		image_creation_parameters.pattern_type             = pattern_state.pattern_type;
		image_creation_parameters.creation_parameters_hash = calculate_creation_parameters_hash(image_creation_parameters);
		frame_image_creation_parameters.push_back(image_creation_parameters);
	}
}

uint64_t FillBufferLinearImage::calculate_creation_parameters_hash(const FrameImageCreationParameters &image_creation_parameters) const
{
	size_t creation_parameters_hash_value = 0;
	vkb::hash_combine(creation_parameters_hash_value, width);
	vkb::hash_combine(creation_parameters_hash_value, height);
	vkb::hash_combine(creation_parameters_hash_value, static_cast<int32_t>(image_creation_parameters.pattern_type));
	if (image_creation_parameters.pattern_type == PatternType::Gradient)
	{
		vkb::hash_combine(creation_parameters_hash_value, calculate_color_hash(image_creation_parameters.gradient_top_left));
		vkb::hash_combine(creation_parameters_hash_value, calculate_color_hash(image_creation_parameters.gradient_top_right));
		vkb::hash_combine(creation_parameters_hash_value, calculate_color_hash(image_creation_parameters.gradient_bottom_left));
		vkb::hash_combine(creation_parameters_hash_value, calculate_color_hash(image_creation_parameters.gradient_bottom_right));
	}
	else
	{
		vkb::hash_combine(creation_parameters_hash_value, calculate_color_hash(image_creation_parameters.check_color_a));
		vkb::hash_combine(creation_parameters_hash_value, calculate_color_hash(image_creation_parameters.check_color_b));
		vkb::hash_combine(creation_parameters_hash_value, image_creation_parameters.check_size_x);
		vkb::hash_combine(creation_parameters_hash_value, image_creation_parameters.check_size_y);
	}
	return static_cast<uint64_t>(creation_parameters_hash_value);
}

uint64_t FillBufferLinearImage::calculate_cpu_image_pixels_hash(const BufferImageData &image_data) const
{
	size_t cpu_image_pixels_hash_value = 0;
	vkb::hash_combine(cpu_image_pixels_hash_value, image_data.width);
	vkb::hash_combine(cpu_image_pixels_hash_value, image_data.height);

	for (const ColorPixelData &pixel : image_data.data)
	{
		vkb::hash_combine(cpu_image_pixels_hash_value, pack_color_word(pixel));
	}

	return static_cast<uint64_t>(cpu_image_pixels_hash_value);
}

uint64_t FillBufferLinearImage::calculate_commands_hash(const std::vector<FillBufferLinearImage::CompressedFillCommand> &commands) const
{
	size_t commands_hash = 0;

	for (const CompressedFillCommand &cmd : commands)
	{
		vkb::hash_combine(commands_hash, cmd.offset);
		vkb::hash_combine(commands_hash, cmd.size);
		vkb::hash_combine(commands_hash, cmd.word);
	}

	return static_cast<uint64_t>(commands_hash);
}

const FillBufferLinearImage::FrameImageCreationParameters &FillBufferLinearImage::get_frame_image_creation_parameters(uint32_t frame_image_index)
{
	// Image creation parameters are stable until the sample is rebuilt.
	assert(frame_image_creation_parameters.size() > frame_image_index);
	LOGI("fill_buffer_linear_image: frame_image_creation_parameter_cache hit frame_image_index {} creation_parameters_hash {:#018x}",
	     frame_image_index,
	     frame_image_creation_parameters.at(static_cast<size_t>(frame_image_index)).creation_parameters_hash);

	return frame_image_creation_parameters.at(static_cast<size_t>(frame_image_index));
}

FillBufferLinearImage::BufferImageData FillBufferLinearImage::generate_cpu_image(const FrameImageCreationParameters &image_creation_parameters) const
{
	return (image_creation_parameters.pattern_type == PatternType::Gradient) ?
	           init_gradient(image_creation_parameters.gradient_top_left,
	                         image_creation_parameters.gradient_top_right,
	                         image_creation_parameters.gradient_bottom_left,
	                         image_creation_parameters.gradient_bottom_right) :
	           init_check_pattern(image_creation_parameters.check_color_a,
	                              image_creation_parameters.check_color_b,
	                              image_creation_parameters.check_size_x,
	                              image_creation_parameters.check_size_y);
}

void FillBufferLinearImage::clear_cpu_images()
{
	cpu_image_cache.clear();
	frame_image_creation_parameters.clear();
}

bool FillBufferLinearImage::prepare(const vkb::ApplicationOptions &options)
{
	if (!ApiVulkanSample::prepare(options))
	{
		return false;
	}

	if (get_render_context().get_swapchain().get_format() == VK_FORMAT_UNDEFINED)
	{
		throw std::runtime_error("fill_buffer_linear_image: requires a valid swapchain format before resource creation");
	}

	update_swapchain_image_usage_flags({VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT});

	create_resources();
	build_command_buffers();

	supported = true;
	prepared  = true;
	return true;
}

bool FillBufferLinearImage::resize(const uint32_t width_, const uint32_t height_)
{
	const bool resized = ApiVulkanSample::resize(width_, height_);
	if (!resized)
	{
		return false;
	}

	if (has_device())
	{
		rebuild_sample();
	}

	return true;
}

std::vector<FillBufferLinearImage::FillCommand> FillBufferLinearImage::build_row_fill_commands(const BufferImageData &image_data, const std::vector<bool> *skip_mask) const
{
	std::vector<FillCommand> commands{};

	// Emits one fill per pixel, or one fill per horizontal run when adjacent pixels carry
	// the same packed 32-bit color.
	for (uint32_t y = 0; y < image_data.height; ++y)
	{
		uint32_t x = 0;
		while (x < image_data.width)
		{
			const size_t pixel_index = static_cast<size_t>(y) * image_data.width + x;
			if (skip_mask != nullptr && (*skip_mask)[pixel_index])
			{
				++x;
				continue;
			}

			const uint32_t word       = pack_color_word(image_data.get_pixel(x, y));
			uint32_t       run_length = 1;
			if (pattern_state.combine_continuous_pixels)
			{
				while ((x + run_length) < image_data.width)
				{
					const size_t next_pixel_index = static_cast<size_t>(y) * image_data.width + (x + run_length);
					if ((skip_mask != nullptr && (*skip_mask)[next_pixel_index]) || pack_color_word(image_data.get_pixel(x + run_length, y)) != word)
					{
						break;
					}
					++run_length;
				}
			}

			const VkDeviceSize offset = linear_alias.image_offset +
			                            static_cast<VkDeviceSize>(y) * linear_alias.row_pitch +
			                            static_cast<VkDeviceSize>(x) * kBytesPerPixel;
			commands.push_back(FillCommand{
			    .offset = offset,
			    .size   = static_cast<VkDeviceSize>(run_length) * kBytesPerPixel,
			    .word   = word,
			});
			x += run_length;
		}
	}

	return commands;
}

std::vector<FillBufferLinearImage::FillCommand> FillBufferLinearImage::build_common_color_fill_commands(const BufferImageData &image_data, std::vector<bool> *covered_mask) const
{
	// Alternative strategy: flood the whole aliased image with its most frequent color first, then patch the
	// exceptions with row-local fills. This can beat the row-only strategy for checkerboards and sparse patterns.
	std::unordered_map<uint32_t, uint32_t> word_frequencies{};
	word_frequencies.reserve(image_data.data.size());
	for (const ColorPixelData &pixel : image_data.data)
	{
		++word_frequencies[pack_color_word(pixel)];
	}

	auto most_common_it = std::max_element(word_frequencies.begin(), word_frequencies.end(), [](const auto &lhs, const auto &rhs) {
		return lhs.second < rhs.second;
	});

	if (most_common_it == word_frequencies.end() || most_common_it->second < kMinimumRepeatedOffsets)
	{
		return {};
	}

	const uint32_t           base_word = most_common_it->first;
	std::vector<FillCommand> commands{};
	const VkDeviceSize       base_fill_size = align_down_to_fill_size(linear_alias.total_size);
	if (base_fill_size < sizeof(uint32_t))
	{
		return {};
	}

	commands.push_back(FillCommand{
	    .offset = linear_alias.image_offset,
	    .size   = base_fill_size,
	    .word   = base_word,
	});

	if (covered_mask != nullptr)
	{
		covered_mask->assign(image_data.data.size(), false);
		for (size_t pixel_index = 0; pixel_index < image_data.data.size(); ++pixel_index)
		{
			(*covered_mask)[pixel_index] = pack_color_word(image_data.data[pixel_index]) == base_word;
		}
	}

	return commands;
}

std::vector<FillBufferLinearImage::FillCommand> FillBufferLinearImage::calculate_optimized_fill_cmd(const BufferImageData &image_data) const
{
	std::vector<FillCommand> best_commands = build_row_fill_commands(image_data, nullptr);

	if (!pattern_state.combine_most_common && !pattern_state.combine_both)
	{
		return best_commands;
	}

	std::vector<bool> common_color_mask{};
	auto              common_color_commands = build_common_color_fill_commands(image_data, &common_color_mask);
	if (!common_color_commands.empty())
	{
		auto mixed_commands = common_color_commands;
		auto row_exceptions = build_row_fill_commands(image_data, &common_color_mask);
		mixed_commands.insert(mixed_commands.end(), row_exceptions.begin(), row_exceptions.end());
		if (mixed_commands.size() < best_commands.size())
		{
			best_commands = std::move(mixed_commands);
		}
	}

	if (pattern_state.combine_both)
	{
		// Exhaustive teaching mode: try "fill everything with one repeated word, then patch the remaining pixels"
		// for every color that appears often enough, and keep the cheapest recording.
		std::unordered_map<uint32_t, std::vector<size_t>> positions_by_word{};
		for (size_t pixel_index = 0; pixel_index < image_data.data.size(); ++pixel_index)
		{
			positions_by_word[pack_color_word(image_data.data[pixel_index])].push_back(pixel_index);
		}

		for (const auto &[word, positions] : positions_by_word)
		{
			if (positions.size() < kMinimumRepeatedOffsets)
			{
				continue;
			}

			std::vector<bool> custom_mask(image_data.data.size(), false);
			for (size_t pixel_index : positions)
			{
				custom_mask[pixel_index] = true;
			}

			std::vector<FillCommand> candidate{};
			candidate.push_back(FillCommand{
			    .offset = linear_alias.image_offset,
			    .size   = align_down_to_fill_size(linear_alias.total_size),
			    .word   = word,
			});

			for (size_t pixel_index = 0; pixel_index < custom_mask.size(); ++pixel_index)
			{
				custom_mask[pixel_index] = !custom_mask[pixel_index];
			}

			auto remainder = build_row_fill_commands(image_data, &custom_mask);
			candidate.insert(candidate.end(), remainder.begin(), remainder.end());
			if (candidate.size() < best_commands.size())
			{
				best_commands = std::move(candidate);
			}
		}
	}

	return best_commands;
}

std::vector<FillBufferLinearImage::FillCommand> FillBufferLinearImage::normalize_fill_commands(std::vector<FillCommand> commands) const
{
	// Canonicalize the fill plan so cache identity can depend on the actual recorded writes rather than on which
	// heuristic happened to append commands first.
	std::sort(commands.begin(), commands.end(), [](const FillCommand &lhs, const FillCommand &rhs) {
		if (lhs.offset != rhs.offset)
		{
			return lhs.offset < rhs.offset;
		}
		if (lhs.size != rhs.size)
		{
			return lhs.size < rhs.size;
		}
		return lhs.word < rhs.word;
	});

	for (size_t command_index = 1; command_index < commands.size(); ++command_index)
	{
		const FillCommand &previous = commands[command_index - 1];
		const FillCommand &current  = commands[command_index];
		assert((previous.offset + previous.size) <= current.offset && "Fill commands must not overlap");
	}

	return commands;
}

std::vector<FillBufferLinearImage::CompressedFillCommand> FillBufferLinearImage::compress_fill_commands(const std::vector<FillCommand> &commands) const
{
	std::vector<CompressedFillCommand> compressed_commands{};
	compressed_commands.reserve(commands.size());

	for (const FillCommand &command : commands)
	{
		assert(command.offset <= std::numeric_limits<uint32_t>::max());
		assert(command.size <= std::numeric_limits<uint32_t>::max());
		compressed_commands.push_back(CompressedFillCommand{
		    .offset = static_cast<uint32_t>(command.offset),
		    .size   = static_cast<uint32_t>(command.size),
		    .word   = command.word,
		});
	}

	return compressed_commands;
}

std::vector<FillBufferLinearImage::FillCommand> FillBufferLinearImage::decompress_fill_commands(const std::vector<CompressedFillCommand> &compressed_commands) const
{
	std::vector<FillCommand> commands{};
	commands.reserve(compressed_commands.size());

	for (const CompressedFillCommand &command : compressed_commands)
	{
		commands.push_back(FillCommand{
		    .offset = static_cast<VkDeviceSize>(command.offset),
		    .size   = static_cast<VkDeviceSize>(command.size),
		    .word   = command.word,
		});
	}

	return commands;
}

uint64_t FillBufferLinearImage::calculate_cpu_image_fill_commands_hash(uint64_t cpu_image_pixels_hash) const
{
	size_t cpu_image_fill_commands_hash_value = 0;
	vkb::hash_combine(cpu_image_fill_commands_hash_value, cpu_image_pixels_hash);
	vkb::hash_combine(cpu_image_fill_commands_hash_value, pattern_state.combine_continuous_pixels);
	vkb::hash_combine(cpu_image_fill_commands_hash_value, pattern_state.combine_most_common);
	vkb::hash_combine(cpu_image_fill_commands_hash_value, pattern_state.combine_both);
	vkb::hash_combine(cpu_image_fill_commands_hash_value, linear_alias.image_offset);
	vkb::hash_combine(cpu_image_fill_commands_hash_value, linear_alias.row_pitch);
	vkb::hash_combine(cpu_image_fill_commands_hash_value, linear_alias.total_size);
	return static_cast<uint64_t>(cpu_image_fill_commands_hash_value);
}

uint64_t FillBufferLinearImage::calculate_command_buffer_hash(uint32_t swapchain_image_index, VkImageLayout previous_swapchain_layout, uint64_t fill_commands_hash) const
{
	size_t recorded_command_buffer_hash_value = 0;

	vkb::hash_combine(recorded_command_buffer_hash_value, swapchain_image_index);
	vkb::hash_combine(recorded_command_buffer_hash_value, previous_swapchain_layout);
	vkb::hash_combine(recorded_command_buffer_hash_value, fill_commands_hash);

	return static_cast<uint64_t>(recorded_command_buffer_hash_value);
}

FillBufferLinearImage::FillCommandsCacheEntry &FillBufferLinearImage::find_or_create_fill_commands_cache_entry(uint32_t image_index, const FrameImageCreationParameters &image_creation_parameters)
{
	std::optional<BufferImageData> generated_image{};
	CachedCpuImageEntry            cached_cpu_image_entry{};
	bool                           valid_cached_cpu_image_entry = false;

	// Check if the CPU Image was already generated.
	// We do not get the CPU Image yet, just its hash.
	// If we create the image we store it, as it might be necessary later
	LOGI("fill_buffer_linear_image: cpu_image_cache input_parameters creation_parameters_hash {:#018x}", image_creation_parameters.creation_parameters_hash);
	for (const CachedCpuImageEntry &entry : cpu_image_cache.entries)
	{
		if (entry.creation_parameters_hash == image_creation_parameters.creation_parameters_hash)
		{
			LOGI("fill_buffer_linear_image: cpu_image_cache hit creation_parameters_hash {:#018x} cpu_image_pixels_hash {:#018x} no transient image returned",
			     image_creation_parameters.creation_parameters_hash,
			     entry.cpu_image_pixels_hash);
			cached_cpu_image_entry       = entry;
			valid_cached_cpu_image_entry = true;
			break;
		}
	}

	// This is a uncached CPU Image, so we need to create it.
	if (!valid_cached_cpu_image_entry)
	{
		LOGI("fill_buffer_linear_image: cpu_image_cache miss creation_parameters_hash {:#018x}", image_creation_parameters.creation_parameters_hash);

		generated_image                                 = generate_cpu_image(image_creation_parameters);
		cached_cpu_image_entry.creation_parameters_hash = image_creation_parameters.creation_parameters_hash;
		cached_cpu_image_entry.cpu_image_pixels_hash    = calculate_cpu_image_pixels_hash(generated_image.value());

		LOGI("fill_buffer_linear_image: cpu_image_cache generated transient image creation_parameters_hash {:#018x} cpu_image_pixels_hash {:#018x} ({}x{}, t={})",
		     cached_cpu_image_entry.creation_parameters_hash,
		     cached_cpu_image_entry.cpu_image_pixels_hash,
		     generated_image.value().width,
		     generated_image.value().height,
		     image_creation_parameters.normalized_position);

		cpu_image_cache.insert_or_assign(cpu_image_cache_size, cached_cpu_image_entry);
	}

	const uint64_t cpu_image_fill_commands_hash = calculate_cpu_image_fill_commands_hash(cached_cpu_image_entry.cpu_image_pixels_hash);
	LOGI("fill_buffer_linear_image: cpu_image_fill_commands_cache access cpu_image_pixels_hash {:#018x} cpu_image_fill_commands_hash {:#018x}",
	     cached_cpu_image_entry.cpu_image_pixels_hash,
	     cpu_image_fill_commands_hash);

	std::optional<CpuImageFillCommandsCacheEntry> cached_cpu_fill_commands_entry{};

	// We try to use the hash of the CPU Image combined with current settings to get the fill commands hash
	for (const CpuImageFillCommandsCacheEntry &entry : cpu_image_fill_commands_cache.entries)
	{
		if (entry.cpu_image_fill_commands_hash == cpu_image_fill_commands_hash)
		{
			LOGI("fill_buffer_linear_image: cpu_image_fill_commands_cache hit cpu_image_fill_commands_hash {:#018x} fill_commands_hash {:#018x}",
			     entry.cpu_image_fill_commands_hash,
			     entry.fill_commands_hash);
			cached_cpu_fill_commands_entry = entry;
			break;
		}
	}

	// We did not find the hash commands so we need to create them
	if (cached_cpu_fill_commands_entry.has_value())
	{
		const uint32_t fill_cache_index =
		    fill_commands_cache.find_index([fill_commands_hash = cached_cpu_fill_commands_entry->fill_commands_hash](const FillCommandsCacheEntry &entry) {
			    return entry.fill_commands_hash == fill_commands_hash;
		    });
		if (fill_cache_index != FillBufferLinearImage::FixedRingCache<FillCommandsCacheEntry>::kInvalidIndex)
		{
			LOGI("fill_buffer_linear_image: fill_commands_cache hit image_index {} fill_commands_hash {:#018x} cache_index {}",
			     image_index,
			     cached_cpu_fill_commands_entry->fill_commands_hash,
			     fill_cache_index);
			return fill_commands_cache.at(fill_cache_index);
		}
	}

	const BufferImageData *image_data = nullptr;
	BufferImageData        regenerated_image{};
	if (generated_image.has_value())
	{
		LOGI("fill_buffer_linear_image: fill_commands_cache miss image_index {} reuses transient cpu image from cpu_image_cache", image_index);
		image_data = &generated_image.value();
	}
	else
	{
		LOGI("fill_buffer_linear_image: fill_commands_cache miss image_index {} regenerates cpu image from creation_parameters_hash {:#018x}",
		     image_index,
		     image_creation_parameters.creation_parameters_hash);
		regenerated_image = generate_cpu_image(image_creation_parameters);
		image_data        = &regenerated_image;
	}

	FillCommandsCacheEntry new_entry{};
	new_entry.commands           = compress_fill_commands(normalize_fill_commands(calculate_optimized_fill_cmd(*image_data)));
	new_entry.fill_commands_hash = calculate_commands_hash(new_entry.commands);
	LOGI("fill_buffer_linear_image: fill_commands_cache miss image_index {} fill_commands_hash {:#018x}",
	     image_index,
	     new_entry.fill_commands_hash);

	FillCommandsCacheEntry &stored_entry = fill_commands_cache.insert_or_assign(fill_commands_cache_size, std::move(new_entry));

	if (!cached_cpu_fill_commands_entry.has_value())
	{
		CpuImageFillCommandsCacheEntry cpu_image_fill_cache_entry{};
		cpu_image_fill_cache_entry.cpu_image_fill_commands_hash = cpu_image_fill_commands_hash;
		cpu_image_fill_cache_entry.fill_commands_hash           = stored_entry.fill_commands_hash;
		LOGI("fill_buffer_linear_image: cpu_image_fill_commands_cache miss cpu_image_fill_commands_hash {:#018x} fill_commands_hash {:#018x}",
		     cpu_image_fill_cache_entry.cpu_image_fill_commands_hash,
		     cpu_image_fill_cache_entry.fill_commands_hash);
		cpu_image_fill_commands_cache.insert_or_assign(cpu_image_fill_commands_cache_size, cpu_image_fill_cache_entry);
	}

	return stored_entry;
}

void FillBufferLinearImage::record_alias_fill_and_copy(VkCommandBuffer                     command_buffer,
                                                       uint32_t                            swapchain_image_index,
                                                       uint32_t                            frame_image_index,
                                                       const FrameImageCreationParameters &image_creation_parameters,
                                                       uint32_t                            cache_index)
{
	const VkImage       swapchain_image      = get_render_context().get_swapchain().get_images()[swapchain_image_index];
	const VkImageLayout swapchain_old_layout = swapchain_image_layouts.at(swapchain_image_index);

	const VkImageSubresourceRange color_range{
	    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
	    .baseMipLevel   = 0,
	    .levelCount     = 1,
	    .baseArrayLayer = 0,
	    .layerCount     = 1,
	};

	const FillCommandsCacheEntry &fill_commands                              = find_or_create_fill_commands_cache_entry(frame_image_index, image_creation_parameters);
	auto                          fill_command_sequence                      = decompress_fill_commands(fill_commands.commands);
	recorded_command_buffer_cache.entries.at(cache_index).fill_command_count = static_cast<uint32_t>(fill_command_sequence.size());
	LOGI("fill_buffer_linear_image: recorded_command_buffer_cache cache_index {} uses {} fill commands",
	     cache_index,
	     recorded_command_buffer_cache.entries.at(cache_index).fill_command_count);

	// The aliased memory was left in GENERAL after resource creation or the previous frame's copy. Keeping the image
	// in GENERAL while the buffer interpretation is being written avoids pretending the bytes already satisfy a
	// specialized image layout.
	VkImageMemoryBarrier to_general{
	    .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .srcAccessMask       = 0,
	    .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
	    .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = linear_alias.image,
	    .subresourceRange    = color_range,
	};

	vkCmdPipelineBarrier(command_buffer,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0,
	                     0,
	                     nullptr,
	                     0,
	                     nullptr,
	                     1,
	                     &to_general);

	for (const FillCommand &fill_command : fill_command_sequence)
	{
		// vkCmdFillBuffer writes one 32-bit pattern repeatedly across [offset, offset + size). Because the buffer
		// aliases the image, these transfer writes materialize texel data in the linear image storage.
		vkCmdFillBuffer(command_buffer, linear_alias.buffer, fill_command.offset, fill_command.size, fill_command.word);
	}

	VkBufferMemoryBarrier fill_visible{
	    .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
	    .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .buffer              = linear_alias.buffer,
	    .offset              = 0,
	    .size                = VK_WHOLE_SIZE,
	};

	VkImageMemoryBarrier to_transfer_src{
	    .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
	    .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
	    .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = linear_alias.image,
	    .subresourceRange    = color_range,
	};

	vkCmdPipelineBarrier(command_buffer,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0,
	                     0,
	                     nullptr,
	                     1,
	                     &fill_visible,
	                     1,
	                     &to_transfer_src);

	VkImageMemoryBarrier swapchain_to_transfer_dst{
	    .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .srcAccessMask       = 0,
	    .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .oldLayout           = swapchain_old_layout,
	    .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = swapchain_image,
	    .subresourceRange    = color_range,
	};

	vkCmdPipelineBarrier(command_buffer,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0,
	                     0,
	                     nullptr,
	                     0,
	                     nullptr,
	                     1,
	                     &swapchain_to_transfer_dst);

	VkImageCopy copy_region{
	    .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
	    .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
	    .extent         = {width, height, 1},
	};
	vkCmdCopyImage(command_buffer, linear_alias.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

	VkImageMemoryBarrier swapchain_to_present{
	    .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .dstAccessMask       = 0,
	    .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    .newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = swapchain_image,
	    .subresourceRange    = color_range,
	};

	VkImageMemoryBarrier linear_back_to_general{
	    .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
	    .dstAccessMask       = 0,
	    .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	    .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = linear_alias.image,
	    .subresourceRange    = color_range,
	};

	std::array<VkImageMemoryBarrier, 2> final_barriers = {linear_back_to_general, swapchain_to_present};
	vkCmdPipelineBarrier(command_buffer,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	                     0,
	                     0,
	                     nullptr,
	                     0,
	                     nullptr,
	                     static_cast<uint32_t>(final_barriers.size()),
	                     final_barriers.data());
}

void FillBufferLinearImage::build_command_buffer_cache()
{
	// Bootstrap the cache stack for the current sample state. Frame Image Creation Parameter Cache comes first, while
	// fill commands and recorded command buffers remain lazy.
	build_frame_image_creation_parameters();
}

void FillBufferLinearImage::build_command_buffers()
{
	if (!resources_ready)
	{
		return;
	}

	destroy_command_buffer_cache();
	build_command_buffer_cache();
}

void FillBufferLinearImage::rebuild_sample()
{
	if (!has_device())
	{
		return;
	}

	get_device().wait_idle();
	create_resources();
	build_command_buffers();
}

uint32_t FillBufferLinearImage::get_or_create_command_buffer_for_current_frame(uint32_t swapchain_image_index, uint32_t frame_image_index)
{
	const FrameImageCreationParameters &image_creation_parameters = get_frame_image_creation_parameters(frame_image_index);
	const FillCommandsCacheEntry &      fill_commands             = find_or_create_fill_commands_cache_entry(frame_image_index, image_creation_parameters);
	const uint64_t                      fill_commands_hash        = fill_commands.fill_commands_hash;
	const VkImageLayout                 previous_swapchain_layout = swapchain_image_layouts.at(swapchain_image_index);
	const uint64_t                      command_buffer_hash       = calculate_command_buffer_hash(swapchain_image_index, previous_swapchain_layout, fill_commands_hash);
	LOGI("fill_buffer_linear_image: recorded_command_buffer_cache access swapchain_image_index {} frame_image_index {} command_hash {:#018x}",
	     swapchain_image_index,
	     frame_image_index,
	     command_buffer_hash);
	const uint32_t cached_index = recorded_command_buffer_cache.entries.find_index([command_buffer_hash](const RecordedCommandBufferEntry &entry) {
		return entry.recorded_command_buffer_hash == command_buffer_hash;
	});
	if (cached_index != FillBufferLinearImage::FixedRingCache<RecordedCommandBufferEntry>::kInvalidIndex)
	{
		LOGI("fill_buffer_linear_image: recorded_command_buffer_cache hit swapchain_image_index {} frame_image_index {} command_hash {:#018x} cache_index {}",
		     swapchain_image_index,
		     frame_image_index,
		     command_buffer_hash,
		     cached_index);
		return cached_index;
	}
	LOGI("fill_buffer_linear_image: recorded_command_buffer_cache miss swapchain_image_index {} frame_image_index {} command_hash {:#018x}",
	     swapchain_image_index,
	     frame_image_index,
	     command_buffer_hash);

	if (cached_cmd_pool == VK_NULL_HANDLE)
	{
		// We cache command buffers, so we need to override the command pool in the framework
		LOGI("Creating command pool");
		VkCommandPoolCreateInfo command_pool_info{
		    .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		    .queueFamilyIndex = get_device().get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0).get_family_index(),
		};
		VK_CHECK(vkCreateCommandPool(get_device().get_handle(), &command_pool_info, nullptr, &cached_cmd_pool));
	}

	uint32_t                   cache_index    = 0;
	VkCommandBuffer            command_buffer = VK_NULL_HANDLE;
	RecordedCommandBufferEntry entry{};
	if (recorded_command_buffer_cache.entries.size() < command_buffer_cache_size)
	{
		VkCommandBufferAllocateInfo allocate_info = vkb::initializers::command_buffer_allocate_info(cached_cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK(vkAllocateCommandBuffers(get_device().get_handle(), &allocate_info, &command_buffer));
		entry.command_buffer = command_buffer;
		recorded_command_buffer_cache.entries.insert_or_assign(command_buffer_cache_size, entry, &cache_index);
	}
	else
	{
		cache_index    = recorded_command_buffer_cache.entries.next_slot;
		entry          = recorded_command_buffer_cache.entries.at(cache_index);
		command_buffer = entry.command_buffer;
		LOGI("fill_buffer_linear_image: recorded_command_buffer_cache evict cache_index {} old_command_hash {:#018x}",
		     cache_index,
		     entry.recorded_command_buffer_hash);

		vkFreeCommandBuffers(get_device().get_handle(), cached_cmd_pool, 1, &command_buffer);
		VkCommandBufferAllocateInfo allocate_info = vkb::initializers::command_buffer_allocate_info(cached_cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK(vkAllocateCommandBuffers(get_device().get_handle(), &allocate_info, &command_buffer));
		entry.command_buffer = command_buffer;
		recorded_command_buffer_cache.entries.insert_or_assign(command_buffer_cache_size, entry, &cache_index);
	}

	recorded_command_buffer_cache.entries.at(cache_index).recorded_command_buffer_hash = command_buffer_hash;
	recorded_command_buffer_cache.entries.at(cache_index).fill_command_count           = 0;

	VkCommandBufferBeginInfo begin_info = vkb::initializers::command_buffer_begin_info();
	VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));
	record_alias_fill_and_copy(command_buffer,
	                           swapchain_image_index,
	                           frame_image_index,
	                           image_creation_parameters,
	                           cache_index);
	VK_CHECK(vkEndCommandBuffer(command_buffer));
	return cache_index;
}

void FillBufferLinearImage::render(float delta_time)
{
	if (!prepared || !supported)
	{
		return;
	}

	ApiVulkanSample::prepare_frame();

	const uint32_t frame_image_count     = std::max<uint32_t>(static_cast<uint32_t>(pattern_state.frame_image_count), 1u);
	const uint32_t swapchain_image_count = std::max<uint32_t>(static_cast<uint32_t>(get_render_context().get_swapchain().get_images().size()), 1u);
	const uint32_t frame_image_index     = static_cast<uint32_t>(frame_id % frame_image_count);

	if (current_buffer >= swapchain_image_count)
	{
		throw std::runtime_error("Acquired swapchain image index exceeds the current swapchain image count");
	}

	const uint32_t        cached_command_index = get_or_create_command_buffer_for_current_frame(current_buffer, frame_image_index);
	const VkCommandBuffer command_buffer       = recorded_command_buffer_cache.entries.at(cached_command_index).command_buffer;

	LOGI("fill_buffer_linear_image: render frame_id {} frame_image_index {} swapchain_image_index {} command_buffer_index {}",
	     frame_id,
	     frame_image_index,
	     current_buffer,
	     cached_command_index);

	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command_buffer;
	VK_CHECK(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
	ApiVulkanSample::submit_frame();
	swapchain_image_layouts.at(current_buffer) = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	++frame_id;
}

std::unique_ptr<vkb::VulkanSampleC> create_fill_buffer_linear_image()
{
	return std::make_unique<FillBufferLinearImage>();
}
