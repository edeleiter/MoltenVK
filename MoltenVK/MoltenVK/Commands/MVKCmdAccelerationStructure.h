/*
 * MVKCmdAccelerationStructure.h
 *
 * Copyright (c) 2015-2026 The Brenwill Workshop Ltd. (http://www.brenwill.com)
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

#include "MVKCommand.h"
#include "MVKSmallVector.h"

#import <Metal/Metal.h>

class MVKAccelerationStructure;
class MVKDevice;


#pragma mark -
#pragma mark MVKCmdBuildAccelerationStructures

/**
 * Vulkan command to build acceleration structures (VK_KHR_ray_query fork). Defers to encode(): BOTTOM_LEVEL
 * (BLAS) builds encode an MTLPrimitiveAccelerationStructure, and TOP_LEVEL (TLAS) builds host-read the instance
 * buffer, resolve each accelerationStructureReference to its BLAS (pointer-as-token), and encode an
 * MTLInstanceAccelerationStructure — both on an MTLAccelerationStructureCommandEncoder.
 *
 * The class name must be exactly MVKCmdBuildAccelerationStructures (token-pasted by MVKCommandTypePools.def
 * into the pool member + getTypePool body), and must stay default-constructible (the pool does `new T()`).
 */
class MVKCmdBuildAccelerationStructures : public MVKCommand {

public:
	VkResult setContent(MVKCommandBuffer* cmdBuff,
						uint32_t infoCount,
						const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
						const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos);

	void encode(MVKCommandEncoder* cmdEncoder) override;

protected:
	MVKCommandTypePool<MVKCommand>* getTypePool(MVKCommandPool* cmdPool) override;

	// One captured build per pInfos[i]. Geometries + their primitive counts/offsets are flattened across all
	// builds into the parallel _geometries/_primitiveCounts/_primitiveOffsets vectors; firstGeometryIndex
	// slices this build's range.
	struct MVKASBuild {
		MVKAccelerationStructure* dst;
		MVKAccelerationStructure* src;   // MODE_UPDATE (refit) source; == dst for in-place. null for MODE_BUILD.
		VkAccelerationStructureTypeKHR type;
		VkBuildAccelerationStructureModeKHR mode;
		VkBuildAccelerationStructureFlagsKHR flags;   // AllowUpdate → the built AS gets MTLAccelerationStructureUsageRefit
		VkDeviceAddress scratchAddress;
		uint32_t geometryCount;
		uint32_t firstGeometryIndex;
	};

	void encodeBLAS(MVKCommandEncoder* cmdEncoder, MVKDevice* dev, const MVKASBuild& b);
	void encodeTLAS(MVKCommandEncoder* cmdEncoder, MVKDevice* dev, const MVKASBuild& b);

	MVKSmallVector<MVKASBuild> _builds;
	// INVARIANT: _geometries, _primitiveCounts and _primitiveOffsets are pushed in lockstep (one entry per
	// geometry), so index i of each aligns — getMTLDescriptor + encodeTLAS rely on this.
	MVKSmallVector<VkAccelerationStructureGeometryKHR> _geometries;
	MVKSmallVector<uint32_t> _primitiveCounts;
	MVKSmallVector<uint32_t> _primitiveOffsets;
};
