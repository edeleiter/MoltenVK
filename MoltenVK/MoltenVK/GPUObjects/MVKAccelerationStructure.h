/*
 * MVKAccelerationStructure.h
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

#include "MVKDevice.h"


#pragma mark MVKAccelerationStructure

/**
 * Represents a Vulkan VK_KHR_acceleration_structure object (BLAS or TLAS), backed by an
 * id<MTLAccelerationStructure>. Part of the VulkanEd MoltenVK fork's ray-query support:
 * the engine builds the TLAS host-side each frame and traverses it inline in a compute
 * shader (no ray-tracing pipeline / SBT). MoltenVK is MRC, so the Metal object is manually
 * retained/released here.
 */
class MVKAccelerationStructure : public MVKVulkanAPIDeviceObject {

public:

	/** Returns the Vulkan type of this object. */
	VkObjectType getVkObjectType() override { return VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR; }

	/** Returns the debug report object type of this object. */
	VkDebugReportObjectTypeEXT getVkDebugReportObjectType() override { return VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT; }

	/** Whether this is a bottom-level (BLAS) or top-level (TLAS) structure. */
	VkAccelerationStructureTypeKHR getType() { return _type; }

	/** The Metal acceleration structure backing this object (nil until allocated at build). */
	id<MTLAccelerationStructure> getMTLAccelerationStructure() { return _mtlAccelStruct; }

	/** Sets (retains) the Metal acceleration structure, releasing any previous one (MRC). */
	void setMTLAccelerationStructure(id<MTLAccelerationStructure> mtlAS);

	/** The synthetic device address handed out by vkGetAccelerationStructureDeviceAddressKHR (0 until assigned). */
	VkDeviceAddress getDeviceAddress() { return _deviceAddress; }

	/** Records the synthetic device address (token) for this structure. */
	void setDeviceAddress(VkDeviceAddress addr) { _deviceAddress = addr; }

	/** TLAS only: the BLAS this TLAS references, recorded at build time for compute-encoder residency. */
	MVKSmallVector<MVKAccelerationStructure*>& getReferencedBLAS() { return _referencedBLAS; }

	void destroy() override;

	/**
	 * Builds the Metal acceleration-structure descriptor for a Vulkan build info. BLAS (BOTTOM_LEVEL) →
	 * an MTLPrimitiveAccelerationStructureDescriptor of triangle geometries (R32G32B32 positions, vertex
	 * data resolved from the buffer device address); TLAS (TOP_LEVEL) → an
	 * MTLInstanceAccelerationStructureDescriptor sized by instance count (its instances are materialized
	 * at build time). pPrimitiveCounts is per-geometry (BuildSizes' maxPrimitiveCounts, or the build ranges).
	 * Shared by vkGetAccelerationStructureBuildSizesKHR and the build command.
	 */
	static MTLAccelerationStructureDescriptor* getMTLDescriptor(MVKDevice* device,
															   const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
															   const uint32_t* pPrimitiveCounts);


#pragma mark Construction

	MVKAccelerationStructure(MVKDevice* device, const VkAccelerationStructureCreateInfoKHR* pCreateInfo)
		: MVKVulkanAPIDeviceObject(device), _type(pCreateInfo->type) {}

protected:
	void propagateDebugName() override {}
	void detachMetal();

	id<MTLAccelerationStructure> _mtlAccelStruct = nil;
	VkAccelerationStructureTypeKHR _type;
	VkDeviceAddress _deviceAddress = 0;
	MVKSmallVector<MVKAccelerationStructure*> _referencedBLAS;
};
