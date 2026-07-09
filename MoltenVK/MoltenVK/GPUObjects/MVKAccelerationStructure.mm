/*
 * MVKAccelerationStructure.mm
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

#include "MVKAccelerationStructure.h"


#pragma mark MVKAccelerationStructure

void MVKAccelerationStructure::setMTLAccelerationStructure(id<MTLAccelerationStructure> mtlAS) {
	if (_mtlAccelStruct == mtlAS) { return; }
	[mtlAS retain];
	[_mtlAccelStruct release];
	_mtlAccelStruct = mtlAS;
}

void MVKAccelerationStructure::destroy() {
	detachMetal();
	MVKVulkanAPIDeviceObject::destroy();
}

// Potentially called from destroy() (and safe to call more than once), so null everything out.
void MVKAccelerationStructure::detachMetal() {
	[_mtlAccelStruct release];
	_mtlAccelStruct = nil;
	_referencedBLAS.clear();
}

MTLAccelerationStructureDescriptor* MVKAccelerationStructure::getMTLDescriptor(
	MVKDevice* device,
	const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
	const uint32_t* pPrimitiveCounts) {

	if (pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
		// TLAS: sized by instance count. The actual instance array is materialized at build time (the shim).
		auto* instDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
		instDesc.instanceCount = pPrimitiveCounts ? pPrimitiveCounts[0] : 0;
		return instDesc;
	}

	// BLAS: one Metal triangle geometry per Vulkan triangle geometry (the engine uses positions-only soup).
	auto* geoms = [NSMutableArray<MTLAccelerationStructureGeometryDescriptor*> array];
	for (uint32_t i = 0; i < pBuildInfo->geometryCount; i++) {
		const VkAccelerationStructureGeometryKHR& g = pBuildInfo->pGeometries ? pBuildInfo->pGeometries[i]
																			  : *pBuildInfo->ppGeometries[i];
		if (g.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR) { continue; }
		const VkAccelerationStructureGeometryTrianglesDataKHR& tri = g.geometry.triangles;

		auto* triDesc = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
		triDesc.triangleCount = pPrimitiveCounts ? pPrimitiveCounts[i] : 0;
		triDesc.vertexFormat  = MTLAttributeFormatFloat3;   // VK_FORMAT_R32G32B32_SFLOAT
		triDesc.vertexStride  = tri.vertexStride;
		NSUInteger vbOffset = 0;
		triDesc.vertexBuffer = device->getMTLBufferForDeviceAddress(tri.vertexData.deviceAddress, &vbOffset);
		triDesc.vertexBufferOffset = vbOffset;
		triDesc.opaque = mvkIsAnyFlagEnabled(g.flags, VK_GEOMETRY_OPAQUE_BIT_KHR);
		[geoms addObject: triDesc];
	}
	auto* primDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
	primDesc.geometryDescriptors = geoms;
	return primDesc;
}
