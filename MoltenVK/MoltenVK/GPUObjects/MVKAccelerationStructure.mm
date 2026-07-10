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
	if (_mtlAccelStruct) {
		// Paired with the makeResident() at create (vkCreateAccelerationStructureKHR) — the residency set
		// retains the allocation, so without this the AS never deallocates and the set grows unbounded.
		getDevice()->removeResidency(_mtlAccelStruct);
		getDevice()->removeAccelerationStructure(this);   // paired with addAccelerationStructure() at create
		[_mtlAccelStruct release];
		_mtlAccelStruct = nil;
	}
	_referencedBLAS.clear();
}

// Map a Vulkan triangle-position format to the Metal attribute format that MTLAccelerationStructureTriangleGeometryDescriptor
// accepts. The VulkanEd fork supports Float3 (what the engine's soup emits) and Half3 (compressed positions). Anything
// else returns MTLAttributeFormatInvalid so the caller fails loud rather than building a geometrically-wrong BLAS.
static MTLAttributeFormat mvkMTLAttributeFormatFromVkFormat(VkFormat vkFormat) {
	switch (vkFormat) {
		case VK_FORMAT_R32G32B32_SFLOAT: return MTLAttributeFormatFloat3;
		case VK_FORMAT_R16G16B16_SFLOAT: return MTLAttributeFormatHalf3;
		default:                         return MTLAttributeFormatInvalid;
	}
}

// Map a Vulkan index type to the Metal index type (out-param), returning false for an unsupported width (e.g. UINT8) so
// the caller fails loud. VK_INDEX_TYPE_NONE_KHR is handled by the non-indexed path and never reaches here.
static bool mvkMTLIndexTypeFromVkIndexType(VkIndexType vkIndexType, MTLIndexType* pMTLIndexType) {
	switch (vkIndexType) {
		case VK_INDEX_TYPE_UINT16: *pMTLIndexType = MTLIndexTypeUInt16; return true;
		case VK_INDEX_TYPE_UINT32: *pMTLIndexType = MTLIndexTypeUInt32; return true;
		default:                   return false;
	}
}

MTLAccelerationStructureDescriptor* MVKAccelerationStructure::getMTLDescriptor(
	MVKDevice* device,
	const VkAccelerationStructureBuildGeometryInfoKHR* pBuildInfo,
	const uint32_t* pPrimitiveCounts) {

	// AllowUpdate → the built AS must carry MTLAccelerationStructureUsageRefit so a later refitAccelerationStructure:
	// (MODE_UPDATE) is legal AND so accelerationStructureSizesWithDescriptor: reports a real refitScratchBufferSize.
	// getMTLDescriptor is shared by BuildSizes + the initial MODE_BUILD, so both stamp the same usage from the flags.
	MTLAccelerationStructureUsage mtlUsage = MTLAccelerationStructureUsageNone;
	if (pBuildInfo->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) {
		mtlUsage |= MTLAccelerationStructureUsageRefit;
	}

	if (pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
		// TLAS: sized by instance count. The actual instance array is materialized at build time (the shim).
		auto* instDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
		instDesc.instanceCount = pPrimitiveCounts ? pPrimitiveCounts[0] : 0;
		instDesc.usage = mtlUsage;
		return instDesc;
	}

	// BLAS: one Metal triangle geometry per Vulkan triangle geometry (the engine uses positions-only soup).
	auto* geoms = [NSMutableArray<MTLAccelerationStructureGeometryDescriptor*> array];
	for (uint32_t i = 0; i < pBuildInfo->geometryCount; i++) {
		const VkAccelerationStructureGeometryKHR& g = pBuildInfo->pGeometries ? pBuildInfo->pGeometries[i]
																			  : *pBuildInfo->ppGeometries[i];
		if (g.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR) { continue; }
		const VkAccelerationStructureGeometryTrianglesDataKHR& tri = g.geometry.triangles;

		// The fork accepts the position formats mapped by mvkMTLAttributeFormatFromVkFormat (Float3 / Half3) and index
		// types UINT16/UINT32 (plus non-indexed). Anything else is geometry we would build WRONG, so fail LOUD and skip
		// it rather than silently corrupting the BLAS. (Same call path as BuildSizes → the skip is consistent between
		// sizing and building.) static method → object-less reportMessage (MVKLogError needs `this`).
		MTLAttributeFormat mtlVtxFormat = mvkMTLAttributeFormatFromVkFormat(tri.vertexFormat);
		if (mtlVtxFormat == MTLAttributeFormatInvalid) {
			MVKBaseObject::reportMessage(nullptr, MVK_CONFIG_LOG_LEVEL_ERROR,
				"vkCmdBuildAccelerationStructuresKHR: BLAS geometry %u has unsupported vertexFormat=%d — the VulkanEd "
				"fork maps only VK_FORMAT_R32G32B32_SFLOAT and VK_FORMAT_R16G16B16_SFLOAT; skipping this geometry.",
				i, (int)tri.vertexFormat);
			continue;
		}

		auto* triDesc = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
		triDesc.triangleCount = pPrimitiveCounts ? pPrimitiveCounts[i] : 0;   // triangle count, indexed or not
		triDesc.vertexFormat  = mtlVtxFormat;
		triDesc.vertexStride  = tri.vertexStride;
		NSUInteger vbOffset = 0;
		triDesc.vertexBuffer = device->getMTLBufferForDeviceAddress(tri.vertexData.deviceAddress, &vbOffset);
		triDesc.vertexBufferOffset = vbOffset;

		// Indexed geometry: wire the index buffer so Metal reads shared vertices via indices (what desktop Vulkan RT
		// drivers do; the engine's soup path stays non-indexed and skips this block). getMTLBufferForDeviceAddress is
		// address-generic — the same resolver used for the vertex/instance/scratch buffers.
		if (tri.indexType != VK_INDEX_TYPE_NONE_KHR) {
			MTLIndexType mtlIndexType;
			if ( !mvkMTLIndexTypeFromVkIndexType(tri.indexType, &mtlIndexType) ) {
				MVKBaseObject::reportMessage(nullptr, MVK_CONFIG_LOG_LEVEL_ERROR,
					"vkCmdBuildAccelerationStructuresKHR: BLAS geometry %u has unsupported indexType=%d — the VulkanEd "
					"fork maps only UINT16/UINT32; skipping this geometry.", i, (int)tri.indexType);
				continue;
			}
			NSUInteger ibOffset = 0;
			id<MTLBuffer> ibuf = device->getMTLBufferForDeviceAddress(tri.indexData.deviceAddress, &ibOffset);
			if ( !ibuf ) {
				MVKBaseObject::reportMessage(nullptr, MVK_CONFIG_LOG_LEVEL_ERROR,
					"vkCmdBuildAccelerationStructuresKHR: BLAS geometry %u indexData address did not resolve to a Metal "
					"buffer; skipping this geometry.", i);
				continue;
			}
			triDesc.indexBuffer       = ibuf;
			triDesc.indexBufferOffset = ibOffset;
			triDesc.indexType         = mtlIndexType;
		}

		triDesc.opaque = mvkIsAnyFlagEnabled(g.flags, VK_GEOMETRY_OPAQUE_BIT_KHR);
		[geoms addObject: triDesc];
	}
	auto* primDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
	primDesc.geometryDescriptors = geoms;
	primDesc.usage = mtlUsage;   // Refit when ALLOW_UPDATE (see above) — makes MODE_UPDATE refit legal
	return primDesc;
}
