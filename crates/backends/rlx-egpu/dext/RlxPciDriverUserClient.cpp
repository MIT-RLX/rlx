// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The userspace-facing half. Four external methods plus the dual-purpose
// CopyClientMemoryForType, which is how BARs and DMA memory both arrive in the
// caller's address space.

#include "RlxPciDriverUserClient.h"
#include "RlxPciDriver.h"
#include <DriverKit/DriverKit.h>
#include <DriverKit/OSSharedPtr.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct RlxPciDriverUserClient_IVars {
    OSSharedPtr<RlxPciDriver> provider = nullptr;

    RlxDmaAllocation *allocations = nullptr;
    size_t count = 0;
    size_t capacity = 0;

    // Not thread-safe: one client at a time, by construction.
    int reserve(size_t needed)
    {
        if (needed <= capacity) return 0;
        size_t grown = capacity ? capacity * 2 : 16;
        while (grown < needed) grown *= 2;

        auto *fresh = IONewZero(RlxDmaAllocation, grown);
        if (!fresh) return -1;
        if (allocations && count) memcpy(fresh, allocations, count * sizeof(RlxDmaAllocation));
        IOSafeDeleteNULL(allocations, RlxDmaAllocation, capacity);
        allocations = fresh;
        capacity = grown;
        return 0;
    }
};

bool RlxPciDriverUserClient::init()
{
    if (!super::init()) return false;
    ivars = IONewZero(RlxPciDriverUserClient_IVars, 1);
    return ivars != nullptr;
}

void RlxPciDriverUserClient::free()
{
    IOSafeDeleteNULL(ivars, RlxPciDriverUserClient_IVars, 1);
    super::free();
}

kern_return_t IMPL(RlxPciDriverUserClient, Start)
{
    if (!provider) return kIOReturnBadArgument;
    kern_return_t err = Start(provider, SUPERDISPATCH);
    if (err) return err;
    ivars->provider = OSSharedPtr(OSDynamicCast(RlxPciDriver, provider), OSRetain);
    return kIOReturnSuccess;
}

kern_return_t IMPL(RlxPciDriverUserClient, Stop)
{
    // Release every pinned DMA mapping. Leaving one pinned after the client is
    // gone keeps physical memory reserved for the lifetime of the extension.
    if (ivars) {
        for (size_t i = 0; i < ivars->count; i++) {
            if (ivars->allocations[i].command) {
                ivars->allocations[i].command->CompleteDMA(kIODMACommandCompleteDMANoOptions);
                ivars->allocations[i].command->release();
                ivars->allocations[i].command = nullptr;
            }
        }
        ivars->count = 0;
        IOSafeDeleteNULL(ivars->allocations, RlxDmaAllocation, ivars->capacity);
        ivars->provider.reset();
    }
    return Stop(provider, SUPERDISPATCH);
}

// Not an IMPL: ExternalMethod is a plain virtual override, so it is defined
// with its own parameter names rather than the ones iig would generate.
kern_return_t RlxPciDriverUserClient::ExternalMethod(uint64_t selector,
                                                     IOUserClientMethodArguments *arguments,
                                                     const IOUserClientMethodDispatch *dispatch,
                                                     OSObject *target,
                                                     void *reference)
{
    (void)dispatch; (void)target; (void)reference;
    if (!ivars->provider.get()) return kIOReturnNotAttached;

    switch (selector) {
    case kRlxPciReadConfig: {
        if (arguments->scalarInputCount != 2 || arguments->scalarOutputCount < 1)
            return kIOReturnBadArgument;
        uint32_t value = 0;
        kern_return_t err = ivars->provider->ConfigRead((uint32_t)arguments->scalarInput[0],
                                                        (uint32_t)arguments->scalarInput[1],
                                                        &value);
        if (err) return err;
        arguments->scalarOutput[0] = value;
        arguments->scalarOutputCount = 1;
        return kIOReturnSuccess;
    }

    case kRlxPciWriteConfig:
        if (arguments->scalarInputCount != 3) return kIOReturnBadArgument;
        return ivars->provider->ConfigWrite((uint32_t)arguments->scalarInput[0],
                                            (uint32_t)arguments->scalarInput[1],
                                            (uint32_t)arguments->scalarInput[2]);

    case kRlxPciReset:
        return ivars->provider->ResetDevice();

    case kRlxPciPrepareDma: {
        // Both descriptors must exist: IOMemoryDescriptor arguments only appear
        // for buffers over 4096 bytes, so a smaller request arrives as scalars
        // and would be silently misread.
        if (!arguments->structureInputDescriptor || !arguments->structureOutputDescriptor)
            return kIOReturnBadArgument;
        if (ivars->reserve(ivars->count + 1)) return kIOReturnNoMemory;

        uint64_t size = 0;
        arguments->structureInputDescriptor->GetLength(&size);

        IOAddressSegment segments[32];
        uint32_t segmentCount = 32;
        IODMACommand *dma = nullptr;
        kern_return_t err = ivars->provider->PrepareDma(arguments->structureInputDescriptor,
                                                        size, &dma, segments, &segmentCount);
        if (err) return err;

        IOMemoryMap *map = nullptr;
        err = arguments->structureOutputDescriptor->CreateMapping(0, 0, 0, 0, 0, &map);
        if (err || !map) {
            dma->release();
            return err ?: kIOReturnError;
        }
        uint64_t *out = (uint64_t *)map->GetAddress();
        for (uint32_t i = 0; i < segmentCount; i++) {
            out[i * 2] = segments[i].address;
            out[i * 2 + 1] = segments[i].length;
        }
        out[segmentCount * 2] = 0;
        out[segmentCount * 2 + 1] = 0;
        map->release();

        ivars->allocations[ivars->count++] = { nullptr, dma };
        return kIOReturnSuccess;
    }

    default:
        return kIOReturnUnsupported;
    }
}

kern_return_t IMPL(RlxPciDriverUserClient, CopyClientMemoryForType)
{
    if (!memory) return kIOReturnBadArgument;
    if (!ivars->provider.get()) return kIOReturnNotAttached;

    // Below the BAR limit the type is a BAR index; at or above it the type IS
    // the allocation size. Two meanings on one argument is the DriverKit idiom
    // for getting more than one kind of memory to a client.
    if (type < kRlxPciBarLimit) {
        return ivars->provider->MapBar((uint32_t)type, memory);
    }

    if (ivars->reserve(ivars->count + 1)) return kIOReturnNoMemory;

    RlxDmaAllocation allocation {};
    kern_return_t err = ivars->provider->CreateDma(type, &allocation);
    if (err) return err;

    ivars->allocations[ivars->count++] = allocation;
    *memory = allocation.buffer;
    return kIOReturnSuccess;
}
