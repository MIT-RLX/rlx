// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The extension's whole job: claim the device, enable bus mastering, and give
// userspace config space, BARs, a reset, and DMA memory with known physical
// addresses. It contains no GPU knowledge and drives nothing.

#include "RlxPciDriver.h"
#include "RlxPciDriverUserClient.h"
#include <DriverKit/IOLib.h>
#include <DriverKit/IOMemoryMap.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct RlxPciDriver_IVars {
    IOPCIDevice *pci = nullptr;
};

bool RlxPciDriver::init()
{
    if (!super::init()) return false;
    ivars = IONewZero(RlxPciDriver_IVars, 1);
    return ivars != nullptr;
}

void RlxPciDriver::free()
{
    IOSafeDeleteNULL(ivars, RlxPciDriver_IVars, 1);
    super::free();
}

kern_return_t IMPL(RlxPciDriver, Start)
{
    kern_return_t err = Start(provider, SUPERDISPATCH);
    if (err) return err;

    ivars->pci = OSDynamicCast(IOPCIDevice, provider);
    if (!ivars->pci) return kIOReturnNoDevice;

    err = ivars->pci->Open(this, 0);
    if (err) {
        os_log(OS_LOG_DEFAULT, "rlxpci: Open failed 0x%08x", err);
        ivars->pci = nullptr;
        return err;
    }

    uint16_t vendor = 0, device = 0;
    ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetVendorID, &vendor);
    ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetDeviceID, &device);
    os_log(OS_LOG_DEFAULT, "rlxpci: claimed %04x:%04x", vendor, device);

    // Bus mastering is what lets the device DMA at all; without it every ring
    // and page-table read from the card silently goes nowhere.
    uint16_t command = 0;
    ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &command);
    command |= (kIOPCICommandBusMaster | kIOPCICommandMemorySpace | kIOPCICommandIOSpace);
    ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, command);

    // The name userspace looks up. Keep it in step with RLX_EGPU_SERVICE /
    // DextConfig::service() on the rlx side.
    IOServiceName name;
    memcpy((void *)name, (void *)"rlxpci\0", 7);
    SetName(name);

    RegisterService();
    return kIOReturnSuccess;
}

kern_return_t IMPL(RlxPciDriver, Stop)
{
    if (ivars->pci) ivars->pci->Close(this, 0);
    return Stop(provider, SUPERDISPATCH);
}

kern_return_t IMPL(RlxPciDriver, NewUserClient)
{
    IOService *service = nullptr;
    kern_return_t err = Create(this, "RlxPciDriverUserClientProperties", &service);
    if (err) {
        os_log(OS_LOG_DEFAULT, "rlxpci: user client create failed 0x%08x", err);
        return err;
    }
    *userClient = OSDynamicCast(IOUserClient, service);
    return kIOReturnSuccess;
}

kern_return_t RlxPciDriver::MapBar(uint32_t bar, IOMemoryDescriptor **memory)
{
    uint8_t index = 0, type = 0;
    uint64_t size = 0;
    kern_return_t err = ivars->pci->GetBARInfo((uint8_t)bar, &index, &size, &type);
    if (err) return err;
    return ivars->pci->_CopyDeviceMemoryWithIndex(index, memory, this);
}

kern_return_t RlxPciDriver::PrepareDma(IOMemoryDescriptor *memory, uint64_t size,
                                       IODMACommand **command,
                                       IOAddressSegment *segments, uint32_t *segmentCount)
{
    // 40 address bits is what the GPU page tables and ring descriptors address;
    // asking for more would let the allocator hand back memory the device
    // cannot reach, and the failure would appear as a silent bad read.
    IODMACommandSpecification spec = { .options = 0, .maxAddressBits = 40 };
    IODMACommand *dma = nullptr;

    kern_return_t err = IODMACommand::Create(ivars->pci, kIODMACommandCreateNoOptions, &spec, &dma);
    if (err) return err;

    uint64_t flags = kIOMemoryDirectionInOut;
    err = dma->PrepareForDMA(kIODMACommandPrepareForDMANoOptions, memory, 0, size,
                             &flags, segmentCount, segments);
    if (err) {
        dma->release();
        return err;
    }
    *command = dma;
    return kIOReturnSuccess;
}

kern_return_t RlxPciDriver::CreateDma(size_t size, RlxDmaAllocation *allocation)
{
    IOBufferMemoryDescriptor *buffer = nullptr;
    kern_return_t err = IOBufferMemoryDescriptor::Create(kIOMemoryDirectionInOut, size,
                                                         IOVMPageSize, &buffer);
    if (err) return err;

    IOAddressSegment segments[32];
    uint32_t count = 32;
    IODMACommand *dma = nullptr;
    err = PrepareDma(buffer, size, &dma, segments, &count);
    if (err) {
        buffer->release();
        return err;
    }

    // Physical segments go into the head of the buffer as
    // [addr0, len0, addr1, len1, ..., 0, 0]. This is the only way userspace
    // learns physical addresses, and they are exactly what a GPU's page tables
    // and ring descriptors need.
    IOMemoryMap *map = nullptr;
    err = buffer->CreateMapping(0, 0, 0, 0, 0, &map);
    if (err || !map) {
        dma->CompleteDMA(kIODMACommandCompleteDMANoOptions);
        dma->release();
        buffer->release();
        return err ?: kIOReturnError;
    }
    uint64_t *out = (uint64_t *)map->GetAddress();
    for (uint32_t i = 0; i < count; i++) {
        out[i * 2] = segments[i].address;
        out[i * 2 + 1] = segments[i].length;
    }
    out[count * 2] = 0;
    out[count * 2 + 1] = 0;
    map->release();

    allocation->buffer = buffer;
    allocation->command = dma;
    return kIOReturnSuccess;
}

kern_return_t RlxPciDriver::ConfigRead(uint32_t offset, uint32_t size, uint32_t *value)
{
    if (!ivars->pci || !value) return kIOReturnNotReady;
    switch (size) {
        case 1: { uint8_t  v = 0; ivars->pci->ConfigurationRead8(offset, &v);  *value = v; break; }
        case 2: { uint16_t v = 0; ivars->pci->ConfigurationRead16(offset, &v); *value = v; break; }
        case 4: { uint32_t v = 0; ivars->pci->ConfigurationRead32(offset, &v); *value = v; break; }
        default: return kIOReturnBadArgument;
    }
    return kIOReturnSuccess;
}

kern_return_t RlxPciDriver::ConfigWrite(uint32_t offset, uint32_t size, uint32_t value)
{
    if (!ivars->pci) return kIOReturnNotReady;
    switch (size) {
        case 1: ivars->pci->ConfigurationWrite8(offset, (uint8_t)value); break;
        case 2: ivars->pci->ConfigurationWrite16(offset, (uint16_t)value); break;
        case 4: ivars->pci->ConfigurationWrite32(offset, value); break;
        default: return kIOReturnBadArgument;
    }
    return kIOReturnSuccess;
}

kern_return_t RlxPciDriver::ResetDevice()
{
    if (!ivars->pci) return kIOReturnNotReady;
    kern_return_t err = ivars->pci->Reset(kIOPCIDeviceResetTypeFunctionReset);
    // Not every part implements FLR; a hot reset is the fallback.
    return err == kIOReturnSuccess ? err : ivars->pci->Reset(kIOPCIDeviceResetTypeHotReset);
}
