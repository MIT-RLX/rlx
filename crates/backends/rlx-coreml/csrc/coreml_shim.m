// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Objective-C bridge over CoreML.framework. Compiled by build.rs (cc +
// -fobjc-arc) on Apple platforms only. Keeps the heavy CoreML object
// graph (MLModel / MLMultiArray / MLComputePlan) on the ObjC side and
// exposes a flat C ABI (see coreml_shim.h) to Rust.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <sys/sysctl.h>

#include "coreml_shim.h"

struct RlxCoremlModel {
  CFTypeRef model;       // retained MLModel*
  CFTypeRef compiledURL; // retained NSURL* of the .mlmodelc (for MLComputePlan)
  int compute_units;
};

static void rlx_set_err(char *err, int err_len, NSError *e) {
  if (!err || err_len <= 0) return;
  const char *msg = e ? e.localizedDescription.UTF8String : "unknown error";
  if (!msg) msg = "unknown error";
  strncpy(err, msg, (size_t)err_len - 1);
  err[err_len - 1] = '\0';
}

static float rlx_half_bits_to_f32(uint16_t bits) {
  __fp16 h;
  memcpy(&h, &bits, sizeof(uint16_t));
  return (float)h;
}

// Whether `a` is laid out as a dense row-major (C-contiguous) buffer.
// CoreML/ANE frequently pads the inner dimension for alignment, in which
// case the backing buffer has gaps and a flat memcpy is wrong.
static BOOL rlx_is_contiguous(MLMultiArray *a) {
  NSArray<NSNumber *> *shape = a.shape;
  NSArray<NSNumber *> *strides = a.strides;
  NSInteger expected = 1;
  for (NSInteger i = (NSInteger)shape.count - 1; i >= 0; i--) {
    if (strides[i].integerValue != expected) return NO;
    expected *= shape[i].integerValue;
  }
  return YES;
}

static MLComputeUnits rlx_compute_units(int cu) {
  switch (cu) {
    case 1: return MLComputeUnitsCPUOnly;
    case 2: return MLComputeUnitsCPUAndGPU;
    case 3: return MLComputeUnitsCPUAndNeuralEngine;
    default: return MLComputeUnitsAll;
  }
}

RlxCoremlModel *rlx_coreml_load(const char *mlpackage_path, int compute_units,
                                const char *compiled_cache_path, char *err,
                                int err_len) {
  @autoreleasepool {
    NSError *e = nil;
    NSFileManager *fm = NSFileManager.defaultManager;
    NSURL *cacheURL = (compiled_cache_path && compiled_cache_path[0])
                          ? [NSURL fileURLWithPath:@(compiled_cache_path)]
                          : nil;

    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = rlx_compute_units(compute_units);

    MLModel *model = nil;
    NSURL *compiled = nil;

    // Try the cached .mlmodelc first. A stale or version-incompatible cache must
    // NOT permanently break loading — if it fails to load (either nil+NSError or
    // a thrown NSException), drop it and recompile from the .mlpackage.
    if (cacheURL && [fm fileExistsAtPath:cacheURL.path]) {
      @try {
        model = [MLModel modelWithContentsOfURL:cacheURL
                                  configuration:config
                                          error:&e];
      } @catch (NSException *ex) {
        model = nil;
      }
      if (model && !e) {
        compiled = cacheURL;
      } else {
        model = nil;
        e = nil;
        [fm removeItemAtURL:cacheURL error:nil]; // discard the bad cache entry
      }
    }

    // Cache miss / invalid cache → compile from the .mlpackage, persist, load.
    if (!model) {
      NSURL *url = [NSURL fileURLWithPath:@(mlpackage_path)];
      NSURL *tmp = [MLModel compileModelAtURL:url error:&e];
      if (!tmp || e) {
        rlx_set_err(err, err_len, e);
        return NULL;
      }
      compiled = tmp;
      if (cacheURL) {
        // Persist the compiled model to the cache (best-effort).
        [fm removeItemAtURL:cacheURL error:nil];
        [fm createDirectoryAtURL:[cacheURL URLByDeletingLastPathComponent]
            withIntermediateDirectories:YES
                             attributes:nil
                                  error:nil];
        if ([fm copyItemAtURL:tmp toURL:cacheURL error:nil]) {
          compiled = cacheURL;
        }
      }
      model = [MLModel modelWithContentsOfURL:compiled
                                configuration:config
                                        error:&e];
      if (!model || e) {
        rlx_set_err(err, err_len, e);
        return NULL;
      }
    }

    struct RlxCoremlModel *handle = malloc(sizeof(struct RlxCoremlModel));
    handle->model = CFBridgingRetain(model);
    handle->compiledURL = CFBridgingRetain(compiled);
    handle->compute_units = compute_units;
    return handle;
  }
}

int rlx_coreml_predict(RlxCoremlModel *handle, int n_inputs,
                       const char *const *in_names, const void *const *in_data,
                       const int64_t *const *in_shapes, const int *in_ranks,
                       const int *in_dtypes, int n_outputs,
                       const char *const *out_names, float *const *out_data,
                       const int *out_len, char *err, int err_len) {
  @autoreleasepool {
    MLModel *model = (__bridge MLModel *)handle->model;
    NSError *e = nil;

    NSMutableDictionary<NSString *, MLFeatureValue *> *feats =
        [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)n_inputs];

    for (int i = 0; i < n_inputs; i++) {
      int rank = in_ranks[i];
      NSMutableArray<NSNumber *> *shape =
          [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
      NSUInteger count = 1;
      for (int d = 0; d < rank; d++) {
        int64_t s = in_shapes[i][d];
        [shape addObject:@(s)];
        count *= (NSUInteger)s;
      }
      int idt = (in_dtypes && in_dtypes[i] == 1) ? 1 : 0;
      MLMultiArrayDataType arr_type =
          idt ? MLMultiArrayDataTypeFloat16 : MLMultiArrayDataTypeFloat32;
      MLMultiArray *arr =
          [[MLMultiArray alloc] initWithShape:shape
                                     dataType:arr_type
                                        error:&e];
      if (!arr || e) {
        rlx_set_err(err, err_len, e);
        return 1;
      }
      if (idt) {
        memcpy(arr.dataPointer, in_data[i], count * sizeof(uint16_t));
      } else {
        memcpy(arr.dataPointer, in_data[i], count * sizeof(float));
      }
      NSString *name = [NSString stringWithUTF8String:in_names[i]];
      feats[name] = [MLFeatureValue featureValueWithMultiArray:arr];
    }

    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:feats
                                                          error:&e];
    if (!provider || e) {
      rlx_set_err(err, err_len, e);
      return 2;
    }

    id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                           error:&e];
    if (!result || e) {
      rlx_set_err(err, err_len, e);
      return 3;
    }

    for (int o = 0; o < n_outputs; o++) {
      NSString *name = [NSString stringWithUTF8String:out_names[o]];
      MLFeatureValue *fv = [result featureValueForName:name];
      MLMultiArray *arr = fv.multiArrayValue;
      if (!arr) {
        rlx_set_err(err, err_len,
                    [NSError errorWithDomain:@"rlx.coreml"
                                        code:4
                                    userInfo:@{
                                      NSLocalizedDescriptionKey :
                                          [NSString stringWithFormat:
                                                        @"missing output '%@'",
                                                        name]
                                    }]);
        return 4;
      }
      NSUInteger n = (NSUInteger)out_len[o];
      NSUInteger have = arr.count;
      NSUInteger copy = n < have ? n : have;
      // Copy element-by-element through NSNumber to be robust to the
      // backing dtype CoreML chose (Float16/Float32) and non-trivial
      // strides; output buffers are small relative to compute.
      if (arr.dataType == MLMultiArrayDataTypeFloat32 && rlx_is_contiguous(arr)) {
        memcpy(out_data[o], arr.dataPointer, copy * sizeof(float));
      } else if (arr.dataType == MLMultiArrayDataTypeFloat16 &&
                 rlx_is_contiguous(arr)) {
        const uint16_t *src = (const uint16_t *)arr.dataPointer;
        float *dst = out_data[o];
        for (NSUInteger k = 0; k < copy; k++) {
          dst[k] = rlx_half_bits_to_f32(src[k]);
        }
      } else {
        // Strided / non-f32 backing: index logically (row-major) so the
        // inner-dim padding CoreML adds for ANE alignment is skipped.
        float *dst = out_data[o];
        for (NSUInteger k = 0; k < copy; k++) {
          dst[k] = [arr objectAtIndexedSubscript:(NSInteger)k].floatValue;
        }
      }
    }
    return 0;
  }
}

void rlx_coreml_free(RlxCoremlModel *handle) {
  if (!handle) return;
  if (handle->model) CFBridgingRelease(handle->model);
  if (handle->compiledURL) CFBridgingRelease(handle->compiledURL);
  free(handle);
}

int rlx_coreml_compute_plan(RlxCoremlModel *handle, int *counts, char *err,
                            int err_len) {
  counts[0] = counts[1] = counts[2] = counts[3] = 0;
  if (@available(macOS 14.4, iOS 17.4, *)) {
    @autoreleasepool {
      NSURL *url = (__bridge NSURL *)handle->compiledURL;
      MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
      config.computeUnits = rlx_compute_units(handle->compute_units);

      // MLComputePlan loads asynchronously; block on a semaphore since
      // callers want a synchronous probe.
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      __block MLComputePlan *plan = nil;
      __block NSError *loadErr = nil;
      [MLComputePlan loadContentsOfURL:url
                         configuration:config
                     completionHandler:^(MLComputePlan *p, NSError *e) {
                       plan = p;
                       loadErr = e;
                       dispatch_semaphore_signal(sem);
                     }];
      dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

      if (!plan) {
        rlx_set_err(err, err_len, loadErr);
        return -1;
      }

      MLModelStructureProgram *program = plan.modelStructure.program;
      if (!program) return -1; // not an ML Program (shouldn't happen here)
      MLModelStructureProgramFunction *fn = program.functions[@"main"];
      if (!fn) return -1;

      for (MLModelStructureProgramOperation *op in fn.block.operations) {
        MLComputePlanDeviceUsage *usage =
            [plan computeDeviceUsageForMLProgramOperation:op];
        id<MLComputeDeviceProtocol> dev = usage.preferredComputeDevice;
        if ([dev isKindOfClass:[MLCPUComputeDevice class]]) {
          counts[0]++;
        } else if ([dev isKindOfClass:[MLGPUComputeDevice class]]) {
          counts[1]++;
        } else if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
          counts[2]++;
        } else {
          counts[3]++;
        }
      }
      return 0;
    }
  }
  (void)err;
  (void)err_len;
  return -1;
}

// --- introspection ---------------------------------------------------------

int rlx_coreml_ane_available(void) {
  int has = 0;
  size_t len = sizeof(has);
  if (sysctlbyname("hw.optional.neuralengine", &has, &len, NULL, 0) == 0) {
    return has ? 1 : 0;
  }
  // macOS 26 / Darwin 25 stopped reporting that sysctl on some SKUs;
  // fall back to the CPU brand string (every Apple-silicon SoC ships an
  // ANE).
  char brand[256] = {0};
  size_t blen = sizeof(brand);
  if (sysctlbyname("machdep.cpu.brand_string", brand, &blen, NULL, 0) == 0) {
    if (strstr(brand, "Apple") != NULL) return 1;
  }
  return 0;
}

void rlx_coreml_chip_brand(char *buf, int len) {
  if (!buf || len <= 0) return;
  buf[0] = '\0';
  size_t blen = (size_t)len;
  sysctlbyname("machdep.cpu.brand_string", buf, &blen, NULL, 0);
  buf[len - 1] = '\0';
}

void rlx_coreml_chip_model(char *buf, int len) {
  if (!buf || len <= 0) return;
  buf[0] = '\0';
  size_t blen = (size_t)len;
  sysctlbyname("hw.model", buf, &blen, NULL, 0);
  buf[len - 1] = '\0';
}

void rlx_coreml_os_version(char *buf, int len) {
  if (!buf || len <= 0) return;
  @autoreleasepool {
    NSOperatingSystemVersion v =
        [NSProcessInfo processInfo].operatingSystemVersion;
    snprintf(buf, (size_t)len, "%ld.%ld.%ld", (long)v.majorVersion,
             (long)v.minorVersion, (long)v.patchVersion);
  }
}
