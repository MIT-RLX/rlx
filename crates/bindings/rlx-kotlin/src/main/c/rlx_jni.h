/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The helpers the generated rlx_jni.c stands on: Java arrays and string lists
 * as the C arrays rlx-capi takes.
 */

#pragma once

#include <jni.h>
#include <stdint.h>
#include <stdlib.h>

#include "rlx.h"

/* A Java String[] as `const char* const*`. */
typedef struct {
    const char** ptrs;
    jstring* refs;
    jsize n;
} RlxJniStrings;

/* A Java float[][] / byte[][] as an array of element pointers. */
typedef struct {
    void** ptrs;
    jarray* refs;
    jsize n;
    int bytes;
} RlxJniArrays;

size_t* rlx_jni_sizes(JNIEnv* env, jlongArray a);
RlxJniStrings rlx_jni_strings(JNIEnv* env, jobjectArray a);
void rlx_jni_strings_release(JNIEnv* env, jobjectArray a, RlxJniStrings* s);
RlxJniArrays rlx_jni_arrays(JNIEnv* env, jobjectArray a, int bytes);
void rlx_jni_arrays_release(JNIEnv* env, jobjectArray a, RlxJniArrays* s);
