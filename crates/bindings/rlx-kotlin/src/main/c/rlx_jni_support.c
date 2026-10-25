/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The hand-written JNI half: array and string-list conversion for the
 * generated functions, and `org.rlx.kt.NativeSupport` — reading what rlx
 * returns (strings, {ptr, len} arrays, outputs, tensor maps), the last error,
 * and `Optimizer.step`, whose parameters are updated in place.
 */

#include <string.h>

#include "rlx_jni.h"

size_t* rlx_jni_sizes(JNIEnv* env, jlongArray a) {
    if (!a) return NULL;
    jsize n = (*env)->GetArrayLength(env, a);
    size_t* out = (size_t*)malloc(sizeof(size_t) * (n > 0 ? n : 1));
    jlong* v = (*env)->GetLongArrayElements(env, a, NULL);
    for (jsize i = 0; i < n; i++) out[i] = (size_t)v[i];
    (*env)->ReleaseLongArrayElements(env, a, v, JNI_ABORT);
    return out;
}

RlxJniStrings rlx_jni_strings(JNIEnv* env, jobjectArray a) {
    RlxJniStrings s = {NULL, NULL, 0};
    if (!a) return s;
    s.n = (*env)->GetArrayLength(env, a);
    s.ptrs = (const char**)calloc(s.n > 0 ? s.n : 1, sizeof(char*));
    s.refs = (jstring*)calloc(s.n > 0 ? s.n : 1, sizeof(jstring));
    for (jsize i = 0; i < s.n; i++) {
        s.refs[i] = (jstring)(*env)->GetObjectArrayElement(env, a, i);
        s.ptrs[i] = (*env)->GetStringUTFChars(env, s.refs[i], NULL);
    }
    return s;
}

void rlx_jni_strings_release(JNIEnv* env, jobjectArray a, RlxJniStrings* s) {
    (void)a;
    for (jsize i = 0; i < s->n; i++) {
        (*env)->ReleaseStringUTFChars(env, s->refs[i], s->ptrs[i]);
        (*env)->DeleteLocalRef(env, s->refs[i]);
    }
    free(s->ptrs);
    free(s->refs);
}

RlxJniArrays rlx_jni_arrays(JNIEnv* env, jobjectArray a, int bytes) {
    RlxJniArrays s = {NULL, NULL, 0, bytes};
    if (!a) return s;
    s.n = (*env)->GetArrayLength(env, a);
    s.ptrs = (void**)calloc(s.n > 0 ? s.n : 1, sizeof(void*));
    s.refs = (jarray*)calloc(s.n > 0 ? s.n : 1, sizeof(jarray));
    for (jsize i = 0; i < s.n; i++) {
        s.refs[i] = (jarray)(*env)->GetObjectArrayElement(env, a, i);
        s.ptrs[i] = bytes ? (void*)(*env)->GetByteArrayElements(env, (jbyteArray)s.refs[i], NULL)
                          : (void*)(*env)->GetFloatArrayElements(env, (jfloatArray)s.refs[i], NULL);
    }
    return s;
}

void rlx_jni_arrays_release(JNIEnv* env, jobjectArray a, RlxJniArrays* s) {
    (void)a;
    for (jsize i = 0; i < s->n; i++) {
        if (s->bytes) (*env)->ReleaseByteArrayElements(env, (jbyteArray)s->refs[i], (jbyte*)s->ptrs[i], JNI_ABORT);
        else (*env)->ReleaseFloatArrayElements(env, (jfloatArray)s->refs[i], (jfloat*)s->ptrs[i], JNI_ABORT);
        (*env)->DeleteLocalRef(env, s->refs[i]);
    }
    free(s->ptrs);
    free(s->refs);
}

/* ── org.rlx.kt.NativeSupport ───────────────────────────────── */

JNIEXPORT jstring JNICALL Java_org_rlx_kt_NativeSupport_lastError(JNIEnv* env, jclass cls) {
    (void)cls;
    return (*env)->NewStringUTF(env, rlx_last_error());
}

JNIEXPORT jstring JNICALL Java_org_rlx_kt_NativeSupport_takeString(JNIEnv* env, jclass cls, jlong p) {
    (void)cls;
    if (!p) return (*env)->NewStringUTF(env, "");
    jstring s = (*env)->NewStringUTF(env, (const char*)(intptr_t)p);
    rlx_string_free((char*)(intptr_t)p);
    return s;
}

JNIEXPORT jfloatArray JNICALL Java_org_rlx_kt_NativeSupport_takeFloats(JNIEnv* env, jclass cls, jlong p, jlong n) {
    (void)cls;
    RlxF32s v = {(float*)(intptr_t)p, (size_t)n};
    jfloatArray out = (*env)->NewFloatArray(env, (jsize)n);
    if (n) (*env)->SetFloatArrayRegion(env, out, 0, (jsize)n, v.ptr);
    rlx_f32s_free(v);
    return out;
}

JNIEXPORT jbyteArray JNICALL Java_org_rlx_kt_NativeSupport_takeBytes(JNIEnv* env, jclass cls, jlong p, jlong n) {
    (void)cls;
    RlxBytes v = {(uint8_t*)(intptr_t)p, (size_t)n};
    jbyteArray out = (*env)->NewByteArray(env, (jsize)n);
    if (n) (*env)->SetByteArrayRegion(env, out, 0, (jsize)n, (const jbyte*)v.ptr);
    rlx_bytes_free(v);
    return out;
}

JNIEXPORT jintArray JNICALL Java_org_rlx_kt_NativeSupport_takeInts(JNIEnv* env, jclass cls, jlong p, jlong n) {
    (void)cls;
    RlxU32s v = {(uint32_t*)(intptr_t)p, (size_t)n};
    jintArray out = (*env)->NewIntArray(env, (jsize)n);
    if (n) (*env)->SetIntArrayRegion(env, out, 0, (jsize)n, (const jint*)v.ptr);
    rlx_u32s_free(v);
    return out;
}

JNIEXPORT jobjectArray JNICALL Java_org_rlx_kt_NativeSupport_takeOutputs(JNIEnv* env, jclass cls, jlong p) {
    (void)cls;
    RlxOutputs* o = (RlxOutputs*)(intptr_t)p;
    size_t n = rlx_outputs_len(o);
    jobjectArray out = (*env)->NewObjectArray(env, (jsize)n, (*env)->FindClass(env, "[F"), NULL);
    for (size_t i = 0; i < n; i++) {
        const float* d = NULL;
        size_t len = 0;
        rlx_outputs_get(o, i, &d, &len);
        jfloatArray a = (*env)->NewFloatArray(env, (jsize)len);
        if (len) (*env)->SetFloatArrayRegion(env, a, 0, (jsize)len, d);
        (*env)->SetObjectArrayElement(env, out, (jsize)i, a);
        (*env)->DeleteLocalRef(env, a);
    }
    rlx_outputs_free(o);
    return out;
}

/* Typed outputs as Object[] { byte[], String, byte[], String, … }. */
JNIEXPORT jobjectArray JNICALL Java_org_rlx_kt_NativeSupport_takeTypedOutputs(JNIEnv* env, jclass cls, jlong p) {
    (void)cls;
    RlxTypedOutputs* o = (RlxTypedOutputs*)(intptr_t)p;
    size_t n = rlx_typed_outputs_len(o);
    jobjectArray out = (*env)->NewObjectArray(env, (jsize)(2 * n), (*env)->FindClass(env, "java/lang/Object"), NULL);
    for (size_t i = 0; i < n; i++) {
        const uint8_t* d = NULL;
        size_t len = 0;
        const char* dt = NULL;
        rlx_typed_outputs_get(o, i, &d, &len, &dt);
        jbyteArray a = (*env)->NewByteArray(env, (jsize)len);
        if (len) (*env)->SetByteArrayRegion(env, a, 0, (jsize)len, (const jbyte*)d);
        (*env)->SetObjectArrayElement(env, out, (jsize)(2 * i), a);
        (*env)->SetObjectArrayElement(env, out, (jsize)(2 * i + 1), (*env)->NewStringUTF(env, dt));
    }
    rlx_typed_outputs_free(o);
    return out;
}

/* A tensor map as Object[] { String, float[], String, float[], … }. */
JNIEXPORT jobjectArray JNICALL Java_org_rlx_kt_NativeSupport_takeTensors(JNIEnv* env, jclass cls, jlong p) {
    (void)cls;
    RlxTensorMap* m = (RlxTensorMap*)(intptr_t)p;
    size_t n = rlx_tensor_map_len(m);
    jobjectArray out = (*env)->NewObjectArray(env, (jsize)(2 * n), (*env)->FindClass(env, "java/lang/Object"), NULL);
    for (size_t i = 0; i < n; i++) {
        const char* name = NULL;
        const float* d = NULL;
        size_t len = 0;
        rlx_tensor_map_get(m, i, &name, &d, &len);
        jfloatArray a = (*env)->NewFloatArray(env, (jsize)len);
        if (len) (*env)->SetFloatArrayRegion(env, a, 0, (jsize)len, d);
        (*env)->SetObjectArrayElement(env, out, (jsize)(2 * i), (*env)->NewStringUTF(env, name));
        (*env)->SetObjectArrayElement(env, out, (jsize)(2 * i + 1), a);
    }
    rlx_tensor_map_free(m);
    return out;
}

/* Optimizer.step: params (float[][]) updated in place from grads. */
JNIEXPORT jint JNICALL Java_org_rlx_kt_NativeSupport_optimizerStep(JNIEnv* env, jclass cls, jlong h,
                                                                    jobjectArray names, jobjectArray params,
                                                                    jobjectArray grads) {
    (void)cls;
    RlxJniStrings n = rlx_jni_strings(env, names);
    jsize count = n.n;
    float** ps = (float**)calloc(count > 0 ? count : 1, sizeof(float*));
    const float** gs = (const float**)calloc(count > 0 ? count : 1, sizeof(float*));
    size_t* lens = (size_t*)calloc(count > 0 ? count : 1, sizeof(size_t));
    jfloatArray* pa = (jfloatArray*)calloc(count > 0 ? count : 1, sizeof(jfloatArray));
    jfloatArray* ga = (jfloatArray*)calloc(count > 0 ? count : 1, sizeof(jfloatArray));
    for (jsize i = 0; i < count; i++) {
        pa[i] = (jfloatArray)(*env)->GetObjectArrayElement(env, params, i);
        ga[i] = (jfloatArray)(*env)->GetObjectArrayElement(env, grads, i);
        ps[i] = (*env)->GetFloatArrayElements(env, pa[i], NULL);
        gs[i] = (*env)->GetFloatArrayElements(env, ga[i], NULL);
        lens[i] = (size_t)(*env)->GetArrayLength(env, pa[i]);
    }
    int32_t rc = rlx_optimizer_step((RlxOptimizer*)(intptr_t)h, n.ptrs, ps, gs, lens, NULL, NULL, (size_t)count);
    for (jsize i = 0; i < count; i++) {
        (*env)->ReleaseFloatArrayElements(env, pa[i], ps[i], 0); /* copy back */
        (*env)->ReleaseFloatArrayElements(env, ga[i], (jfloat*)gs[i], JNI_ABORT);
    }
    rlx_jni_strings_release(env, names, &n);
    free(ps); free(gs); free(lens); free(pa); free(ga);
    return rc;
}
