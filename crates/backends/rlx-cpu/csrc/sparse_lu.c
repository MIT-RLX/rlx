// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// A plain-C face of Accelerate's sparse LU for `sparse_lu.rs`: the Sparse
// API is overloaded inline functions, which Rust cannot call. macOS 15.5+.
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SparseMatrixStructure s;
    SparseOpaqueSymbolicFactorization sym;
    SparseOpaqueFactorization_Double num;
    int has_num;
} rlx_splu_t;

void *rlx_splu_symbolic(int n, const long *colptr, const int *rowidx, int kind) {
    rlx_splu_t *h = calloc(1, sizeof *h);
    long nnz = colptr[n];
    long *cp = malloc(sizeof(long) * (n + 1));
    int *ri = malloc(sizeof(int) * (nnz ? nnz : 1));
    memcpy(cp, colptr, sizeof(long) * (n + 1));
    memcpy(ri, rowidx, sizeof(int) * nnz);
    SparseAttributes_t att = {0};
    att.kind = SparseOrdinary;
    h->s = (SparseMatrixStructure){.rowCount = n, .columnCount = n, .columnStarts = cp, .rowIndices = ri, .attributes = att, .blockSize = 1};
    SparseFactorization_t t = kind == 0 ? SparseFactorizationLU : kind == 1 ? SparseFactorizationLUTPP : kind == 2 ? SparseFactorizationLUSPP : SparseFactorizationLUUnpivoted;
    h->sym = SparseFactor(t, h->s);
    if (h->sym.status != SparseStatusOK) {
        free(cp); free(ri); free(h);
        return NULL;
    }
    return h;
}

int rlx_splu_numeric(void *hv, const double *vals) {
    rlx_splu_t *h = hv;
    SparseMatrix_Double A = {.structure = h->s, .data = (double *)vals};
    if (h->has_num) {
        SparseRefactor(A, &h->num);
    } else {
        h->num = SparseFactor(h->sym, A);
        h->has_num = 1;
    }
    return h->num.status;
}

int rlx_splu_solve(void *hv, double *b) {
    rlx_splu_t *h = hv;
    DenseVector_Double x = {.count = h->s.columnCount, .data = b};
    SparseSolve(h->num, x);
    return h->num.status;
}

void rlx_splu_free(void *hv) {
    rlx_splu_t *h = hv;
    if (!h) return;
    if (h->has_num) SparseCleanup(h->num);
    SparseCleanup(h->sym);
    free((void *)h->s.columnStarts);
    free((void *)h->s.rowIndices);
    free(h);
}
