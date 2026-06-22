#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static inline void row_fma_segment(double* c, const double* b, double av, int len) {
    int j = 0;
#ifdef __AVX512F__
    const __m512d avec = _mm512_set1_pd(av);
    for (; j + 31 < len; j += 32) {
        __m512d c0 = _mm512_loadu_pd(c + j);
        __m512d c1 = _mm512_loadu_pd(c + j + 8);
        __m512d c2 = _mm512_loadu_pd(c + j + 16);
        __m512d c3 = _mm512_loadu_pd(c + j + 24);
        c0 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j), c0);
        c1 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j + 8), c1);
        c2 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j + 16), c2);
        c3 = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j + 24), c3);
        _mm512_storeu_pd(c + j, c0);
        _mm512_storeu_pd(c + j + 8, c1);
        _mm512_storeu_pd(c + j + 16, c2);
        _mm512_storeu_pd(c + j + 24, c3);
    }
    for (; j + 7 < len; j += 8) {
        __m512d cv = _mm512_loadu_pd(c + j);
        cv = _mm512_fmadd_pd(avec, _mm512_loadu_pd(b + j), cv);
        _mm512_storeu_pd(c + j, cv);
    }
#endif
    for (; j < len; ++j) {
        c[j] += av * b[j];
    }
}

static void row_tiny_output_large_k(int m, int n, int k,
                                    const double* A, const double* B, double* C) {
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const int out_size = m * n;
    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

#ifdef _OPENMP
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(static)
        for (int l = 0; l < k; ++l) {
            const double* a = A + static_cast<std::size_t>(l) * m;
            const double* b = B + static_cast<std::size_t>(l) * n;
            for (int i = 0; i < m; ++i) {
                row_fma_segment(ct + static_cast<std::size_t>(i) * n, b, a[i], n);
            }
        }
    }
#else
    double* ct = tmp.data();
    for (int l = 0; l < k; ++l) {
        const double* a = A + static_cast<std::size_t>(l) * m;
        const double* b = B + static_cast<std::size_t>(l) * n;
        for (int i = 0; i < m; ++i) {
            row_fma_segment(ct + static_cast<std::size_t>(i) * n, b, a[i], n);
        }
    }
#endif

    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    for (int idx = 0; idx < out_size; ++idx) {
        double sum = 0.0;
        for (int t = 0; t < nthreads; ++t) {
            sum += tmp[static_cast<std::size_t>(t) * out_size + idx];
        }
        C[idx] = sum;
    }
}

static inline void row_kernel_8x16(int m, int n, int k,
                                   const double* A, const double* B, double* C,
                                   int i0, int j0) {
#ifdef __AVX512F__
    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
    __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
    __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
    __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma unroll(4)
#endif
    for (int l = 0; l < k; ++l) {
        const double* b = B + static_cast<std::size_t>(l) * n + j0;
        const double* a = A + static_cast<std::size_t>(l) * m + i0;
        const __m512d b0 = _mm512_loadu_pd(b);
        const __m512d b1 = _mm512_loadu_pd(b + 8);

        const __m512d a0 = _mm512_set1_pd(a[0]);
        const __m512d a1 = _mm512_set1_pd(a[1]);
        const __m512d a2 = _mm512_set1_pd(a[2]);
        const __m512d a3 = _mm512_set1_pd(a[3]);
        const __m512d a4 = _mm512_set1_pd(a[4]);
        const __m512d a5 = _mm512_set1_pd(a[5]);
        const __m512d a6 = _mm512_set1_pd(a[6]);
        const __m512d a7 = _mm512_set1_pd(a[7]);

        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        c11 = _mm512_fmadd_pd(a1, b1, c11);
        c20 = _mm512_fmadd_pd(a2, b0, c20);
        c21 = _mm512_fmadd_pd(a2, b1, c21);
        c30 = _mm512_fmadd_pd(a3, b0, c30);
        c31 = _mm512_fmadd_pd(a3, b1, c31);
        c40 = _mm512_fmadd_pd(a4, b0, c40);
        c41 = _mm512_fmadd_pd(a4, b1, c41);
        c50 = _mm512_fmadd_pd(a5, b0, c50);
        c51 = _mm512_fmadd_pd(a5, b1, c51);
        c60 = _mm512_fmadd_pd(a6, b0, c60);
        c61 = _mm512_fmadd_pd(a6, b1, c61);
        c70 = _mm512_fmadd_pd(a7, b0, c70);
        c71 = _mm512_fmadd_pd(a7, b1, c71);
    }

    double* c = C + static_cast<std::size_t>(i0) * n + j0;
    _mm512_storeu_pd(c, c00); _mm512_storeu_pd(c + 8, c01);
    c += n;
    _mm512_storeu_pd(c, c10); _mm512_storeu_pd(c + 8, c11);
    c += n;
    _mm512_storeu_pd(c, c20); _mm512_storeu_pd(c + 8, c21);
    c += n;
    _mm512_storeu_pd(c, c30); _mm512_storeu_pd(c + 8, c31);
    c += n;
    _mm512_storeu_pd(c, c40); _mm512_storeu_pd(c + 8, c41);
    c += n;
    _mm512_storeu_pd(c, c50); _mm512_storeu_pd(c + 8, c51);
    c += n;
    _mm512_storeu_pd(c, c60); _mm512_storeu_pd(c + 8, c61);
    c += n;
    _mm512_storeu_pd(c, c70); _mm512_storeu_pd(c + 8, c71);
#else
    for (int ii = 0; ii < 8; ++ii) {
        double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
        for (int jj = 0; jj < 16; ++jj) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                       B[static_cast<std::size_t>(l) * n + j0 + jj];
            }
            c[jj] = sum;
        }
    }
#endif
}

static void row_tile_i8_j16(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    constexpr int IB = 8;
    constexpr int JB = 16;
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB;
            const int j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);

            if (ilen == IB && jlen == JB) {
                row_kernel_8x16(m, n, k, A, B, C, i0, j0);
                continue;
            }

#ifdef __AVX512F__
            if (jlen >= 8) {
                __m512d acc0[IB];
                __m512d acc1[IB];
                for (int ii = 0; ii < ilen; ++ii) {
                    acc0[ii] = _mm512_setzero_pd();
                    acc1[ii] = _mm512_setzero_pd();
                }

                for (int l = 0; l < k; ++l) {
                    const double* b = B + static_cast<std::size_t>(l) * n + j0;
                    const double* a = A + static_cast<std::size_t>(l) * m + i0;
                    const __m512d b0 = _mm512_loadu_pd(b);
                    const __m512d b1 = (jlen >= 16) ? _mm512_loadu_pd(b + 8) : _mm512_setzero_pd();
                    for (int ii = 0; ii < ilen; ++ii) {
                        const __m512d av = _mm512_set1_pd(a[ii]);
                        acc0[ii] = _mm512_fmadd_pd(av, b0, acc0[ii]);
                        if (jlen >= 16) {
                            acc1[ii] = _mm512_fmadd_pd(av, b1, acc1[ii]);
                        }
                    }
                }

                for (int ii = 0; ii < ilen; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    _mm512_storeu_pd(c, acc0[ii]);
                    if (jlen >= 16) {
                        _mm512_storeu_pd(c + 8, acc1[ii]);
                    } else {
                        for (int jj = 8; jj < jlen; ++jj) {
                            double sum = 0.0;
                            for (int l = 0; l < k; ++l) {
                                sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                                       B[static_cast<std::size_t>(l) * n + j0 + jj];
                            }
                            c[jj] = sum;
                        }
                    }
                }
                continue;
            }
#endif

            for (int ii = 0; ii < ilen; ++ii) {
                double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                for (int jj = 0; jj < jlen; ++jj) {
                    double sum = 0.0;
                    for (int l = 0; l < k; ++l) {
                        sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                               B[static_cast<std::size_t>(l) * n + j0 + jj];
                    }
                    c[jj] = sum;
                }
            }
        }
    }
}

static void row_tile_i8_j16_grouped(int m, int n, int k,
                                    const double* A, const double* B, double* C) {
    constexpr int IB = 8;
    constexpr int JB = 16;
    constexpr int JGROUP = 64;
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;
    const int nbjg = (nbj + JGROUP - 1) / JGROUP;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bjg = 0; bjg < nbjg; ++bjg) {
            const int i0 = bi * IB;
            const int ilen = std::min(IB, m - i0);
            const int bj_begin = bjg * JGROUP;
            const int bj_end = std::min(nbj, bj_begin + JGROUP);

            for (int bj = bj_begin; bj < bj_end; ++bj) {
                const int j0 = bj * JB;
                const int jlen = std::min(JB, n - j0);

                if (ilen == IB && jlen == JB) {
                    row_kernel_8x16(m, n, k, A, B, C, i0, j0);
                    continue;
                }

                for (int ii = 0; ii < ilen; ++ii) {
                    double* c = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    for (int jj = 0; jj < jlen; ++jj) {
                        double sum = 0.0;
                        for (int l = 0; l < k; ++l) {
                            sum += A[static_cast<std::size_t>(l) * m + i0 + ii] *
                                   B[static_cast<std::size_t>(l) * n + j0 + jj];
                        }
                        c[jj] = sum;
                    }
                }
            }
        }
    }
}

static inline double hsum_vec(
#ifdef __AVX512F__
    __m512d v
#else
    double v
#endif
) {
#ifdef __AVX512F__
    return _mm512_reduce_add_pd(v);
#else
    return v;
#endif
}

static inline double dot_contiguous(const double* a, const double* b, int k) {
    int l = 0;
#ifdef __AVX512F__
    __m512d acc = _mm512_setzero_pd();
    for (; l + 7 < k; l += 8) {
        acc = _mm512_fmadd_pd(_mm512_loadu_pd(a + l), _mm512_loadu_pd(b + l), acc);
    }
    double sum = hsum_vec(acc);
#else
    double sum = 0.0;
#endif
    for (; l < k; ++l) {
        sum += a[l] * b[l];
    }
    return sum;
}

static inline void col_i8_kernel(int ilen, int k,
                                 const double* A, const double* b, double* c) {
#ifdef __AVX512F__
    __m512d acc[8];
    for (int ii = 0; ii < ilen; ++ii) {
        acc[ii] = _mm512_setzero_pd();
    }
    int l = 0;
    for (; l + 7 < k; l += 8) {
        const __m512d bv = _mm512_loadu_pd(b + l);
        for (int ii = 0; ii < ilen; ++ii) {
            const double* a = A + static_cast<std::size_t>(ii) * k + l;
            acc[ii] = _mm512_fmadd_pd(_mm512_loadu_pd(a), bv, acc[ii]);
        }
    }
    for (int ii = 0; ii < ilen; ++ii) {
        const double* a = A + static_cast<std::size_t>(ii) * k;
        double sum = hsum_vec(acc[ii]);
        for (int lt = l; lt < k; ++lt) {
            sum += a[lt] * b[lt];
        }
        c[ii] = sum;
    }
#else
    double sum[8] = {};
    for (int l = 0; l < k; ++l) {
        const double bv = b[l];
        for (int ii = 0; ii < ilen; ++ii) {
            sum[ii] += A[static_cast<std::size_t>(ii) * k + l] * bv;
        }
    }
    for (int ii = 0; ii < ilen; ++ii) {
        c[ii] = sum[ii];
    }
#endif
}

static void col_dot_element_parallel(int m, int n, int k,
                                     const double* A, const double* B, double* C) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[static_cast<std::size_t>(j) * m + i] =
                dot_contiguous(A + static_cast<std::size_t>(i) * k,
                               B + static_cast<std::size_t>(j) * k,
                               k);
        }
    }
}

static void col_dot_i8_block(int m, int n, int k,
                             const double* A, const double* B, double* C,
                             bool collapse_i) {
    constexpr int IB = 8;

    if (collapse_i) {
        const int nbi = (m + IB - 1) / IB;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = 0; j < n; ++j) {
            for (int bi = 0; bi < nbi; ++bi) {
                const int i0 = bi * IB;
                const int ilen = std::min(IB, m - i0);
                const double* b = B + static_cast<std::size_t>(j) * k;
                double* c = C + static_cast<std::size_t>(j) * m + i0;
                col_i8_kernel(ilen, k, A + static_cast<std::size_t>(i0) * k, b, c);
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        const double* b = B + static_cast<std::size_t>(j) * k;
        for (int i0 = 0; i0 < m; i0 += IB) {
            const int ilen = std::min(IB, m - i0);
            double* c = C + static_cast<std::size_t>(j) * m + i0;
            col_i8_kernel(ilen, k, A + static_cast<std::size_t>(i0) * k, b, c);
        }
    }
}

static inline void col_kernel_4x4(int m, int k,
                                  const double* A, const double* B, double* C,
                                  int i0, int j0) {
#ifdef __AVX512F__
    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    __m512d c02 = _mm512_setzero_pd(), c03 = _mm512_setzero_pd();
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    __m512d c12 = _mm512_setzero_pd(), c13 = _mm512_setzero_pd();
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    __m512d c22 = _mm512_setzero_pd(), c23 = _mm512_setzero_pd();
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    __m512d c32 = _mm512_setzero_pd(), c33 = _mm512_setzero_pd();

    const double* a0 = A + static_cast<std::size_t>(i0 + 0) * k;
    const double* a1 = A + static_cast<std::size_t>(i0 + 1) * k;
    const double* a2 = A + static_cast<std::size_t>(i0 + 2) * k;
    const double* a3 = A + static_cast<std::size_t>(i0 + 3) * k;
    const double* b0p = B + static_cast<std::size_t>(j0 + 0) * k;
    const double* b1p = B + static_cast<std::size_t>(j0 + 1) * k;
    const double* b2p = B + static_cast<std::size_t>(j0 + 2) * k;
    const double* b3p = B + static_cast<std::size_t>(j0 + 3) * k;

    int l = 0;
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma unroll(4)
#endif
    for (; l + 7 < k; l += 8) {
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
        if (l + 32 < k) {
            _mm_prefetch(reinterpret_cast<const char*>(a0 + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(a1 + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(a2 + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(a3 + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b0p + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b1p + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b2p + l + 32), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b3p + l + 32), _MM_HINT_T0);
        }
#endif
        const __m512d a0v = _mm512_loadu_pd(a0 + l);
        const __m512d a1v = _mm512_loadu_pd(a1 + l);
        const __m512d a2v = _mm512_loadu_pd(a2 + l);
        const __m512d a3v = _mm512_loadu_pd(a3 + l);
        const __m512d b0v = _mm512_loadu_pd(b0p + l);
        const __m512d b1v = _mm512_loadu_pd(b1p + l);
        const __m512d b2v = _mm512_loadu_pd(b2p + l);
        const __m512d b3v = _mm512_loadu_pd(b3p + l);

        c00 = _mm512_fmadd_pd(a0v, b0v, c00);
        c01 = _mm512_fmadd_pd(a0v, b1v, c01);
        c02 = _mm512_fmadd_pd(a0v, b2v, c02);
        c03 = _mm512_fmadd_pd(a0v, b3v, c03);
        c10 = _mm512_fmadd_pd(a1v, b0v, c10);
        c11 = _mm512_fmadd_pd(a1v, b1v, c11);
        c12 = _mm512_fmadd_pd(a1v, b2v, c12);
        c13 = _mm512_fmadd_pd(a1v, b3v, c13);
        c20 = _mm512_fmadd_pd(a2v, b0v, c20);
        c21 = _mm512_fmadd_pd(a2v, b1v, c21);
        c22 = _mm512_fmadd_pd(a2v, b2v, c22);
        c23 = _mm512_fmadd_pd(a2v, b3v, c23);
        c30 = _mm512_fmadd_pd(a3v, b0v, c30);
        c31 = _mm512_fmadd_pd(a3v, b1v, c31);
        c32 = _mm512_fmadd_pd(a3v, b2v, c32);
        c33 = _mm512_fmadd_pd(a3v, b3v, c33);
    }

    double s00 = hsum_vec(c00), s01 = hsum_vec(c01), s02 = hsum_vec(c02), s03 = hsum_vec(c03);
    double s10 = hsum_vec(c10), s11 = hsum_vec(c11), s12 = hsum_vec(c12), s13 = hsum_vec(c13);
    double s20 = hsum_vec(c20), s21 = hsum_vec(c21), s22 = hsum_vec(c22), s23 = hsum_vec(c23);
    double s30 = hsum_vec(c30), s31 = hsum_vec(c31), s32 = hsum_vec(c32), s33 = hsum_vec(c33);
    for (; l < k; ++l) {
        const double av0 = a0[l], av1 = a1[l], av2 = a2[l], av3 = a3[l];
        const double bv0 = b0p[l], bv1 = b1p[l], bv2 = b2p[l], bv3 = b3p[l];
        s00 += av0 * bv0; s01 += av0 * bv1; s02 += av0 * bv2; s03 += av0 * bv3;
        s10 += av1 * bv0; s11 += av1 * bv1; s12 += av1 * bv2; s13 += av1 * bv3;
        s20 += av2 * bv0; s21 += av2 * bv1; s22 += av2 * bv2; s23 += av2 * bv3;
        s30 += av3 * bv0; s31 += av3 * bv1; s32 += av3 * bv2; s33 += av3 * bv3;
    }

    double* c0p = C + static_cast<std::size_t>(j0 + 0) * m + i0;
    double* c1p = C + static_cast<std::size_t>(j0 + 1) * m + i0;
    double* c2p = C + static_cast<std::size_t>(j0 + 2) * m + i0;
    double* c3p = C + static_cast<std::size_t>(j0 + 3) * m + i0;
    c0p[0] = s00; c0p[1] = s10; c0p[2] = s20; c0p[3] = s30;
    c1p[0] = s01; c1p[1] = s11; c1p[2] = s21; c1p[3] = s31;
    c2p[0] = s02; c2p[1] = s12; c2p[2] = s22; c2p[3] = s32;
    c3p[0] = s03; c3p[1] = s13; c3p[2] = s23; c3p[3] = s33;
#else
    for (int jj = 0; jj < 4; ++jj) {
        for (int ii = 0; ii < 4; ++ii) {
            C[static_cast<std::size_t>(j0 + jj) * m + i0 + ii] =
                dot_contiguous(A + static_cast<std::size_t>(i0 + ii) * k,
                               B + static_cast<std::size_t>(j0 + jj) * k,
                               k);
        }
    }
#endif
}

static inline void col_outer_kernel_8x8(int m, int k,
                                        const double* Apack, const double* B, double* C,
                                        int i0, int j0) {
#ifdef __AVX512F__
    __m512d c0 = _mm512_setzero_pd();
    __m512d c1 = _mm512_setzero_pd();
    __m512d c2 = _mm512_setzero_pd();
    __m512d c3 = _mm512_setzero_pd();
    __m512d c4 = _mm512_setzero_pd();
    __m512d c5 = _mm512_setzero_pd();
    __m512d c6 = _mm512_setzero_pd();
    __m512d c7 = _mm512_setzero_pd();

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma unroll(4)
#endif
    for (int l = 0; l < k; ++l) {
        const __m512d av = _mm512_loadu_pd(Apack + static_cast<std::size_t>(l) * m + i0);
        const double* b = B + static_cast<std::size_t>(j0) * k + l;
        c0 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[0 * k]), c0);
        c1 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[1 * k]), c1);
        c2 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[2 * k]), c2);
        c3 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[3 * k]), c3);
        c4 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[4 * k]), c4);
        c5 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[5 * k]), c5);
        c6 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[6 * k]), c6);
        c7 = _mm512_fmadd_pd(av, _mm512_set1_pd(b[7 * k]), c7);
    }

    double* c0p = C + static_cast<std::size_t>(j0 + 0) * m + i0;
    double* c1p = C + static_cast<std::size_t>(j0 + 1) * m + i0;
    double* c2p = C + static_cast<std::size_t>(j0 + 2) * m + i0;
    double* c3p = C + static_cast<std::size_t>(j0 + 3) * m + i0;
    double* c4p = C + static_cast<std::size_t>(j0 + 4) * m + i0;
    double* c5p = C + static_cast<std::size_t>(j0 + 5) * m + i0;
    double* c6p = C + static_cast<std::size_t>(j0 + 6) * m + i0;
    double* c7p = C + static_cast<std::size_t>(j0 + 7) * m + i0;
    _mm512_storeu_pd(c0p, c0);
    _mm512_storeu_pd(c1p, c1);
    _mm512_storeu_pd(c2p, c2);
    _mm512_storeu_pd(c3p, c3);
    _mm512_storeu_pd(c4p, c4);
    _mm512_storeu_pd(c5p, c5);
    _mm512_storeu_pd(c6p, c6);
    _mm512_storeu_pd(c7p, c7);
#else
    for (int jj = 0; jj < 8; ++jj) {
        for (int ii = 0; ii < 8; ++ii) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += Apack[static_cast<std::size_t>(l) * m + i0 + ii] *
                       B[static_cast<std::size_t>(j0 + jj) * k + l];
            }
            C[static_cast<std::size_t>(j0 + jj) * m + i0 + ii] =
                sum;
        }
    }
#endif
}

static void col_outer_packed_i8_j8(int m, int n, int k,
                                   const double* A, const double* B, double* C) {
    constexpr int IB = 8;
    constexpr int JB = 8;
    std::vector<double> Apack(static_cast<std::size_t>(k) * m);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int l = 0; l < k; ++l) {
        double* dst = Apack.data() + static_cast<std::size_t>(l) * m;
        for (int i = 0; i < m; ++i) {
            dst[i] = A[static_cast<std::size_t>(i) * k + l];
        }
    }

    const int nbi = m / IB;
    const int nbj = n / JB;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int bj = 0; bj < nbj; ++bj) {
        const int j0 = bj * JB;
        for (int bi = 0; bi < nbi; ++bi) {
            col_outer_kernel_8x8(m, k, Apack.data(), B, C, bi * IB, j0);
        }
    }

    const int i_tail = nbi * IB;
    const int j_tail = nbj * JB;
    if (i_tail == m && j_tail == n) return;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (i < i_tail && j < j_tail) continue;
            C[static_cast<std::size_t>(j) * m + i] =
                dot_contiguous(A + static_cast<std::size_t>(i) * k,
                               B + static_cast<std::size_t>(j) * k,
                               k);
        }
    }
}

static void col_tile_i4_j4(int m, int n, int k,
                           const double* A, const double* B, double* C) {
    constexpr int IB = 4;
    constexpr int JB = 4;
    const int nbi = m / IB;
    const int nbj = n / JB;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int bj = 0; bj < nbj; ++bj) {
        for (int bi = 0; bi < nbi; ++bi) {
            col_kernel_4x4(m, k, A, B, C, bi * IB, bj * JB);
        }
    }

    const int i_tail = nbi * IB;
    const int j_tail = nbj * JB;
    if (i_tail == m && j_tail == n) return;

    if (i_tail < m) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = 0; j < j_tail; ++j) {
            for (int i = i_tail; i < m; ++i) {
                C[static_cast<std::size_t>(j) * m + i] =
                    dot_contiguous(A + static_cast<std::size_t>(i) * k,
                                   B + static_cast<std::size_t>(j) * k,
                                   k);
            }
        }
    }

    if (j_tail < n) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = j_tail; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                C[static_cast<std::size_t>(j) * m + i] =
                    dot_contiguous(A + static_cast<std::size_t>(i) * k,
                                   B + static_cast<std::size_t>(j) * k,
                                   k);
            }
        }
    }
}

static void tsmm_opt_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m == 4000 && n == 16000 && k == 128) {
        row_tile_i8_j16_grouped(m, n, k, A, B, C);
    } else if (m == 8 && n == 16 && k == 16000) {
        row_tiny_output_large_k(m, n, k, A, B, C);
    } else {
        row_tile_i8_j16(m, n, k, A, B, C);
    }
}

static void tsmm_opt_col(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m <= 16 && n <= 64 && k >= 1024) {
        col_dot_element_parallel(m, n, k, A, B, C);
    } else if (m == 4000 && n == 16000 && k == 128) {
        col_outer_packed_i8_j8(m, n, k, A, B, C);
    } else if (m >= 32 && n >= 1024 && k <= 256) {
        col_tile_i4_j4(m, n, k, A, B, C);
    } else {
        col_dot_i8_block(m, n, k, A, B, C, n < 512);
    }
}

void tsmm_opt(int m, int n, int k,
              const double* A, const double* B, double* C,
              Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_opt_row(m, n, k, A, B, C);
    } else {
        tsmm_opt_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("opt", tsmm_opt);
