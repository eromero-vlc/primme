/*******************************************************************************
 * Copyright (c) 2018, College of William & Mary
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the College of William & Mary nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COLLEGE OF WILLIAM & MARY BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * PRIMME: https://github.com/primme/primme
 * Contact: Andreas Stathopoulos, a n d r e a s _at_ c s . w m . e d u
 *******************************************************************************
 * File: numerical_private.h
 *
 * Purpose - Contains definitions and prototypes of BLAS and LAPACK functions.
 *
 ******************************************************************************/

#ifndef NUMERICAL_PRIVATE_H
#define NUMERICAL_PRIVATE_H

#include "numerical.h"

#if !defined(PRIMME_BLASINT_SIZE) || PRIMME_BLASINT_SIZE == 32
#  include <stdint.h>
#  include <inttypes.h>
#  define PRIMME_BLASINT int32_t
#  define PRIMME_BLASINT_P PRId32
#  define PRIMME_BLASINT_MAX INT32_MAX
#elif PRIMME_BLASINT_SIZE == 0
#  include <limits.h>
#  define PRIMME_BLASINT int
#  define PRIMME_BLASINT_P "d"
#  define PRIMME_BLASINT_MAX INT_MAX
#elif PRIMME_BLASINT_SIZE == 64
#  include <stdint.h>
#  include <inttypes.h>
#  define PRIMME_BLASINT int64_t
#  define PRIMME_BLASINT_P PRId64
#  define PRIMME_BLASINT_MAX INT64_MAX
#else
#  define PRIMME_BLASINT PRIMME_BLASINT_SIZE
#  define PRIMME_BLASINT_P "d"
#  define PRIMME_BLASINT_MAX ((PRIMME_BLASINT_SIZE)INT_MAX)*INT_MAX
#endif

#if (!defined(USE_HALF) && !defined(USE_HALFCOMPLEX)) || defined(BLASLAPACK_WITH_HALF)

#ifndef PRIMME_BLAS_SUFFIX
#  define PRIMME_BLAS_SUFFIX
#endif

#define SET_LAPACK_SUFFIX(X) FORTRAN_FUNCTION(CONCAT(X,PRIMME_BLAS_SUFFIX))

#ifdef USE_DOUBLE
#  define LAPACK_FUNCTION(H,K,S,C,D,Z) SET_LAPACK_SUFFIX(D)
#elif defined(USE_DOUBLECOMPLEX)
#  define LAPACK_FUNCTION(H,K,S,C,D,Z) SET_LAPACK_SUFFIX(Z)
#elif defined(USE_FLOAT)
#  define LAPACK_FUNCTION(H,K,S,C,D,Z) SET_LAPACK_SUFFIX(S)
#elif defined(USE_FLOATCOMPLEX)
#  define LAPACK_FUNCTION(H,K,S,C,D,Z) SET_LAPACK_SUFFIX(C)
#elif defined(USE_HALF)
#  define LAPACK_FUNCTION(H,K,S,C,D,Z) SET_LAPACK_SUFFIX(H)
#elif defined(USE_HALFCOMPLEX)
#  define LAPACK_FUNCTION(H,K,S,C,D,Z) SET_LAPACK_SUFFIX(K)
#endif

#define XCOPY     LAPACK_FUNCTION(hcopy , kcopy , scopy , ccopy , dcopy , zcopy )   
#define XSWAP     LAPACK_FUNCTION(hswap , kswap , sswap , cswap , dswap , zswap )
#define XGEMM     LAPACK_FUNCTION(hgemm , kgemm , sgemm , cgemm , dgemm , zgemm )
#define XTRMM     LAPACK_FUNCTION(htrmm , ktrmm , strmm , ctrmm , dtrmm , ztrmm )
#define XTRSM     LAPACK_FUNCTION(htrsm , ktrsm , strsm , ctrsm , dtrsm , ztrsm )
#define XHEMM     LAPACK_FUNCTION(hsymm , khemm , ssymm , chemm , dsymm , zhemm )
#define XHEMV     LAPACK_FUNCTION(hsymv , khemv , ssymv , chemv , dsymv , zhemv )
#define XAXPY     LAPACK_FUNCTION(haxpy , kaxpy , saxpy , caxpy , daxpy , zaxpy )
#define XGEMV     LAPACK_FUNCTION(hgemv , kgemv , sgemv , cgemv , dgemv , zgemv )
#define XDOT      LAPACK_FUNCTION(hdot  ,       , sdot  ,       , ddot  ,       )
#define XSCAL     LAPACK_FUNCTION(hscal , kscal , sscal , cscal , dscal , zscal )
#define XLARNV    LAPACK_FUNCTION(hlarnv, klarnv, slarnv, clarnv, dlarnv, zlarnv)
#define XHEEV     LAPACK_FUNCTION(hsyev , kheev , ssyev , cheev , dsyev , zheev )
#define XHEEVX    LAPACK_FUNCTION(hsyevx, kheevx, ssyevx, cheevx, dsyevx, zheevx)
#define XHEGV     LAPACK_FUNCTION(hsygv , khegv , ssygv , chegv , dsygv , zhegv )
#define XHEGVX    LAPACK_FUNCTION(hsygvx, khegvx, ssygvx, chegvx, dsygvx, zhegvx)
#define XGESVD    LAPACK_FUNCTION(hgesvd, kgesvd, sgesvd, cgesvd, dgesvd, zgesvd)
#define XHETRF    LAPACK_FUNCTION(hsytrf, khetrf, ssytrf, chetrf, dsytrf, zhetrf)
#define XHETRS    LAPACK_FUNCTION(hsytrs, khetrs, ssytrs, chetrs, dsytrs, zhetrs)
#define XPOTRF    LAPACK_FUNCTION(hpotrf, kpotrf, spotrf, cpotrf, dpotrf, zpotrf)
#define XGETRF    LAPACK_FUNCTION(hgetrf, kgetrf, sgetrf, cgetrf, dgetrf, zgetrf)
#define XGETRS    LAPACK_FUNCTION(hgetrs, kgetrs, sgetrs, cgetrs, dgetrs, zgetrs)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define STRING const char * 

void XCOPY(PRIMME_BLASINT *n, SCALAR *x, PRIMME_BLASINT *incx, SCALAR *y, PRIMME_BLASINT *incy);
void XSWAP(PRIMME_BLASINT *n, SCALAR *x, PRIMME_BLASINT *incx, SCALAR *y, PRIMME_BLASINT *incy);
void XGEMM(STRING transa, STRING transb, PRIMME_BLASINT *m, PRIMME_BLASINT *n, PRIMME_BLASINT *k, SCALAR *alpha, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb, SCALAR *beta, SCALAR *c, PRIMME_BLASINT *ldc);
void XGEMV(STRING transa, PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *x, PRIMME_BLASINT *incx, SCALAR *beta, SCALAR *y, PRIMME_BLASINT *incy);
void XTRMM(STRING side, STRING uplo, STRING transa, STRING diag, PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb);
void XTRSM(STRING side, STRING uplo, STRING transa, STRING diag, PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb);
void XHEMM(STRING side, STRING uplo, PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb, SCALAR *beta, SCALAR *c, PRIMME_BLASINT *ldc);
void XHEMV(STRING uplo, PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *x, PRIMME_BLASINT *lncx, SCALAR *beta, SCALAR *y, PRIMME_BLASINT *lncy);
void XAXPY(PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *x, PRIMME_BLASINT *incx, SCALAR *y, PRIMME_BLASINT *incy);
#ifndef USE_COMPLEX
SCALAR XDOT(PRIMME_BLASINT *n, SCALAR *x, PRIMME_BLASINT *incx, SCALAR *y, PRIMME_BLASINT *incy);
void XHEEV(STRING jobz, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *w, SCALAR *work, PRIMME_BLASINT *ldwork, PRIMME_BLASINT *info);
void XHEEVX(STRING jobz, STRING range, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *vl, SCALAR *vu, PRIMME_BLASINT *il, PRIMME_BLASINT *iu,  SCALAR *abstol, PRIMME_BLASINT *m,  SCALAR *w, SCALAR *z, PRIMME_BLASINT *ldz, SCALAR *work, PRIMME_BLASINT *ldwork, PRIMME_BLASINT *iwork, PRIMME_BLASINT *ifail, PRIMME_BLASINT *info);
void XHEGV(PRIMME_BLASINT *itype, STRING jobz, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb, SCALAR *w, SCALAR *work, PRIMME_BLASINT *ldwork, PRIMME_BLASINT *info);
void XHEGVX(PRIMME_BLASINT *itype, STRING jobz, STRING range, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb, SCALAR *vl, SCALAR *vu, PRIMME_BLASINT *il, PRIMME_BLASINT *iu,  SCALAR *abstol, PRIMME_BLASINT *m,  SCALAR *w, SCALAR *z, PRIMME_BLASINT *ldz, SCALAR *work, PRIMME_BLASINT *ldwork, PRIMME_BLASINT *iwork, PRIMME_BLASINT *ifail, PRIMME_BLASINT *info);
void XGESVD(STRING jobu, STRING jobvt, PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *s, SCALAR *u, PRIMME_BLASINT *ldu, SCALAR *vt, PRIMME_BLASINT *ldvt, SCALAR *work, PRIMME_BLASINT *ldwork, PRIMME_BLASINT *info); 
#else
void XHEEV(STRING jobz, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, REAL *w, SCALAR *work, PRIMME_BLASINT *ldwork, REAL *rwork, PRIMME_BLASINT *info);
void XHEEVX(STRING jobz, STRING range, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, REAL *vl, REAL *vu, PRIMME_BLASINT *il, PRIMME_BLASINT *iu, REAL *abstol, PRIMME_BLASINT *m,  REAL *w, SCALAR *z, PRIMME_BLASINT *ldz, SCALAR *work, PRIMME_BLASINT *ldwork, REAL *rwork, PRIMME_BLASINT *iwork, PRIMME_BLASINT *ifail, PRIMME_BLASINT *info);
void XHEGV(PRIMME_BLASINT *itype, STRING jobz, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb, REAL *w, SCALAR *work, PRIMME_BLASINT *ldwork, REAL *rwork, PRIMME_BLASINT *info);
void XHEGVX(PRIMME_BLASINT *itype, STRING jobz, STRING range, STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, SCALAR *b, PRIMME_BLASINT *ldb, REAL *vl, REAL *vu, PRIMME_BLASINT *il, PRIMME_BLASINT *iu, REAL *abstol, PRIMME_BLASINT *m,  REAL *w, SCALAR *z, PRIMME_BLASINT *ldz, SCALAR *work, PRIMME_BLASINT *ldwork, REAL *rwork, PRIMME_BLASINT *iwork, PRIMME_BLASINT *ifail, PRIMME_BLASINT *info);
void XGESVD(STRING jobu, STRING jobvt, PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, REAL *s, SCALAR *u, PRIMME_BLASINT *ldu, SCALAR *vt, PRIMME_BLASINT *ldvt, SCALAR *work, PRIMME_BLASINT *ldwork, REAL *rwork, PRIMME_BLASINT *info);
#endif
void XSCAL(PRIMME_BLASINT *n, SCALAR *alpha, SCALAR *x, PRIMME_BLASINT *incx);
void XLARNV(PRIMME_BLASINT *idist, PRIMME_BLASINT *iseed, PRIMME_BLASINT *n, SCALAR *x);
void XHETRF(STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, PRIMME_BLASINT *ipivot, SCALAR *work, PRIMME_BLASINT *ldwork, PRIMME_BLASINT *info);
void XHETRS(STRING uplo, PRIMME_BLASINT *n, PRIMME_BLASINT *nrhs, SCALAR *a, PRIMME_BLASINT *lda, PRIMME_BLASINT *ipivot, SCALAR *b, PRIMME_BLASINT *ldb, PRIMME_BLASINT *info);
void XGESV(PRIMME_BLASINT *n, PRIMME_BLASINT *nrhs, SCALAR *a, PRIMME_BLASINT *lda, PRIMME_BLASINT *ipivot, SCALAR *b, PRIMME_BLASINT *ldb, PRIMME_BLASINT *info);
void XPOTRF(STRING uplo, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, PRIMME_BLASINT *info);
void XGETRF(PRIMME_BLASINT *m, PRIMME_BLASINT *n, SCALAR *a, PRIMME_BLASINT *lda, PRIMME_BLASINT *ipivot, PRIMME_BLASINT *info);
void XGETRS(STRING trans, PRIMME_BLASINT *n, PRIMME_BLASINT *nrhs, SCALAR *a, PRIMME_BLASINT *lda, PRIMME_BLASINT *ipivot, SCALAR *b, PRIMME_BLASINT *ldb, PRIMME_BLASINT *info);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* (!defined(USE_HALF) && !defined(USE_HALFCOMPLEX)) || defined(BLASLAPACK_WITH_HALF) */

#endif /* NUMERICAL_PRIVATE_H */
