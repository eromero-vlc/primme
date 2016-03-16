/*******************************************************************************
 *   PRIMME PReconditioned Iterative MultiMethod Eigensolver
 *   Copyright (C) 2015 College of William & Mary,
 *   James R. McCombs, Eloy Romero Alcalde, Andreas Stathopoulos, Lingfei Wu
 *
 *   This file is part of PRIMME.
 *
 *   PRIMME is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Lesser General Public
 *   License as published by the Free Software Foundation; either
 *   version 2.1 of the License, or (at your option) any later version.
 *
 *   PRIMME is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *   Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *******************************************************************************
 * File: mat.c
 * 
 * Purpose - Wrapper implemented with routines that handle matrices in
 *           CSR format.
 * 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "native.h"
#include "shared_utils.h"

static void getDiagonal(const CSRMatrix *matrix, PRIMME_NUM *diag);

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USE_DOUBLECOMPLEX
void FORTRAN_FUNCTION(amux)(int*, double*, double*, double*, int*, int*);
void FORTRAN_FUNCTION(atmuxr)(int*, int*, double*, double*, double*, int*, int*);
void FORTRAN_FUNCTION(ilut)(int*, double*, int*, int*, int*, double*, double*, int*, int*, int*,
                            double*, double*, int*, int*, int*, int*);
void FORTRAN_FUNCTION(lusol0)(int*, double*, double*, double*, int*, int*);
#else
void FORTRAN_FUNCTION(zamux)(int*, PRIMME_NUM*, PRIMME_NUM*, PRIMME_NUM*, int*, int*);
void FORTRAN_FUNCTION(zatmuxr)(int*, int*, PRIMME_NUM*, PRIMME_NUM*, PRIMME_NUM*, int*, int*);
void FORTRAN_FUNCTION(zilut)(int*, PRIMME_NUM*, int*, int*, int*, double*, PRIMME_NUM*, int*, int*, int*,
                             PRIMME_NUM*, int*, int*);
void FORTRAN_FUNCTION(zlusol)(int*, PRIMME_NUM*, PRIMME_NUM*, PRIMME_NUM*, int*, int*);
#endif

#ifdef __cplusplus
}
#endif

/******************************************************************************
 * Applies the matrix vector multiplication on a block of vectors.
 * Because a block function is not available, we call blockSize times
 * the SPARSKIT function amux(). Note the (void *) parameters x, y that must 
 * be cast as doubles for use in amux()
 *
******************************************************************************/
void CSRMatrixMatvec(void *x, void *y, int *blockSize, primme_params *primme) {
   
   int i;
   PRIMME_NUM *xvec, *yvec;
   CSRMatrix *matrix;
   
   if (!x || !y || !blockSize) return;
   matrix = (CSRMatrix *)primme->matrix;
   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;

   for (i=0;i<*blockSize;i++) {
#ifndef USE_DOUBLECOMPLEX
      FORTRAN_FUNCTION(amux)
#else
      FORTRAN_FUNCTION(zamux)
#endif
            (&primme->n, &xvec[primme->nLocal*i], &yvec[primme->nLocal*i], 
             matrix->AElts, matrix->JA, matrix->IA);
   }
}

void CSRMatrixMatvecSVD(void *x, int *ldx, void *y, int *ldy, int *blockSize,
                        int *trans, primme_svds_params *primme_svds) {
   
   int i;
   PRIMME_NUM *xvec, *yvec;
   CSRMatrix *matrix;
   
   matrix = (CSRMatrix *)primme_svds->matrix;
   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;

   if (*trans == 0) {
      for (i=0;i<*blockSize;i++) {
#ifndef USE_DOUBLECOMPLEX
         FORTRAN_FUNCTION(amux)
#else
         FORTRAN_FUNCTION(zamux)
#endif
              (&primme_svds->m, &xvec[(*ldx)*i], &yvec[(*ldy)*i], 
               matrix->AElts, matrix->JA, matrix->IA);
      }
   } else {
      for (i=0;i<*blockSize;i++) {
#ifndef USE_DOUBLECOMPLEX
         FORTRAN_FUNCTION(atmuxr)
#else
         FORTRAN_FUNCTION(zatmuxr)
#endif
           (&primme_svds->n, &primme_svds->m, &xvec[(*ldx)*i], 
            &yvec[(*ldy)*i], matrix->AElts, matrix->JA, matrix->IA);
      }
   }
}


/******************************************************************************
 * Applies the (already inverted) diagonal preconditioner
 *
 *    y(i) = P*x(i), i=1:blockSize, 
 *    with P = (Diag(A)-shift)^(-1)
 *
******************************************************************************/

static int createInvDiagPrecNative(const CSRMatrix *matrix, PRIMME_NUM shift, PRIMME_NUM **prec) {
   int i;
   double *diag;

   diag = (PRIMME_NUM*)primme_calloc(matrix->n, sizeof(PRIMME_NUM), "diag");
   getDiagonal(matrix, diag);
   for (i=0; i<matrix->n; i++)
      diag[i] -= shift;
   *prec = diag;
   return 1;
}

static void ApplyInvDiagPrecNativeGen(PRIMME_NUM *xvec, int ldx, PRIMME_NUM *yvec,
      int ldy, int nLocal, int bs, double *diag, double *shifts, double aNorm) {
   int i, j;
   const double minDenominator = 1e-14*(aNorm >= 0.0L ? aNorm : 1.);

   for (i=0; i<bs; i++) {
      double shift = shifts ? shifts[i] : 0.0;
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for (j=0; j<nLocal; j++) {
         double d = diag[j] - shift;
         d = (fabs(d) > minDenominator) ? d : copysign(minDenominator, d);
         yvec[ldy*i+j] = xvec[ldx*i+j]/d;
      }
   }
}


void ApplyInvDiagPrecNative(void *x, void *y, int *blockSize, primme_params *primme) {
   ApplyInvDiagPrecNativeGen((PRIMME_NUM*)x, primme->nLocal, (PRIMME_NUM*)y, primme->nLocal,
      primme->nLocal, *blockSize, (double*)primme->preconditioner, NULL, primme->aNorm);
}

/******************************************************************************
 * Applies a Davidson type preconditioner
 *
 *    x(i) = (Diag(A) - primme.Shifts(i) I)^(-1) * y(i),   i=1:blockSize
 *    
 * NOTE that each block vector may have its own shift provided by dprimme
 * in the array primme->ShiftsForPreconditioner
 *
 * To avoid division with too small numbers we limit how small relatively 
 * to ||A|| the denominators can be. In the absense of ||A|| we use 1e-14.
 *
******************************************************************************/

void ApplyInvDavidsonDiagPrecNative(void *x, void *y, int *blockSize, 
                                        primme_params *primme) {
   ApplyInvDiagPrecNativeGen((PRIMME_NUM*)x, primme->nLocal, (PRIMME_NUM*)y, primme->nLocal,
      primme->nLocal, *blockSize, (double*)primme->preconditioner,
      primme->ShiftsForPreconditioner, primme->aNorm);
}

/******************************************************************************
 * Applies the ILUT preconditioner 
 *
 *    y(i) = U^(-1)*( L^(-1)*x(i)), i=1:blockSize, 
 *    with L,U = ilut(A-shift) 
 * 
 * It calls the SPARSKIT lusol0 function for each block vector.
 *
******************************************************************************/

static int createILUTPrecNative(const CSRMatrix *matrix, PRIMME_NUM shift, int level,
                                double threshold, double filter, CSRMatrix **prec) {
#ifdef USE_DOUBLECOMPLEX
   int ierr;
   int lenFactors;
   PRIMME_NUM *W;
   int *iW;
   CSRMatrix *factors;

   if (shift != 0.0) {
      shiftCSRMatrix(-shift, (CSRMatrix*)matrix);
   }

   /* Work arrays */
   W = (PRIMME_NUM *)primme_calloc(matrix->n+1, sizeof(PRIMME_NUM), "W");
   iW = (int *)primme_calloc(matrix->n*2, sizeof(int), "iW");

   /* Max size of factorization */
   lenFactors = 9*matrix->nnz;

   factors = (CSRMatrix *)primme_calloc(1,  sizeof(CSRMatrix), "factors");
   factors->AElts = (PRIMME_NUM *)primme_calloc(lenFactors,
                                sizeof(PRIMME_NUM), "iluElts");
   factors->JA = (int *)primme_calloc(lenFactors, sizeof(int), "Jilu");
   factors->IA = (int *)primme_calloc(matrix->n+1, sizeof(int), "Iilu");
   factors->n = matrix->n;
   factors->nnz = lenFactors;
   
   FORTRAN_FUNCTION(zilut)
         ((int*)&matrix->n, (PRIMME_NUM*)matrix->AElts, (int*)matrix->JA,
          (int*)matrix->IA, &level, &threshold,
          factors->AElts, factors->JA, factors->IA, &lenFactors, W, iW, &ierr);
   
   if (ierr != 0)  {
      fprintf(stderr, "ZILUT factorization could not be completed\n");
      return(-1);
   }

   if (shift != 0.0L) {
      shiftCSRMatrix(shift, (CSRMatrix*)matrix);
   }

   /* free workspace */
   free(W); free(iW);

   *prec = factors;
   return 0;
#else
   int ierr;
   int lenFactors;
   double *W1, *W2;
   int *iW1, *iW2, *iW3;
   CSRMatrix *factors;

   if (shift != 0.0) {
      shiftCSRMatrix(-shift, (CSRMatrix*)matrix);
   }

   /* Work arrays */
   W1 = (double *)primme_calloc( matrix->n+1,  sizeof(double), "W1");
   W2 = (double *)primme_calloc( matrix->n,  sizeof(double), "W2");
   iW1 = (int *)primme_calloc( matrix->n,  sizeof(int), "iW1");
   iW2 = (int *)primme_calloc( matrix->n,  sizeof(int), "iW2");
   iW3 = (int *)primme_calloc( matrix->n,  sizeof(int), "iW2");
   /* Max size of factorization */
   lenFactors = 9*matrix->nnz;
   factors = (CSRMatrix *)primme_calloc(1,  sizeof(CSRMatrix), "factors");
   factors->AElts = (double *)primme_calloc(lenFactors,
                                sizeof(double), "iluElts");
   factors->JA = (int *)primme_calloc(lenFactors, sizeof(int), "Jilu");
   factors->IA = (int *)primme_calloc(matrix->n+1, sizeof(int), "Iilu");
   factors->n = matrix->n;
   factors->nnz = lenFactors;
   
   FORTRAN_FUNCTION(ilut)
        ((int*)&matrix->n, (double*)matrix->AElts, (int*)matrix->JA,
         (int*)matrix->IA, &level, &threshold,
         factors->AElts, factors->JA, factors->IA, &lenFactors, 
         W1, W2, iW1, iW2, iW3, &ierr);
   
   if (ierr != 0)  {
      fprintf(stderr, "ILUT factorization could not be completed\n");
      return(-1);
   }

   if (shift != 0.0L) {
      shiftCSRMatrix(shift, (CSRMatrix*)matrix);
   }

   /* free workspace */
   free(W1); free(W2); free(iW1); free(iW2); free(iW3);

   *prec = factors;
   return 0;
#endif
}

void ApplyILUTPrecNative(void *x, void *y, int *blockSize, primme_params *primme) {
   int i, bs;
   PRIMME_NUM *xvec, *yvec, shift;
   CSRMatrix *prec;
   driver_params *driver = (driver_params*)primme;
   
   /* Build preconditioner */
   if (primme->rebuildPreconditioner) {
      if (primme->preconditioner) freeCSRMatrix((CSRMatrix*)primme->preconditioner);
      if (primme->numberOfShiftsForPreconditioner == 1) {
         shift = primme->ShiftsForPreconditioner[0];
      #ifdef USE_DOUBLECOMPLEX
      } else if (primme->numberOfShiftsForPreconditioner == -1) {
         shift = ((PRIMME_NUM*)primme->ShiftsForPreconditioner)[0];
      #endif
      } else {
         assert(0);
      }
      createILUTPrecNative((CSRMatrix*)primme->matrix, shift, driver->level, driver->threshold,
                           driver->filter, (CSRMatrix**)&primme->preconditioner);
      primme->rebuildPreconditioner = 0;
   }

   prec = (CSRMatrix *)primme->preconditioner;
   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;
   bs = blockSize ? *blockSize : 0;

   for (i=0; i<bs; i++) {
#ifdef USE_DOUBLECOMPLEX
      FORTRAN_FUNCTION(zlusol)
#else
      FORTRAN_FUNCTION(lusol0)
#endif
             (&primme->n, &xvec[primme->n*i], &yvec[primme->n*i],
              prec->AElts, prec->JA, prec->IA);
   }
}


/******************************************************************************
 * Generates the diagonal of A.
 *
 *    P = Diag(A)
 *
 * This will be used with solver provided shifts as (P-shift_i)^(-1) 
******************************************************************************/
static void getDiagonal(const CSRMatrix *matrix, PRIMME_NUM *diag) {
   int i, j;

   /* IA and JA are indexed using C indexing, but their contents */
   /* assume Fortran indexing.  Thus, the contents of IA and JA  */
   /* must be decremented before being used in C.                */

   for (i=0; i < matrix->n; i++) {
      diag[i] = 0.;
      for (j=matrix->IA[i]; j <= matrix->IA[i+1]-1; j++) {
         if (matrix->JA[j-1]-1 == i) {
            diag[i] = REAL_PART(matrix->AElts[j-1]);
         }
      }
   }
}

/******************************************************************************
 * Generates sum of square values per rows and then per columns 
 *
******************************************************************************/
static void getSumSquares(const CSRMatrix *matrix, double *diag) {
   int i, j;
   double *sumr = diag, *sumc = &diag[matrix->m], v;

   for (i=0; i < matrix->m + matrix->n; i++) {
      diag[i] = 0.0;
   }

   /* IA and JA are indexed using C indexing, but their contents */
   /* assume Fortran indexing.  Thus, the contents of IA and JA  */
   /* must be decremented before being used in C.                */

   for (i=0; i < matrix->m; i++) {
      for (j=matrix->IA[i]; j <= matrix->IA[i+1]-1; j++) {
         v = REAL_PART(matrix->AElts[j-1]*CONJ(matrix->AElts[j-1]));
         sumr[i]   += v;
         sumc[matrix->JA[j-1]-1] += v;
      }
   }
}


/******************************************************************************
 * Applies the diagonal preconditioner over A'A or AA'
 *
******************************************************************************/

int createInvNormalPrecNative(const CSRMatrix *matrix, double shift, double **prec) {
   int i;
   double *diag, minDenominator=1e-14;

   diag = (double*)primme_calloc(matrix->m+matrix->n, sizeof(double), "diag");
   getSumSquares(matrix, diag);
   for (i=0; i<matrix->m+matrix->n; i++) {
      diag[i] -= shift*shift;
      if (fabs(diag[i]) < minDenominator)
         diag[i] = copysign(minDenominator, diag[i]);
   }
   *prec = diag;
   return 1;
}

static void ApplyInvNormalPrecNativeSvdsGen(void *x, int *ldx, void *y, int *ldy, int *blockSize,
                                 int *mode, primme_svds_params *primme_svds, double *shifts) {
   double *diag = (double *)primme_svds->preconditioner;
   PRIMME_NUM *xvec = (PRIMME_NUM *)x, *yvec = (PRIMME_NUM *)y;
   const int bs = *blockSize;
   double *sumr = diag, *sumc = &diag[primme_svds->mLocal];
   
   if (*mode == primme_svds_op_AtA) {
      ApplyInvDiagPrecNativeGen(xvec, *ldx, yvec, *ldy, primme_svds->nLocal, bs,
         sumc, shifts, primme_svds->aNorm);
   }
   else if (*mode == primme_svds_op_AAt) {
      ApplyInvDiagPrecNativeGen(xvec, *ldx, yvec, *ldy, primme_svds->mLocal, bs,
         sumr, shifts, primme_svds->aNorm);
   }
   else if (*mode == primme_svds_op_augmented) {
      ApplyInvDiagPrecNativeGen(xvec, *ldx, yvec, *ldy, primme_svds->nLocal, bs,
         sumc, shifts, primme_svds->aNorm);
      ApplyInvDiagPrecNativeGen(xvec+primme_svds->nLocal, *ldx,
         yvec+primme_svds->nLocal, *ldy, primme_svds->mLocal, bs,
         sumr, shifts, primme_svds->aNorm);
   } 
}

void ApplyInvNormalPrecNative(void *x, int *ldx, void *y, int *ldy, int *blockSize,
                              int *mode, primme_svds_params *primme_svds) {
   ApplyInvNormalPrecNativeSvdsGen(x, ldx, y, ldy, blockSize, mode, primme_svds, NULL);
}

void ApplyInvDavidsonNormalPrecNative(void *x, int *ldx, void *y, int *ldy, int *blockSize,
                                   int *mode, primme_svds_params *primme_svds) {
   primme_params *primme =
      primme_svds->method == *mode ? &primme_svds->primme : &primme_svds->primmeStage2;
   ApplyInvNormalPrecNativeSvdsGen(x, ldx, y, ldy, blockSize, mode, primme_svds,
      primme->ShiftsForPreconditioner);
}

