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
 * File: petscw.c
 * 
 * Purpose - PETSc wrapper.
 * 
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include "mmio.h"
#include <petscpc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petsclog.h>
#include "primme.h"
#include "petscw.h"

static PetscErrorCode preallocation(Mat M,PetscInt *d_nz, PetscInt *o_nz);
static PetscErrorCode loadmtx(const char* filename, Mat *M, PetscBool *pattern);
static PetscErrorCode permutematrix(Mat Ain, Mat Bin, Mat *Aout, Mat *Bout, int **permIndices);

#undef __FUNCT__
#define __FUNCT__ "readMatrixPetsc"
int readMatrixPetsc(const char* matrixFileName, int *m, int *n, int *mLocal, int *nLocal,
                    int *numProcs, int *procID, Mat **matrix, double *fnorm_, int **perm) {

   PetscErrorCode ierr;
   PetscReal fnorm;
   PetscBool pattern;
   PetscViewer viewer;

   PetscFunctionBegin;

   *matrix = (Mat *)primme_calloc(1, sizeof(Mat), "mat");
   if (!strcmp("mtx", &matrixFileName[strlen(matrixFileName)-3])) {  
      // coordinate format storing both lower and upper triangular parts
      ierr = loadmtx(matrixFileName, *matrix, &pattern); CHKERRQ(ierr);
   }
   else if (!strcmp("petsc", &matrixFileName[strlen(matrixFileName)-5])) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, matrixFileName, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD, *matrix); CHKERRQ(ierr);
      ierr = MatSetFromOptions(**matrix); CHKERRQ(ierr);
      ierr = MatLoad(**matrix, viewer); CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
   }
   else {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Could not read matrix file.");
   }
   if (fnorm_) {
      ierr = MatNorm(**matrix, NORM_FROBENIUS, &fnorm); CHKERRQ(ierr);
      *fnorm_ = fnorm;
   }

   if (perm) {
      Mat Atemp;
      ierr = permutematrix(**matrix, NULL, &Atemp, NULL, perm);CHKERRQ(ierr);
      ierr = MatDestroy(*matrix);CHKERRQ(ierr);
      **matrix = Atemp;
   }

   ierr = MatGetSize(**matrix, m, n); CHKERRQ(ierr);
   ierr = MatGetLocalSize(**matrix, mLocal, nLocal); CHKERRQ(ierr);
   MPI_Comm_size(MPI_COMM_WORLD, numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, procID);

   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "loadmtx"
static PetscErrorCode loadmtx(const char* filename, Mat *M, PetscBool *pattern) {
   PetscErrorCode ierr;
   FILE        *f;
   MM_typecode type;
   int         m,n,nz,i,j,k;
   PetscInt    low,high,*d_nz,*o_nz;
   double      re,im;
   PetscScalar s;
   long        pos;

#if !defined(PETSC_i)
#define PETSC_i 0.0
#endif

   PetscFunctionBegin;
   
   f = fopen(filename,"r");
   if (!f) SETERRQ2(PETSC_COMM_SELF,1,"fopen '%s': %s",filename,strerror(errno));
   
   /* first read to set matrix kind and size */
   ierr = mm_read_banner(f,&type);CHKERRQ(ierr);
   if (!mm_is_valid(type) || !mm_is_sparse(type) ||
       !(mm_is_real(type) || mm_is_complex(type) || mm_is_pattern(type)))
      SETERRQ1(PETSC_COMM_SELF,1,"Matrix format '%s' not supported",mm_typecode_to_str(type)); 
#if !defined(PETSC_USE_COMPLEX)
   if (mm_is_complex(type)) SETERRQ(PETSC_COMM_SELF,1,"Complex matrix not supported in real configuration"); 
#endif
   if (pattern) *pattern = mm_is_pattern(type) ? PETSC_TRUE : PETSC_FALSE;
  
   ierr = mm_read_mtx_crd_size(f,&m,&n,&nz);CHKERRQ(ierr);
   pos = ftell(f);
   ierr = MatCreate(PETSC_COMM_WORLD,M);CHKERRQ(ierr);
   ierr = MatSetSizes(*M,PETSC_DECIDE,PETSC_DECIDE,(PetscInt)m,(PetscInt)n);CHKERRQ(ierr);
   ierr = MatSetFromOptions(*M);CHKERRQ(ierr);
   ierr = MatSetUp(*M);CHKERRQ(ierr);

   ierr = MatGetOwnershipRange(*M,&low,&high);CHKERRQ(ierr);  
   ierr = PetscMalloc(sizeof(PetscInt)*(high-low),&d_nz);CHKERRQ(ierr);
   ierr = PetscMalloc(sizeof(PetscInt)*(high-low),&o_nz);CHKERRQ(ierr);
   for (i=0; i<high-low;i++) {
      d_nz[i] = 1;
      o_nz[i] = 0;
   }
   for (k=0;k<nz;k++) {
      ierr = mm_read_mtx_crd_entry(f,&i,&j,&re,&im,type);CHKERRQ(ierr);
      i--; j--;
      if (i!=j) {
         if (i>=low && i<high) {
            if (j>=low && j<high) 
               d_nz[i-low]++;
            else
               o_nz[i-low]++;
         }
         if (j>=low && j<high && !mm_is_general(type)) {
            if (i>=low && i<high) 
               d_nz[j-low]++;
            else
               o_nz[j-low]++;        
         }
      }
   }
   ierr = preallocation(*M,d_nz,o_nz);CHKERRQ(ierr);
   ierr = PetscFree(d_nz);CHKERRQ(ierr);
   ierr = PetscFree(o_nz);CHKERRQ(ierr);
  
   /* second read to load the values */ 
   ierr = fseek(f, pos, SEEK_SET);
   if (ierr) SETERRQ1(PETSC_COMM_SELF,1,"fseek: %s",strerror(errno));
    
   re = 1.0;
   im = 0.0;
   /* Set the diagonal to zero */
   for (i=low; i<high; i++) {
      ierr = MatSetValue(*M,i,i,0.0,INSERT_VALUES);CHKERRQ(ierr);
   }
   for (k=0;k<nz;k++) {
      ierr = mm_read_mtx_crd_entry(f,&i,&j,&re,&im,type);
      i--; j--;
      if (i>=low && i<high) {
         s = re + PETSC_i * im;
         ierr = MatSetValue(*M,i,j,s,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (j>=low && j<high && i != j && !mm_is_general(type)) {
         if (mm_is_symmetric(type)) s = re + PETSC_i * im;
         else if (mm_is_hermitian(type)) s = re - PETSC_i * im;
         else if (mm_is_skew(type)) s = -re - PETSC_i * im;
         else {
            SETERRQ1(PETSC_COMM_SELF,1,"Matrix format '%s' not supported",mm_typecode_to_str(type));
         }
         ierr = MatSetValue(*M,j,i,s,INSERT_VALUES);CHKERRQ(ierr);
      }
   }
   ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   if (mm_is_symmetric(type)) { 
      ierr = MatSetOption(*M,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
   }
   if ((mm_is_symmetric(type) && mm_is_real(type)) || mm_is_hermitian(type)) { 
      ierr = MatSetOption(*M,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
   }

   ierr = fclose(f);
   if (ierr) SETERRQ1(PETSC_COMM_SELF,1,"fclose: %s",strerror(errno));

   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocation"
static PetscErrorCode preallocation(Mat M,PetscInt *d_nz, PetscInt *o_nz) {
   PetscErrorCode ierr;
   PetscBool      isaij,ismpiaij,isseqaij;
   PetscMPIInt    size;

   PetscFunctionBegin;

   ierr = PetscObjectTypeCompare((PetscObject)M,MATAIJ,&isaij);CHKERRQ(ierr);
   ierr = PetscObjectTypeCompare((PetscObject)M,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
   ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
   ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

   if ((isaij && size == 1) || isseqaij) {
      ierr = MatSeqAIJSetPreallocation(M,0,d_nz);CHKERRQ(ierr);
   } else if (isaij || ismpiaij) {
      ierr = MatMPIAIJSetPreallocation(M,0,d_nz,0,o_nz);CHKERRQ(ierr);
   } else {
      ierr = PetscInfo(M,"NOT using preallocation\n");CHKERRQ(ierr);
   }
  
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "permutematrix"
static PetscErrorCode permutematrix(Mat Ain, Mat Bin, Mat *Aout, Mat *Bout, int **permIndices)
{
   PetscErrorCode  ierr;
   MatPartitioning part;
   IS              isn, is;
   PetscInt        *nlocal;
   PetscMPIInt     size, rank;
   MPI_Comm        comm;
 
   PetscFunctionBegin;
 
   ierr = PetscObjectGetComm((PetscObject)Ain,&comm);CHKERRQ(ierr);
   ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
   ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
   ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
   ierr = MatPartitioningSetAdjacency(part,Ain);CHKERRQ(ierr);
   ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
   /* get new processor owner number of each vertex */
   ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
   /* get new global number of each old global number */
   ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
   ierr = PetscMalloc(size*sizeof(int),&nlocal);CHKERRQ(ierr);
   /* get number of new vertices for each processor */
   ierr = ISPartitioningCount(is,size,nlocal);CHKERRQ(ierr);
   ierr = ISDestroy(&is);CHKERRQ(ierr);
 
   /* get old global number of each new global number */
   ierr = ISInvertPermutation(isn,nlocal[rank],&is);CHKERRQ(ierr);
   ierr = ISDestroy(&isn);CHKERRQ(ierr);
   ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);

   /* copy permutation */
   if (permIndices) {
      const PetscInt *indices;
      PetscInt i;
      *permIndices = malloc(sizeof(int)*nlocal[rank]);
      ierr = ISGetIndices(is, &indices);CHKERRQ(ierr);
      for (i=0; i<nlocal[rank]; i++) (*permIndices)[i] = indices[i];
      ierr = ISRestoreIndices(is, &indices);CHKERRQ(ierr);
   }
 
   ierr = PetscFree(nlocal);CHKERRQ(ierr);

   ierr = ISSort(is);CHKERRQ(ierr);
 
   ierr = MatGetSubMatrix(Ain,is,is,MAT_INITIAL_MATRIX,Aout);CHKERRQ(ierr);
   if (Bin && Bout) {
      ierr = MatGetSubMatrix(Bin,is,is,MAT_INITIAL_MATRIX,Bout);CHKERRQ(ierr);
   }
   ierr = ISDestroy(&is);CHKERRQ(ierr);
 
   PetscFunctionReturn(0);
}


void PETScMatvec(void *x, void *y, int *blockSize, primme_params *primme) {
   int i;
   Mat *matrix;
   Vec xvec, yvec;
   PetscErrorCode ierr;

   assert(sizeof(PetscScalar) == sizeof(PRIMME_NUM));   
   matrix = (Mat *)primme->matrix;

   #if PETSC_VERSION_LT(3,6,0)
   ierr = MatGetVecs(*matrix, &xvec, &yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   #else
   ierr = MatCreateVecs(*matrix, &xvec, &yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   #endif
   for (i=0; i<*blockSize; i++) {
      ierr = VecPlaceArray(xvec, ((PetscScalar*)x) + primme->nLocal*i); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = VecPlaceArray(yvec, ((PetscScalar*)y) + primme->nLocal*i); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = MatMult(*matrix, xvec, yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = VecResetArray(xvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = VecResetArray(yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   }
   ierr = VecDestroy(&xvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = VecDestroy(&yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
}

void ApplyPCPrecPETSC(void *x, void *y, int *blockSize, primme_params *primme) {
   int i,bs;
   Mat *matrix;
   PC pc;
   PetscBool ispcnone;
   Vec xvec, yvec;
   PetscScalar shift;
   PetscErrorCode ierr;
   PETScPrecondStruct *s;  
   PetscEventPerfInfo imat0, imatt0, ipc0, imat1, imatt1, ipc1;
   PetscLogEvent emat, ematt, epc;

   assert(sizeof(PetscScalar) == sizeof(PRIMME_NUM));   
   matrix = (Mat *)primme->matrix;
   s = (PETScPrecondStruct *)primme->preconditioner;

   /* Build preconditioner */
   if (primme->rebuildPreconditioner) {
      Mat A;
      if (primme->numberOfShiftsForPreconditioner == 1) {
         shift = primme->ShiftsForPreconditioner[0];
      #ifdef USE_DOUBLECOMPLEX
      } else if (primme->numberOfShiftsForPreconditioner == -1) {
         shift = ((PRIMME_NUM*)primme->ShiftsForPreconditioner)[0];
      #endif
      } else {
         assert(0);
      }
      ierr = KSPGetOperators(s->ksp, &A, NULL); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = MatShift(A, s->prevShift-shift); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      s->prevShift = s->prevShift - (s->prevShift-shift);
      ierr = KSPSetOperators(s->ksp, A, A); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = KSPSetUp(s->ksp); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      primme->rebuildPreconditioner = 0;
   }

   ierr = PetscLogEventGetId("MatMult", &emat); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetId("MatMultTranspose", &ematt); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetId("PCApply", &epc); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, emat, &imat0); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, ematt, &imatt0); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, epc, &ipc0); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);

   #if PETSC_VERSION_LT(3,6,0)
   ierr = MatGetVecs(*matrix, &xvec, &yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   #else
   ierr = MatCreateVecs(*matrix, &xvec, &yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   #endif
   for (i=0,bs=blockSize?*blockSize:0; i<bs; i++) {
      ierr = VecPlaceArray(xvec, ((PetscScalar*)x) + primme->nLocal*i); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = VecPlaceArray(yvec, ((PetscScalar*)y) + primme->nLocal*i); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = KSPSolve(s->ksp, xvec, yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = VecResetArray(xvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
      ierr = VecResetArray(yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   }
   ierr = VecDestroy(&xvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = VecDestroy(&yvec); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);

   ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, emat, &imat1); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, ematt, &imatt1); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, epc, &ipc1); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = KSPGetPC(s->ksp, &pc); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pc, PCNONE, &ispcnone); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   primme->stats.numMatvecs += (imat1.count - imat0.count) + (imatt1.count - imatt0.count);
   if (!ispcnone) primme->stats.numPreconds += ipc1.count - ipc0.count;
}

void ApplyInvDavidsonDiagPrecPETSc(void *x, void *y, int *blockSize, 
                                        primme_params *primme) {
   int i, j, bs;
   double shift, d, minDenominator;
   PRIMME_NUM *xvec, *yvec;
   const int nLocal = primme->nLocal;
   const PetscScalar *diag;
   Vec vec;
   PetscErrorCode ierr;
   
   vec = *(Vec *)primme->preconditioner;
   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;
   minDenominator = 1e-14*(primme->aNorm >= 0.0L ? primme->aNorm : 1.);
   bs = blockSize ? *blockSize : 0;

   ierr = VecGetArrayRead(vec, &diag); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
   for (i=0; i<bs; i++) {
      if (primme->numberOfShiftsForPreconditioner > 0) {
         shift = primme->ShiftsForPreconditioner[min(primme->numberOfShiftsForPreconditioner-1,i)];
      #ifdef USE_DOUBLECOMPLEX
      } else if (primme->numberOfShiftsForPreconditioner < 0) {
         shift = ((PRIMME_NUM*)primme->ShiftsForPreconditioner)[min(-primme->numberOfShiftsForPreconditioner-1,i)];
      #endif
      } else {
         assert(0);
      }
      for (j=0; j<nLocal; j++) {
         d = diag[j] - shift;
         d = (PetscAbsScalar(d) > minDenominator) ? d : ((d != 0) ? d/PetscAbsScalar(d) : 1.)*minDenominator;
         yvec[nLocal*i+j] = xvec[nLocal*i+j]/d;
      }
   }
   ierr = VecRestoreArrayRead(vec, &diag); CHKERRABORT(*(MPI_Comm*)primme->commInfo, ierr);
}
