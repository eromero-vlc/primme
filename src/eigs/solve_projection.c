/*******************************************************************************
 * Copyright (c) 2017, College of William & Mary
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
 * File: solve_H.c
 * 
 * Purpose - Solves the eigenproblem for the matrix V'*A*V.
 *
 ******************************************************************************/

#include <math.h>
#include <assert.h>
#include "const.h"
#include "numerical.h"
#include "solve_projection.h"
#include "ortho.h"
#include "globalsum.h"

static int solve_H_RR_Sprimme(SCALAR *H, int ldH, SCALAR *VtBV, int ldVtBV,
      SCALAR *hVecs, int ldhVecs, REAL *hVals, int basisSize, int numConverged,
      size_t *lrwork, SCALAR *rwork, int liwork, int *iwork,
      primme_params *primme);

static int solve_H_Harm_Sprimme(SCALAR *H, int ldH, SCALAR *QtV, int ldQtV,
   SCALAR *R, int ldR, SCALAR *hVecs, int ldhVecs, SCALAR *hU, int ldhU,
   REAL *hVals, int basisSize, int numConverged, double machEps,
   size_t *lrwork, SCALAR *rwork, int liwork, int *iwork,
   primme_params *primme);

static int solve_H_Ref_Sprimme(SCALAR *H, int ldH, SCALAR *QtBV, int ldQtBV,
      SCALAR *hVecs, int ldhVecs, SCALAR *hVecsRot, int ldhVecsRot, SCALAR *hU,
      int ldhU, REAL *hSVals, SCALAR *R, int ldR, REAL *hVals, int basisSize,
      int targetShiftIndex, size_t *lrwork, SCALAR *rwork, int liwork,
      int *iwork, primme_params *primme);

static int solve_H_brcast_Sprimme(int basisSize, SCALAR *hU, int ldhU,
      SCALAR *hVecs, int ldhVecs, SCALAR *hVecsRot, int ldhVecsRot, REAL *hVals,
      REAL *hSVals, size_t *lrwork, SCALAR *rwork, primme_params *primme);



/*******************************************************************************
 * Subroutine solve_H - This procedure solves the project problem and return
 *       the projected vectors (hVecs) and values (hVals) in the order according
 *       to primme.target.
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H              The matrix V'*A*V
 * basisSize      The dimension of H, R, QtV and hU
 * ldH            The leading dimension of H
 * VtBV           The matrix V'*B*V
 * ldVtBV         The leading dimension of VtBV
 * R              The factor R for the QR decomposition of (A - target*I)*V
 * ldR            The leading dimension of R
 * QtV            Q'*V
 * ldQtV          The leading dimension of QtV
 * numConverged   Number of eigenvalues converged to determine ordering shift
 * lrwork         Length of the work array rwork
 * primme         Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hU             The left singular vectors of R or the eigenvectors of QtV/R
 * ldhU           The leading dimension of hU
 * hVecs          The coefficient vectors such as V*hVecs will be the Ritz vectors
 * ldhVecs        The leading dimension of hVecs
 * hVals          The Ritz values
 * hSVals         The singular values of R
 * rwork          Workspace
 * iwork          Workspace in integers
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 Num_dsyev/zheev was unsuccessful
 ******************************************************************************/

TEMPLATE_PLEASE
int solve_H_Sprimme(SCALAR *H, int basisSize, int ldH, SCALAR *VtBV, int ldVtBV,
      SCALAR *R, int ldR, SCALAR *QtV, int ldQtV, SCALAR *hU, int ldhU,
      SCALAR *hVecs, int ldhVecs, SCALAR *hVecsRot, int ldhVecsRot, REAL *hVals,
      REAL *hSVals, int numConverged, double machEps, size_t *lrwork,
      SCALAR *rwork, int liwork, int *iwork, primme_params *primme) {

   int i;

   /* In parallel (especially with heterogeneous processors/libraries) ensure */
   /* that every process has the same hVecs and hU. Only processor 0 solves   */
   /* the projected problem and broadcasts the resulting matrices to the rest */

   if (primme->procID == 0) {
      switch (primme->projectionParams.projection) {
         case primme_proj_RR:
            CHKERR(solve_H_RR_Sprimme(H, ldH, VtBV, ldVtBV, hVecs, ldhVecs,
                     hVals, basisSize, numConverged, lrwork, rwork, liwork,
                     iwork, primme), -1);
            break;

         // case primme_proj_harmonic:
         //    CHKERR(solve_H_Harm_Sprimme(H, ldH, QtV, ldQtV, R, ldR, hVecs,
         //             ldhVecs, hU, ldhU, hVals, basisSize, numConverged, machEps,
         //             lrwork, rwork, liwork, iwork, primme), -1);
         //    break;

         case primme_proj_refined:
         case primme_proj_harmonic:
         case primme_proj_refined_RR:
         case primme_proj_refined_harmonic:
            CHKERR(solve_H_Ref_Sprimme(H, ldH, QtV, ldQtV, hVecs, ldhVecs,
                         hVecsRot, ldhVecsRot, hU, ldhU, hSVals, R, ldR, hVals,
                         basisSize, numConverged, lrwork, rwork, liwork, iwork,
                         primme), -1);
            break;

         default:
            assert(0);
      }
   }

   /* Broadcast hVecs, hU, hVals, hSVals */

   CHKERR(solve_H_brcast_Sprimme(basisSize, hU, ldhU, hVecs, ldhVecs, hVecsRot,
                ldhVecsRot, hVals, hSVals, lrwork, rwork, primme), -1);

   /* Return memory requirements */

   if (H == NULL) {
      return 0;
   }

   /* -------------------------------------------------------- */
   /* Update the leftmost and rightmost Ritz values ever seen  */
   /* -------------------------------------------------------- */
   for (i=0; i<basisSize; i++) {
      primme->stats.estimateMinEVal = min(primme->stats.estimateMinEVal,
            hVals[i]); 
      primme->stats.estimateMaxEVal = max(primme->stats.estimateMaxEVal,
            hVals[i]); 
   }
   primme->stats.estimateLargestSVal = max(fabs(primme->stats.estimateMinEVal),
                                           fabs(primme->stats.estimateMaxEVal));

   return 0;
}


/*******************************************************************************
 * Subroutine solve_H_RR - This procedure solves the eigenproblem for the
 *            matrix H.
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H              The matrix V'*A*V
 * basisSize      The dimension of H, R, hU
 * ldH            The leading dimension of H
 * VtBV           The matrix V'*B*V
 * ldVtBV         The leading dimension of VtBV
 * numConverged   Number of eigenvalues converged to determine ordering shift
 * lrwork         Length of the work array rwork
 * primme         Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hVecs          The eigenvectors of H or the right singular vectors
 * ldhVecs        The leading dimension of hVecs
 * hVals          The Ritz values
 * hSVals         The singular values of R
 * rwork          Workspace
 * iwork          Workspace in integers
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 Num_dsyev/zheev was unsuccessful
 ******************************************************************************/

static int solve_H_RR_Sprimme(SCALAR *H, int ldH, SCALAR *VtBV, int ldVtBV,
      SCALAR *hVecs, int ldhVecs, REAL *hVals, int basisSize, int numConverged,
      size_t *lrwork, SCALAR *rwork, int liwork, int *iwork,
      primme_params *primme) {

   int i, j; /* Loop variables    */
   int info; /* dsyev error value */
   int index;
   int *permu, *permw;
   double targetShift;

   /* Some LAPACK implementations don't like zero-size matrices */
   if (basisSize == 0) return 0;

   /* Return memory requirements */
   if (H == NULL) {
      SCALAR rwork0;
      CHKERR((Num_hegv_Sprimme("V", "U", basisSize, hVecs, basisSize, VtBV,
                  basisSize, hVals, &rwork0, -1, &info), info), -1);
      *lrwork = max(*lrwork, (size_t)REAL_PART(rwork0));
      *iwork = max(*iwork, 2*basisSize);
      return 0;
   }

   /* ---------------------- */
   /* Divide the iwork space */
   /* ---------------------- */
   assert(liwork >= 2*basisSize);
   permu = iwork;
   permw = permu + basisSize;


   /* ------------------------------------------------------------------- */
   /* Copy the upper triangular portion of H into hvecs.  We need to do   */
   /* this since DSYEV overwrites the input matrix with the eigenvectors. */  
   /* Note that H is maxBasisSize-by-maxBasisSize and the basisSize-by-   */
   /* basisSize submatrix of H is copied into hvecs.                      */
   /* ------------------------------------------------------------------- */

   assert(H != hVecs || ldH == ldhVecs);
   if (primme->target != primme_largest) {
      for (j=0; j < basisSize; j++) {
         for (i=0; i <= j; i++) { 
            hVecs[ldhVecs*j+i] = H[ldH*j+i];
         }
      }      
   }
   else { /* (primme->target == primme_largest) */
      for (j=0; j < basisSize; j++) {
         for (i=0; i <= j; i++) { 
            hVecs[ldhVecs*j+i] = -H[ldH*j+i];
         }
      }
   }

   CHKERR((Num_hegv_Sprimme("V", "U", basisSize, hVecs, ldhVecs, VtBV, ldVtBV,
               hVals, rwork, TO_INT(*lrwork), &info), info), -1);

   /* ---------------------------------------------------------------------- */
   /* ORDER the eigenvalues and their eigenvectors according to the desired  */
   /* target:  smallest/Largest or interior closest abs/leq/geq to a shift   */
   /* ---------------------------------------------------------------------- */

   if (primme->target == primme_smallest) 
      return 0;

   if (primme->target == primme_largest) {
      for (i = 0; i < basisSize; i++) {
         hVals[i] = -hVals[i];
      }
   }
   else { 
      /* ---------------------------------------------------------------- */
      /* Select the interior shift. Use the first unlocked shift, and not */
      /* higher ones, even if some eigenpairs in the basis are converged. */
      /* Then order the ritz values based on the closeness to the shift   */
      /* from the left, from right, or in absolute value terms            */
      /* ---------------------------------------------------------------- */

      /* TODO: order properly when numTargetShifts > 1 */

      targetShift = 
        primme->targetShifts[min(primme->numTargetShifts-1, numConverged)];

      if (primme->target == primme_closest_geq) {
   
         /* ---------------------------------------------------------------- */
         /* find hVal closest to the right of targetShift, i.e., closest_geq */
         /* ---------------------------------------------------------------- */
         for (j=0;j<basisSize;j++) 
              if (hVals[j]>=targetShift) break;
           
         /* figure out this ordering */
         index = 0;
   
         for (i=j; i<basisSize; i++) {
            permu[index++]=i;
         }
         for (i=0; i<j; i++) {
            permu[index++]=i;
         }
      }
      else if (primme->target == primme_closest_leq) {
         /* ---------------------------------------------------------------- */
         /* find hVal closest_leq to targetShift                             */
         /* ---------------------------------------------------------------- */
         for (j=basisSize-1; j>=0 ;j--) 
             if (hVals[j]<=targetShift) break;
           
         /* figure out this ordering */
         index = 0;
   
         for (i=j; i>=0; i--) {
            permu[index++]=i;
         }
         for (i=basisSize-1; i>j; i--) {
            permu[index++]=i;
         }
      }
      else if (primme->target == primme_closest_abs) {

         /* ---------------------------------------------------------------- */
         /* find hVal closest but geq than targetShift                       */
         /* ---------------------------------------------------------------- */
         for (j=0;j<basisSize;j++) 
             if (hVals[j]>=targetShift) break;

         i = j-1;
         index = 0;
         while (i>=0 && j<basisSize) {
            if (fabs(hVals[i]-targetShift) < fabs(hVals[j]-targetShift)) 
               permu[index++] = i--;
            else 
               permu[index++] = j++;
         }
         if (i<0) {
            for (i=j;i<basisSize;i++) 
                    permu[index++] = i;
         }
         else if (j>=basisSize) {
            for (j=i;j>=0;j--)
                    permu[index++] = j;
         }
      }
      else if (primme->target == primme_largest_abs) {

         j = 0;
         i = basisSize-1;
         index = 0;
         while (i>=j) {
            if (fabs(hVals[i]-targetShift) > fabs(hVals[j]-targetShift)) 
               permu[index++] = i--;
            else 
               permu[index++] = j++;
         }

      }

      /* ---------------------------------------------------------------- */
      /* Reorder hVals and hVecs according to the permutation             */
      /* ---------------------------------------------------------------- */
      permute_vecs_Rprimme(hVals, 1, basisSize, 1, permu, (REAL*)rwork, permw);
      permute_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, permu, rwork,
            permw);
   }

   return 0;   
}

/*******************************************************************************
 * Subroutine solve_H_Harm - This procedure implements the harmonic extraction
 *    in a novelty way. In standard harmonic the next eigenproblem is solved:
 *       V'*(A-s*I)'*(A-s*I)*V*X = V'*(A-s*I)'*V*X*L,
 *    where (L_{i,i},X_i) are the harmonic-Ritz pairs. In practice, it is
 *    computed (A-s*I)*V = Q*R and it is solved instead:
 *       R*X = Q'*V*X*L,
 *    which is a generalized non-Hermitian problem. Instead of dealing with
 *    complex solutions, which are unnatural in context of Hermitian problems,
 *    we propose the following. Note that,
 *       (A-s*I)*V = Q*R -> Q'*V*inv(R) = Q'*inv(A-s*I)*Q.
 *    And note that Q'*V*inv(R) is Hermitian if A is, and also that
 *       Q'*V*inv(R)*Y = Y*inv(L) ->  Q'*V*X*L = R*X,
 *    with Y = R*X. So this routine computes X by solving the Hermitian problem
 *    Q'*V*inv(R).
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H             The matrix V'*A*V
 * ldH           The leading dimension of H
 * R             The R factor for the QR decomposition of (A - target*I)*V
 * ldR           The leading dimension of R
 * basisSize     Current size of the orthonormal basis V
 * lrwork        Length of the work array rwork
 * primme        Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hVecs         The orthogonal basis of inv(R) * eigenvectors of QtV/R
 * ldhVecs       The leading dimension of hVecs
 * hU            The eigenvectors of QtV/R
 * ldhU          The leading dimension of hU
 * hVals         The Ritz values of the vectors in hVecs
 * rwork         Workspace
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 Num_dsyev/zheev was unsuccessful
 ******************************************************************************/

static int solve_H_Harm_Sprimme(SCALAR *H, int ldH, SCALAR *QtV, int ldQtV,
   SCALAR *R, int ldR, SCALAR *hVecs, int ldhVecs, SCALAR *hU, int ldhU,
   REAL *hVals, int basisSize, int numConverged, double machEps,
   size_t *lrwork, SCALAR *rwork, int liwork, int *iwork,
   primme_params *primme) {

   int i, ret;
   double *oldTargetShifts, zero=0.0;
   primme_target oldTarget;

   /* Some LAPACK implementations don't like zero-size matrices */
   if (basisSize == 0) return 0;

   /* Return memory requirements */
   if (QtV == NULL) {
      CHKERR(solve_H_RR_Sprimme(QtV, ldQtV, NULL, 0, hVecs, ldhVecs, hVals,
               basisSize, 0, lrwork, rwork, liwork, iwork, primme), -1);
      return 0;
   }

   /* Q'(A-shift*I)Q = QtV*inv(R) */

   Num_copy_matrix_Sprimme(QtV, basisSize, basisSize, ldQtV, hVecs, ldhVecs);
   Num_trsm_Sprimme("R", "U", "N", "N", basisSize, basisSize, 1.0, R, ldR,
         hVecs, ldhVecs);

   /* Compute eigenpairs of Q'(A-shift*I)Q */

   oldTargetShifts = primme->targetShifts;
   oldTarget = primme->target;
   primme->targetShifts = &zero;
   switch(primme->target) {
      case primme_closest_geq:
         primme->target = primme_largest;
         break;
      case primme_closest_leq:
         primme->target = primme_smallest;
         break;
      case primme_closest_abs:
         primme->target = primme_largest_abs;
         break;
      case primme_largest:
         primme->target = primme_closest_geq;
         break;
      case primme_smallest:
         primme->target = primme_closest_leq;
         break;
      case primme_largest_abs:
         primme->target = primme_closest_abs;
         break;
      default:
         assert(0);
   }
   ret = solve_H_RR_Sprimme(hVecs, ldhVecs, NULL, 0, hVecs, ldhVecs, hVals,
         basisSize, 0, lrwork, rwork, liwork, iwork, primme);
   primme->targetShifts = oldTargetShifts;
   primme->target = oldTarget;
   CHKERRM(ret, -1, "Error calling solve_H_RR_Sprimme\n");

   Num_copy_matrix_Sprimme(hVecs, basisSize, basisSize, ldhVecs, hU, ldhU);

   /* Transfer back the eigenvectors to V, hVecs = R\hVecs */

   Num_trsm_Sprimme("L", "U", "N", "N", basisSize, basisSize, 1.0, R, ldR,
         hVecs, ldhVecs);

   /* Move non eligible values to the end */

   if (primme->target == primme_closest_leq ||
       primme->target == primme_closest_geq ||
       primme->target == primme_largest ||
       primme->target == primme_smallest) {

      int *perm = iwork; /* permutation */
      int neligible = 0;   /* number of eligible values */
      int nnoneligible = 0; /* number of non eligible values */
      assert(liwork >= basisSize);

      for (i = 0; i < basisSize; i++) {

         /* Compute |R\x| = |(A-shift*I)\x| */
         REAL hsval = sqrt(REAL_PART(Num_dot_Sprimme(
               basisSize, &hVecs[ldhVecs * i], 1, &hVecs[ldhVecs * i], 1)));

         /* Compute the residual norm */
         REAL ri = sqrt(max(0, (hsval + hVals[i]) * (hsval - hVals[i])));

         /* Add to the proper list */
         if (((primme->target == primme_closest_leq ||
                    primme->target == primme_smallest) &&
                   hVals[i] - ri <= 0.0) ||
               ((primme->target == primme_closest_geq ||
                      primme->target == primme_largest) &&
                     hVals[i] + ri >= 0.0)) {

            perm[neligible++] = i;

         }
         else {
           perm[basisSize - ++nnoneligible] = i;
         }
      }

      int *iwork0 = iwork + basisSize;

      permute_vecs_Rprimme(hVals, 1, basisSize, 1, perm, (REAL*)rwork, iwork0);
      permute_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, perm, rwork, iwork0);
   }

   CHKERR(ortho_Sprimme(hVecs, ldhVecs, NULL, 0, 0, basisSize-1, NULL, 0, 0,
         basisSize, primme->iseed, machEps, rwork, lrwork, NULL), -1);
 
   /* Compute Rayleigh quotient lambda_i = x_i'*H*x_i */

   Num_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
      ldH, hVecs, ldhVecs, 0.0, rwork, basisSize);

   for (i=0; i<basisSize; i++) {
      hVals[i] =
         REAL_PART(Num_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
                  &rwork[basisSize*i], 1));
   }

   return 0;
}

/*******************************************************************************
 * Subroutine solve_H_Ref - This procedure solves the singular value
 *            decomposition of matrix R

* Compute harmonic Rayleigh quotient:                                     *
* lambda^harmonic_i = u_i'*Q'(A-shift*I)Q =                               *
*                   = u_i'*QtBV*inv(R)*u_i =                              *
*                   = u_i'*QtBV*v_i / s_i                                 *
* Find all triplets with values larger than the residual vector norm for  *
* the inv(A) problem,                                                     *
*    r_i = (A-shift*I)\x - x*lambda^harmonic = (I-xx')(A-shift*I)\x.      *
*    r_i'*r_i = x'/(A-shift*I)*(I-xx')/(A-shift*I)*x =                    *
*             = x'/(A-shift*I)/(A-shift*I)*x - (x'/(A-shift*I)*x)^2 =     *
*             = x'/R/R*x - (x'*Q'*V/R*x^2.                                *
* If x = u_i, then                                                        *
*    r_i'*r_i = s_i^-2 - (lambda^harmonic_i)^2                            *
* Find all triplets that (lambda^harmonic_i)^2*1.5 > s_i^-2, which is     *
*    (u_i*QtBV*v_i / s_i)^2*1.5 > s_i^2 ->                                *
*    |u_i*QtBV*v_i|^2 > 2/3                                               *
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H             The matrix V'*A*V
 * ldH           The leading dimension of H
 * R             The R factor for the QR decomposition of (A - target*I)*V
 * ldR           The leading dimension of R
 * basisSize     Current size of the orthonormal basis V
 * lrwork        Length of the work array rwork
 * primme        Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hVecs         The right singular vectors of R
 * ldhVecs       The leading dimension of hVecs
 * hU            The left singular vectors of R
 * ldhU          The leading dimension of hU
 * hSVals        The singular values of R
 * hVals         The Ritz values of the vectors in hVecs
 * rwork         Workspace
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 was unsuccessful
 ******************************************************************************/

static int solve_H_Ref_Sprimme(SCALAR *H, int ldH, SCALAR *QtBV, int ldQtBV,
      SCALAR *hVecs, int ldhVecs, SCALAR *hVecsRot, int ldhVecsRot, SCALAR *hU,
      int ldhU, REAL *hSVals, SCALAR *R, int ldR, REAL *hVals, int basisSize,
      int targetShiftIndex, size_t *lrwork, SCALAR *rwork, int liwork,
      int *iwork, primme_params *primme) {

   int i, j, k, l; /* Loop variables    */
   int info; /* error value */

   /* Some LAPACK implementations don't like zero-size matrices */
   if (basisSize == 0) return 0;

   /* Return memory requirements */
   if (H == NULL) {
      size_t lrwork0 = 0;
      CHKERR(solve_H_RR_Sprimme(NULL, 0, NULL, 0, NULL, 0, NULL, basisSize, 0,
                   &lrwork0, rwork, liwork, iwork, primme), -1);
      {
         SCALAR rwork0;
#ifdef USE_COMPLEX
         lrwork0 += (size_t)(3*basisSize);
         CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, R, basisSize,
                       NULL, NULL, basisSize, hVecs, basisSize, &rwork0, -1,
                       hVals, &info), info), -1);
#else
         CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, R, basisSize,
                       NULL, NULL, basisSize, hVecs, basisSize, &rwork0, -1,
                       &info), info), -1);
#endif
         lrwork0 += (size_t)REAL_PART(rwork0);
      }
      lrwork0 += (size_t)basisSize*(size_t)basisSize; /* aux for transpose V and hemm */
      *lrwork = max(*lrwork, lrwork0);
      /* for perm and permute_vecs */
      *iwork = max(*iwork, 2*basisSize);
      return 0;
   }

   /* hVecsRot = I */

   if (hVecsRot) {
      Num_zero_matrix_Sprimme(hVecsRot, basisSize, basisSize, ldhVecsRot);
      for (i=0; i<basisSize; i++)
         hVecsRot[ldhVecsRot*i+i] = 1.0;
   }

   /* Copy R into hVecs */
   Num_copy_matrix_Sprimme(R, basisSize, basisSize, ldR, hVecs, ldhVecs);

   /* Note gesvd returns transpose(V) rather than V and sorted in descending  */
   /* order of the singular values                                            */

#ifdef USE_COMPLEX
   /* zgesvd requires 5*basisSize double work space; booked 3*basisSize complex double */
   assert(*lrwork >= (size_t)(3*basisSize));
   CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, hVecs, ldhVecs,
         hSVals, hU, ldhU, hVecs, ldhVecs, rwork+3*basisSize,
         TO_INT(*lrwork-(size_t)(3*basisSize)), (REAL*)rwork, &info), info),
         -1);
#else
   CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, hVecs, ldhVecs,
         hSVals, hU, ldhU, hVecs, ldhVecs, rwork, TO_INT(*lrwork), &info),
         info), -1);
#endif

   /* Transpose back V */

   assert(*lrwork >= (size_t)basisSize*(size_t)basisSize);
   for (j=0; j < basisSize; j++) {
      for (i=0; i < basisSize; i++) { 
         rwork[basisSize*j+i] = CONJ(hVecs[ldhVecs*i+j]);
      }
   }
   Num_copy_matrix_Sprimme(rwork, basisSize, basisSize, basisSize, hVecs, ldhVecs);

   /* Rearrange V, hSVals and hU in ascending order of singular value         */
   /* if not targeting extreme values.                                        */

   if (primme->target == primme_closest_abs 
         || primme->target == primme_closest_leq
         || primme->target == primme_closest_geq) {
      int *perm = iwork;
      int *iwork0 = iwork + basisSize;
      assert(liwork >= 2*basisSize);

      for (i=0; i<basisSize; i++) perm[i] = basisSize-1-i;
      permute_vecs_Rprimme(hSVals, 1, basisSize, 1, perm, (REAL*)rwork, iwork0);
      permute_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, perm, rwork, iwork0);
      permute_vecs_Sprimme(hU, basisSize, basisSize, ldhU, perm, rwork, iwork0);
   }

   /* If no advance technique is require, just compute the Rayleigh quotients */
   /* hVals(i) = x_i'*H*x_i                                                   */

   if (hVecsRot == NULL ||
         primme->projectionParams.projection == primme_proj_refined) {

      Num_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
            ldH, hVecs, ldhVecs, 0.0, rwork, basisSize);

      for (i=0; i<basisSize; i++) {
         hVals[i] = REAL_PART(Num_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
                  &rwork[basisSize*i], 1));
      }

      return 0;
   }

   /* Move triplets that |u_i'*QtBV*v_i| > sqrt(2/3) first */

   /* 1) QtBVhVecs = QtBV * hVecs */

   SCALAR *rwork0 = rwork, *QtBVhVecs;
   size_t rworkSize0 = *lrwork;
   CHKERR(WRKSP_MALLOC_PRIMME((size_t)basisSize * basisSize, &QtBVhVecs,
                &rwork0, &rworkSize0), -1);

   Num_gemm_Sprimme("N", "N", basisSize, basisSize, basisSize, 1.0, QtBV,
         ldQtBV, hVecs, ldhVecs, 0.0, QtBVhVecs, basisSize);

   /* 2) Put first the triplets that sqrt(1-|u_i'*QtBV*v_i|^2) < s(i)*err,    */
   /*    and then put the rest.                                               */

   int *perm;           /* Permutation that put selected triplets first */
   int asel;            /* Number of selected triplets */
   int *iwork0 = iwork, iworkSize0 = liwork;

   CHKERR(WRKSP_MALLOC_PRIMME(basisSize, &perm, &iwork0, &iworkSize0), -1);
   if (primme->projectionParams.projection == primme_proj_harmonic) {
      asel = basisSize;
      for (i=0; i<basisSize; i++) perm[i] = i;
   }
   else {
      int *excluded;   /* Indices of non-selected triplets */
      int *iwork1 = iwork0, iworkSize1 = iworkSize0;
      CHKERR(WRKSP_MALLOC_PRIMME(basisSize, &excluded, &iwork1, &iworkSize1),
            -1);

      REAL err = primme->stats.estimateResidualError / primme->stats.estimateLargestSVal;
      //printf("err %g ", err);
      for (i=j=0, asel=0; i<basisSize; i++) {
         REAL qv = max(1.0, ABS(Num_dot_Sprimme(basisSize, &hU[ldhU * i], 1,
                                  &QtBVhVecs[basisSize * i], 1)));
         // REAL qv = ABS(Num_dot_Sprimme(basisSize, &hU[ldhU * i], 1,
         //                          &QtBVhVecs[basisSize * i], 1));
         //printf("( %g %g ) ", qv, hSVals[i]);
         //if (sqrt(max(0.0, 1.0 - qv*qv)) < hSVals[i] * err) {
         //if (sqrt(max(0.0, 1.0 - qv*qv)) < .5) {
         if (qv > sqrt(2./3.)) {
            perm[asel++] = i;
         }
         else {
            excluded[j++] = i;
         }
      }
      //printf(" asel %d\n", asel);

      /* perm = [perm excluded] */

      for (i=asel, j=0; j<basisSize - asel; i++, j++)
         perm[i] = excluded[j];

   }

   /* 3) Rearrange hVecs, hSVals, hU and QtBVhVecs according to perm */

   permute_vecs_Rprimme(hSVals, 1, basisSize, 1, perm, (REAL*)rwork0, iwork0);
   permute_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, perm, rwork0, iwork0);
   permute_vecs_Sprimme(hU, basisSize, basisSize, ldhU, perm, rwork0, iwork0);
   permute_vecs_Sprimme(QtBVhVecs, basisSize, basisSize, basisSize, perm, rwork0, iwork0);

   /* Apply two harmonic projections, one with the bases [0:asel-1] and */
   /* with the bases [asel:basisSize]                                   */

   /* aQtBViR = hU(:,p)'*QtBV/R*hU(:,p) =                                  */
   /*         = hU(:,p)'*QtBV*hVecs(:,p)/diag(hSVals(p))                   */
   /*         = hU(:,p)'*QtBVhVecs(:,p)/diag(hSVals(p))                    */

   Num_gemm_Sprimme("C", "N", asel, asel, basisSize, 1.0, hU, ldhU, QtBVhVecs,
         basisSize, 0.0, hVecsRot, ldhVecsRot);

   Num_gemm_Sprimme("C", "N", basisSize - asel, basisSize - asel, basisSize,
         1.0, &hU[ldhU * asel], ldhU, &QtBVhVecs[basisSize * asel], basisSize,
         0.0, &hVecsRot[ldhVecsRot * asel + asel], ldhVecsRot);

   int phase;
   for (phase=0; phase<2; phase++) {
      int i0, aBasisSize;
      if (phase == 0) { /* Use bases [0:asel-1] */
         i0 = 0; aBasisSize = asel;
      }
      else { /* Use bases [asel:basisSize] */
         i0 = asel; aBasisSize = basisSize - asel;
      }

      /* aQtBViR = hU(:,p)'*QtBV/R*hU(:,p) =                                  */
      /*         = hU(:,p)'*QtBV*hVecs(:,p)/diag(hSVals(p))                   */
      /*         = hU(:,p)'*QtBVhVecs(:,p)/diag(hSVals(p))                    */

      SCALAR *aQtBViR = &hVecsRot[ldhVecsRot*i0 + i0];

      for (i=1 /* first column isn't touched */; i<aBasisSize; i++) {
         for (j=0; j<aBasisSize; j++) {
            aQtBViR[ldhVecsRot * i + j] *=
                  (fabs(hSVals[i0 + i]) == 0.0 ? 0.0
                                               : hSVals[i0] / hSVals[i0 + i]);
         }
      }

      /* Transform target and targetShift */

      double *oldTargetShifts, zero=0.0;
      primme_target oldTarget;

      oldTargetShifts = primme->targetShifts;
      oldTarget = primme->target;
      primme->targetShifts = &zero;
      switch(primme->target) {
         case primme_closest_geq:
            primme->target = primme_largest;
            break;
         case primme_closest_leq:
            primme->target = primme_smallest;
            break;
         case primme_closest_abs:
            primme->target = primme_largest_abs;
            break;
         default:
            assert(0);
      }

      /* Find the eigenpairs of the projected problem aQtBViR */

      int ret = solve_H_RR_Sprimme(aQtBViR, ldhVecsRot, NULL, 0,
            &hVecsRot[ldhVecsRot * i0 + i0], ldhVecsRot, &hVals[i0], aBasisSize,
            0, lrwork, rwork, liwork, iwork, primme);
      primme->targetShifts = oldTargetShifts;
      primme->target = oldTarget;
      CHKERRM(ret, -1, "Error calling solve_H_RR_Sprimme\n");

      /* Transfer back the eigenvectors to the right space,                   */
      /*    ahVecs = qr(R\ahVecs)                                             */

      for (i=0; i<aBasisSize; i++) {
         for (j=1 /* first row isn't touched */; j<aBasisSize; j++) {
            hVecsRot[ldhVecsRot * (i0 + i) + j + i0] *=
                  (fabs(hSVals[i0 + j]) == 0.0 ? 0.0
                                               : hSVals[i0] / hSVals[i0 + j]);
         }
      }

      CHKERR(ortho_Sprimme(&hVecsRot[ldhVecsRot * i0 + i0], ldhVecsRot, NULL, 0,
                   0, aBasisSize - 1, NULL, 0, 0, aBasisSize, primme->iseed,
                   MACHINE_EPSILON, rwork, lrwork, NULL), -1);

      /* hVecs(:,1:aBasisSize) = hVecs(:,1:aBasisSize)*hVecsRot */

      Num_gemm_Sprimme("N", "N", basisSize, aBasisSize, aBasisSize, 1.0,
            &hVecs[ldhVecs * i0], ldhVecs, &hVecsRot[ldhVecsRot * i0 + i0],
            ldhVecsRot, 0.0, rwork, basisSize);
      Num_copy_matrix_Sprimme(rwork, basisSize, aBasisSize, basisSize,
            &hVecs[ldhVecs * i0], ldhVecs);
   }

   /* Sort the pairs by the harmonic-Ritz quotients */

   if (primme->projectionParams.projection == primme_proj_harmonic) {

      /* The pairs are already sorted */

      for (i=0; i<basisSize; i++) perm[i] = i;

   }
   else if (primme->target == primme_closest_abs) {

      /* Merge the two sorted lists hVals[0:asel-1] and hVals[asel:basisSize] */
      /* with respect to the absolute value                                   */

      for (i = 0, j = asel, k = 0; k < basisSize; k++) {
         if (j >= basisSize || (i < asel &&
                                     fabs(hVals[i]) * hSVals[asel] >=
                                           fabs(hVals[j]) * hSVals[0])) {
            perm[k] = i++;
         } else {
            perm[k] = j++;
         }
      }
   }
   else {

      /* Merge the two sorted lists hVals[0:asel-1] and hVals[asel:basisSize] */
      /* with respect the value                                               */

      int pm = (primme->target == primme_closest_geq ? 1 : -1);
      for (i = 0, j = asel, k = 0; k < basisSize; k++) {
         if (j >= basisSize ||
               (i < asel &&
                     (hVals[i] * hSVals[asel] - hVals[j] * hSVals[0]) * pm >=
                           0.0)) {
            perm[k] = i++;
         } else {
            perm[k] = j++;
         }
      }
   }

   /* Compute the Rayleigh quotients of the pairs */

   Num_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
         ldH, hVecs, ldhVecs, 0.0, rwork, basisSize);

   for (i=0; i<basisSize; i++) {
      hVals[i] = REAL_PART(Num_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
               &rwork[basisSize*i], 1));
   }

   /* Rearrange hVals, hVecs, hVecsRot */

   permute_vecs_Rprimme(hVals, 1, basisSize, 1, perm, (REAL *)rwork, iwork0);
   permute_vecs_Sprimme(
         hVecs, basisSize, basisSize, ldhVecs, perm, rwork, iwork0);
   permute_vecs_Sprimme(
               hVecsRot, basisSize, basisSize, ldhVecsRot, perm, rwork, iwork0);

   return 0;
}

/*******************************************************************************
 * Subroutine solve_H_brcast - This procedure broadcast the solution of the
 *       projected problem (hVals, hSVals, hVecs, hU) from process 0 to the rest.
 *
 * NOTE: the optimal implementation will use an user-defined broadcast function.
 *       To ensure backward compatibility, we used globalSum instead.
 * 
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * basisSize      The dimension of hVecs, hU, hVals and hSVals
 * lrwork         Length of the work array rwork
 * primme         Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hU             The left singular vectors of R or the eigenvectors of QtV/R
 * ldhU           The leading dimension of hU
 * hVecs          The coefficient vectors such as V*hVecs will be the Ritz vectors
 * ldhVecs        The leading dimension of hVecs
 * hVals          The Ritz values
 * hSVals         The singular values of R
 * rwork          Workspace
 *
 * Return Value
 * ------------
 * error code                          
 ******************************************************************************/

static int solve_H_brcast_Sprimme(int basisSize, SCALAR *hU, int ldhU,
      SCALAR *hVecs, int ldhVecs, SCALAR *hVecsRot, int ldhVecsRot, REAL *hVals,
      REAL *hSVals, size_t *lrwork, SCALAR *rwork, primme_params *primme) {

   int n=0;                            /* number of SCALAR packed */
   SCALAR *rwork0 = rwork;             /* next SCALAR free */
   const size_t c = sizeof(SCALAR)/sizeof(REAL);

   /* Do nothing in sequential */

   if (primme->numProcs <= 1) return 0;

   /* Return memory requirements */

   if (hVecs == NULL) {
      switch (primme->projectionParams.projection) {
         case primme_proj_RR:
            /* Broadcast hVecs, hVals */
            *lrwork = max(*lrwork, (size_t)basisSize*(basisSize+1));
            break;

         // case primme_proj_harmonic:
         //    /* Broadcast hVecs, hVals, hU */
         //    *lrwork = max(*lrwork, (size_t)basisSize*(2*basisSize+1));
         //    break;

         case primme_proj_refined:
         case primme_proj_harmonic:
         case primme_proj_refined_RR:
         case primme_proj_refined_harmonic:
            /* Broadcast hVecs, hVecsRot, hVals, hU, hSVals */
            *lrwork = max(*lrwork, (size_t)basisSize*(3*basisSize+2));
            break;

         default:
            assert(0);
      }
      return 0;
   }
   assert(*lrwork >=
          (size_t)basisSize * (((hU ? 2 : 1) + (hVecsRot ? 1 : 0)) * basisSize +
                                    (hSVals ? 2 : 1)));

   /* Pack hVecs */

   if (primme->procID == 0) {
      Num_copy_matrix_Sprimme(hVecs, basisSize, basisSize, ldhVecs, rwork0,
            basisSize);
   }
   n += basisSize*basisSize;
   rwork0 += basisSize*basisSize;

   /* Pack hU */

   if (hU) {
      if (primme->procID == 0) {
         Num_copy_matrix_Sprimme(hU, basisSize, basisSize, ldhU, rwork0,
               basisSize);
      }
      n += basisSize*basisSize;
      rwork0 += basisSize*basisSize;
   }

   /* Pack hVecsRot */

   if (hVecsRot) {
      if (primme->procID == 0) {
         Num_copy_matrix_Sprimme(
               hVecsRot, basisSize, basisSize, ldhVecsRot, rwork0, basisSize);
      }
      n += basisSize*basisSize;
      rwork0 += basisSize*basisSize;
   }

   /* Pack hVals */

   if (primme->procID == 0) {
      rwork0[basisSize/c] = 0.0; /* When complex, avoid to reduce with an   */
                                   /* uninitialized value                     */
      Num_copy_matrix_Rprimme(hVals, basisSize, 1, basisSize, (REAL*)rwork0,
            basisSize);
   }
   n += (basisSize + c-1)/c;
   rwork0 += (basisSize + c-1)/c;

   /* Pack hSVals */

   if (hSVals) {
      if (primme->procID == 0) {
         rwork0[basisSize/c] = 0.0; /* When complex, avoid to reduce with an*/
                                      /* uninitialized value                  */
         Num_copy_matrix_Rprimme(hSVals, basisSize, 1, basisSize, (REAL*)rwork0,
               basisSize);
      }
      n += (basisSize + c-1)/c;
      rwork0 += (basisSize + c-1)/c;
   }

   /* If this is not proc 0, zero the input rwork */

   if (primme->procID != 0) {
      Num_zero_matrix_Sprimme(rwork, n, 1, n);
   }
 
   /* Perform the broadcast by using a reduction */

   CHKERR(globalSum_Sprimme(rwork, rwork, n, primme), -1);
   rwork0 = rwork;

   /* Unpack hVecs */

   Num_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hVecs,
         ldhVecs);
   rwork0 += basisSize*basisSize;

   /* Unpack hU */

   if (hU) {
      Num_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hU,
            ldhU);
      rwork0 += basisSize*basisSize;
   }

   /* Unpack hVecsRot */

   if (hVecsRot) {
      Num_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hVecsRot,
            ldhVecsRot);
      rwork0 += basisSize*basisSize;
   }

   /* Unpack hVals */

   Num_copy_matrix_Rprimme((REAL*)rwork0, basisSize, 1, basisSize, hVals,
         basisSize);
   rwork0 += (basisSize + c-1)/c;

   /* Unpack hSVals */

   if (hSVals) {
      Num_copy_matrix_Rprimme((REAL*)rwork0, basisSize, 1, basisSize, hSVals,
               basisSize);
      rwork0 += (basisSize + c-1)/c;
   }

   return 0;
}
