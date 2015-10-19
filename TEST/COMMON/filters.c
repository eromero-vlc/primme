/*******************************************************************************
 *   PRIMME PReconditioned Iterative MultiMethod Eigensolver
 *   Copyright (C) 2005  James R. McCombs,  Andreas Stathopoulos
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
 ******************************************************************************/

#include "filters.h"
#include "primme.h"
#include "native.h"
#include "wtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../../PRIMMESRC/DSRC/numerical_d.h"
#include "shared_utils.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static void Apply_original(void *x, void *y, int *blockSize, filter_params *filter,
                    primme_params *primme, int stats);

static void Apply_filter_jackson(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int damping, int delta, int stats);
static void jac2Plt(int m, double a, double b, int damping, double *mu);
static void jacPlt(int m, double alpha, int up, int damping, double *mu);
static void jacDelta(int m, double alpha, int damping, double *mu);
static void jacksonAv(PRIMME_NUM *x, PRIMME_NUM *y, int *blockSize, filter_params *filter,
               primme_params *primme, double *mu, int stats);
static void Apply_filter_cheb(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int lowpass, int stats);
static void Apply_filter_cheb_interior(void *x, void *y, int *blockSize, filter_params *filter,
                                 primme_params *primme, int stats);
void Apply_filter_modelcheb(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int stats);
void modelChebAv(primme_params *primme, PRIMME_NUM *xvec, int m, double a, double b, double lb, double ub, PRIMME_NUM *yvec, PRIMME_NUM *aux, filter_params *filter, int stats);
static void eval_filter(double *x, double *y, int n, filter_params *filter, primme_params *primme);
static void ortho2(PRIMME_NUM *y, const PRIMME_NUM *x, PRIMME_NUM *o, int n, int cols);
#ifdef USE_FILTLAN
static void Apply_filter_filtlan(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int stats);
#endif
#ifdef USE_FEAST
static void Apply_filter_feast(void *x, void *y, int *blockSize, filter_params *filter,
                               primme_params *primme, int stats);
#endif
static void Apply_filter_augmented(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int stats);
static void Apply_filter_normalequation(void *x, void *y, int *blockSize, filter_params *filter,
                                 primme_params *primme, int stats);
static void Apply_filter_gamma5(void *x, void *y, int *blockSize, filter_params *filter,
                                 primme_params *primme, int stats);
static void Apply_filter_gamma5sq(void *x, void *y, int *blockSize, filter_params *filter,
                                 primme_params *primme, int stats);
static int check_filter(double a, int d0, int d1, int setBounds, filter_params *filter, primme_params *primme);

double elapsedTimeAMV, elapsedTimeFilterMV;
int numFilterApplies;

/******************************************************************************
 * Apply a filter functions
 *
 ******************************************************************************/

static void Apply_original(void *x, void *y, int *blockSize, filter_params *filter,
                    primme_params *primme, int stats) {
   int j;
   double t0;

   t0 = primme_get_wtime();
   if (filter->prodIfFullRange) {
      filter->matvec(x, y, blockSize, primme);
   }
   else {
      for(j=0; j<*blockSize*primme->nLocal; j++) ((double*)y)[j] = ((double*)x)[j]; 
   }
   if (stats) elapsedTimeAMV += primme_get_wtime() - t0;

}


/******************************************************************************
 * Configure and apply a filter
 *
 ******************************************************************************/

void Apply_filter(void *x, void *y, int *blockSize, filter_params *filter,
                  primme_params *primme, int stats) {

   switch(filter->filter) {
      case 0:
         Apply_original(x, y, blockSize, filter, primme, stats);
         break;
      case 1:
         Apply_filter_jackson(x, y, blockSize, filter, primme, 0, 0, stats);
         break;
      case 2:
         Apply_filter_jackson(x, y, blockSize, filter, primme, 1, 0, stats);
         break;
      case 3:
         Apply_filter_jackson(x, y, blockSize, filter, primme, 2, 0, stats);
         break;
      case 4:
         Apply_filter_cheb(x, y, blockSize, filter, primme, 1, stats);
         break;
      case 5:
         Apply_filter_cheb(x, y, blockSize, filter, primme, 0, stats);
         break;
#ifdef USE_FILTLAN
      case 6:
         Apply_filter_filtlan(x, y, blockSize, filter, primme, stats);
         break;
#endif /* FILTLAN */
      case 7:
         Apply_filter_cheb_interior(x, y, blockSize, filter, primme, stats);
         break;
      case 8:
         Apply_filter_jackson(x, y, blockSize, filter, primme, 0, 1, stats);
         break;
      case 9:
         Apply_filter_jackson(x, y, blockSize, filter, primme, 1, 1, stats);
         break;
      case 10:
         Apply_filter_jackson(x, y, blockSize, filter, primme, 2, 1, stats);
         break;
      case 11:
         Apply_filter_modelcheb(x, y, blockSize, filter, primme, stats);
         break;
#ifdef USE_FEAST
      case 12:
         Apply_filter_feast(x, y, blockSize, filter, primme, stats);
         break;
#endif /* FEAST */
      case 13:
         Apply_filter_augmented(x, y, blockSize, filter, primme, stats);
         break;
      case 14:
         Apply_filter_normalequation(x, y, blockSize, filter, primme, stats);
         break;
      case 15:
         Apply_filter_gamma5(x, y, blockSize, filter, primme, stats);
         break;
      case 16:
         Apply_filter_gamma5sq(x, y, blockSize, filter, primme, stats);
         break;
      default:
         fprintf(stderr, "ERROR(Apply_filter): Invalid filter '%d'\n", filter->filter);
         return;
   }
}

void getBounds(filter_params *filter, primme_params *primme, double *lb, double *ub) {
   if (lb) {
      if (filter->lowerBound == 0) {
         *lb = filter->minEig;
      } else if (filter->lowerBound < 8) {
         *lb = primme->RitzValuesForPreconditioner[filter->lowerBound-1];
         if (*lb <= -HUGE_VAL) *lb = filter->minEig;
      } else if (filter->lowerBound == 8) {
         *lb = filter->lowerBoundFix;
      } else {
         fprintf(stderr, "ERROR(getBounds): Invalid lowerBound '%d'\n", filter->lowerBound);
         return;
      }
      if (filter->lowerBoundFix != filter->upperBoundFix && min(max(*lb, filter->lowerBoundFix), filter->upperBoundFix) != *lb) {
         if (primme->outputFile)
            fprintf(primme->outputFile, "warning(getBounds): lowerBound(%d)=%g out of fix limits! Fixed!\n", filter->lowerBound, *lb);
         *lb = min(max(*lb, filter->lowerBoundFix), filter->upperBoundFix);
      }
      if (min(max(*lb, filter->minEig), filter->maxEig) != *lb) {
         if (primme->outputFile)
            fprintf(primme->outputFile, "warning(getBounds): lowerBound(%d)=%g out of limits! Fixed!\n", filter->lowerBound, *lb);
         *lb = min(max(*lb, filter->minEig), filter->maxEig);
      }
   }
   if (ub) {
      if (filter->upperBound == 0) {
         *ub = filter->maxEig;
      } else if (filter->upperBound < 8) {
         *ub = primme->RitzValuesForPreconditioner[filter->upperBound-1];
         if (*ub <= -HUGE_VAL) *ub = filter->maxEig;
      } else if (filter->upperBound == 8) {
         *ub = filter->upperBoundFix;
      } else {
         fprintf(stderr, "ERROR(getBounds): Invalid upperBound '%d'\n", filter->upperBound);
         return;
      }
      if (filter->lowerBoundFix != filter->upperBoundFix && min(max(*ub, filter->lowerBoundFix), filter->upperBoundFix) != *ub) {
         if (primme->outputFile)
            fprintf(primme->outputFile, "warning(getBounds): upperBound(%d)=%g out of fix limits! Fixed!\n", filter->upperBound, *ub);
         *ub = min(max(*ub, filter->lowerBoundFix), filter->upperBoundFix);
      }
       if (min(max(*ub, filter->minEig), filter->maxEig) != *ub) {
         if (primme->outputFile)
            fprintf(primme->outputFile, "warning(getBounds): upperBound(%d)=%g out of limits! Fixed!\n", filter->upperBound, *ub);
         *ub = min(max(*ub, filter->minEig), filter->maxEig);
      }
   }
}

void getBoundsTuned(filter_params *filter, primme_params *primme, double *lb, double *ub) {
   if (filter->checkEps == 0) {
      getBounds(filter, primme, lb, ub);
   } else {
      *lb = filter->lowerBoundTuned;
      *ub = filter->upperBoundTuned;
   }
}

/******************************************************************************
 * Applies a polynomial filter as a preconditioner
 *
******************************************************************************/

static void Apply_filter_jackson(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int damping, int delta, int stats) {

   double lowerBound, upperBound, lBNormalized, uBNormalized;
   double *mu;

   if (!blockSize) return;

   getBoundsTuned(filter, primme, &lowerBound, &upperBound);
   lBNormalized = (lowerBound - filter->minEig)/(filter->maxEig - filter->minEig)*2. - 1;
   uBNormalized = (upperBound - filter->minEig)/(filter->maxEig - filter->minEig)*2. - 1;
   if (delta) lBNormalized = uBNormalized = (lBNormalized + uBNormalized)/2.;

   if (lBNormalized <= -.99 && uBNormalized >= .99) {
      Apply_original(x, y, blockSize, filter, primme, stats);
      return;
   }
   if (lBNormalized <= -.99 && uBNormalized <= -.99) {
      filter_params f = *filter;
      f.lowerBound = 3;
      f.upperBound = 7;
      f.degrees = min(40, f.degrees);
      Apply_filter_cheb(x, y, blockSize, &f, primme, 1, stats);
      return;
   }
   if (lBNormalized >= .99 && uBNormalized >= .99) {
      filter_params f = *filter;
      f.lowerBound = 3;
      f.upperBound = 7;
      f.degrees = min(40, f.degrees);
      Apply_filter_cheb(x, y, blockSize, &f, primme, 0, stats);
      return;
   }
   mu = (double *)primme_calloc(filter->degrees+1, sizeof(double), "mu");
   if (lBNormalized == uBNormalized) { // Delta filter
      jacDelta(filter->degrees, uBNormalized, damping, mu);
   } else if (lBNormalized <= -.99) {   // low-pass filter
      jacPlt(filter->degrees, uBNormalized, 0, damping, mu);
   } else if (uBNormalized >= .99) {   // high-pass filter
      jacPlt(filter->degrees, lBNormalized, 1, damping, mu);
   } else {  // mid-pass filter
      jac2Plt(filter->degrees, lBNormalized, uBNormalized, damping, mu);
   }
   jacksonAv((PRIMME_NUM *)x, (PRIMME_NUM *)y, blockSize, filter, primme, mu, stats);
   free(mu);
}


/******************************************************************************
 * Computes Cheb. expansion of a hat function in interval [-1, 1].
 *
 * INPUT
 * -----
 *    m        #degrees of the expansion 
 *    a,b      interval where function is ~1
 *    damping  0: no damping; 1: Jackson; 2: Lanczos sigma damping
 *
 * OUTPUT
 * ------
 *    mu,      (m+1)-vector with the expansion coefficients
******************************************************************************/
static void jac2Plt(int m, double a, double b, int damping, double *mu) {
   double jac, a1, a2, thetJ, thetL, beta1, beta2;
   int k;

   if (a < -1. || b > 1. || b < a) {
      fprintf(stderr, "Error in some bound: %f, %f\n", a, b);
      return;
   }
   if (b-a < 1e-2) {
      jacDelta(m, (b+a)/2., damping, mu);
      return;
   }
   thetJ = M_PI/(m+2);
   thetL = M_PI/(m+1);
   a1 = 1./(m+2);
   a2 = sin(thetJ);
   beta1 = acos(a);
   beta2 = acos(b);
   for (k=0; k<=m; ++k) {
      switch(damping) {
         case 0:
            jac = 1.;
            break;
         case 1:
            // Note: slightly simpler formulas for jackson:
            jac = a1*sin((k+1)*thetJ)/a2 + (1.-(k+1)*a1)*cos(k*thetJ);
            break;
         case 2:
            // Lanczos sigma-damping:
            if (k == 0) jac = 1.;
            else jac = sin(k*thetL)/(k*thetL);
            break;
         default:
            fprintf(stderr, "ERROR(jac2Plt): Invalid damping '%d'\n", damping);
            return;
      }
      if (k == 0) mu[k] = -jac*(beta2-beta1)/M_PI;
      else mu[k] = -2.*jac*(sin(k*beta2)-sin(k*beta1))/(M_PI*k);
   }
}

/******************************************************************************
 * Computes Cheb. expansion of heaviside function in interval [-1, 1].
 *
 * INPUT
 * -----
 *    m           #degrees of the expansion 
 *    alpha, up   if up == 0, function is ~1 if x<alpha else ~0;
 *                if up != 0, function is ~0 if x<alpha else ~1
 *    damping     0: no damping; 1: Jackson; 2: Lanczos sigma damping
 *
 * OUTPUT
 * ------
 *    mu,      (m+1)-vector with the expansion coefficients
******************************************************************************/
static void jacPlt(int m, double alpha, int up, int damping, double *mu) {
   double jac, a1, a2, thet, beta;
   int k;

   if (alpha < -1. || alpha > 1.) {
      fprintf(stderr, "Error in some bound: %f\n", alpha);
      return;
   }
   thet = M_PI/(m+1);
   a1 = 1./(m+2);
   a2 = sin(thet);
   beta = acos(alpha);
   for (k=0; k<=m; ++k) {
      switch(damping) {
         case 0:
            jac = 1.;
            break;
         case 1:
            jac = a1*sin((k+1)*thet)/a2 + (1-(k+1)*a1)*cos(k*thet);
            break;
         case 2:
            // Lanczos sigma-damping:
            if (k == 0) jac = 1.;
            else jac = sin(k*thet)/(k*thet);
            break;
         default:
            fprintf(stderr, "ERROR(jacPlt): Invalid damping '%d'\n", damping);
            return;
      }
      if ( up == 0) {
         if (k == 0) mu[k] = jac*(1.0 - beta/M_PI);
         else mu[k] = -2.*jac*sin(k*beta)/(M_PI*k);
      } else {
         if (k == 0) mu[k] = 1. - jac*(1. - beta/M_PI);
         else mu[k] = 2.*jac*sin(k*beta)/(M_PI*k);
      }
   }
}

/******************************************************************************
 * Computes Cheb. expansion of delta function in interval [-1, 1].
 *
 * INPUT
 * -----
 *    m           #degrees of the expansion 
 *    alpha       point where the function is ~1, others are ~0
 *    damping     0: no damping; 1: Jackson
 *
 * OUTPUT
 * ------
 *    mu,      (m+1)-vector with the expansion coefficients
******************************************************************************/
static void jacDelta(int m, double alpha, int damping, double *mu) {
   double jac, a1, a2, thet, beta, eta;
   int k;

   if (alpha < -1. || alpha > 1.) {
      fprintf(stderr, "Error in some bound: %f\n", alpha);
      return;
   }
   thet = M_PI/(m+1);
   a1 = 1./(m+2);
   a2 = sin(thet);
   beta = acos(alpha);
   eta = 2./(M_PI*sqrt(1.-alpha*alpha));
   for (k=0; k<=m; ++k) {
      switch(damping) {
         case 0:
            jac = 1.;
            break;
         case 1: case 2:
            jac = ((1.-k*a1)*sin(thet)*cos(k*thet) + a1*cos(thet)*sin(k*thet))/a2;
            break;
         default:
            fprintf(stderr, "ERROR(jac2Plt): Invalid damping '%d'\n", damping);
            return;
      }
      if (k == 0) mu[k] = jac*eta/2.;
      else mu[k] = jac*cos(k*beta)*eta;
   }
}

/******************************************************************************
 * Computes p(A)*x being p a polynomial expressed Cheb. coefficients.
 *
 * INPUT
 * -----
 *    primme      primme_params
 *    xvec        input nLocal-vector
 *    m           #degrees of the expansion
 *    mu          (m+1)-vector with the expansion coefficients
 *    a,b         extreme values of the spectrum of A
 *    aux         auxiliary vector of 3*nLocal elements
 *
 * OUTPUT
 * ------
 *    yvec        output nLocal-vector
******************************************************************************/
static void jacksonAv(PRIMME_NUM *xvec, PRIMME_NUM *yvec, int *blockSize, filter_params *filter,
               primme_params *primme, double *mu, int stats) {
   PRIMME_NUM *vk, *vkm1, *vkp1, vkp1_j, *aux;
   double c, w, scal;
   int j, j0, k, k0, bs = *blockSize;
   double t0;

   aux = (PRIMME_NUM *)primme_calloc(primme->nLocal*3*bs, sizeof(PRIMME_NUM), "aux");
   vk = aux;
   vkp1 = &aux[primme->nLocal*bs];
   vkm1 = &aux[primme->nLocal*2*bs];
   c = (filter->maxEig + filter->minEig)/2.;
   w = (filter->maxEig - filter->minEig)/2.;

   for (k=0, k0=primme->nLocal*bs; k<k0; ++k) {
      yvec[k] = 0.;
      vk[k] = xvec[k];
      vkm1[k] = 0.;
   }
   t0 = primme_get_wtime();
   for (k=0; k<=filter->degrees; ++k) {
      for (j=0, j0=primme->nLocal*bs; j<j0; ++j) yvec[j] += mu[k]*vk[j];
      if (k == 0) scal = 1./w;
      else scal = 2./w;
      filter->matvec(vk, vkp1, blockSize, primme);
      if (stats) primme->stats.numMatvecs += bs;
      for (j=0, j0=primme->nLocal*bs; j<j0; ++j) {
         vkp1_j = scal*(vkp1[j]-c*vk[j]) - vkm1[j];
         vkm1[j] = vk[j];
         vk[j] = vkp1_j;
      }
   }
   if (stats) elapsedTimeFilterMV += primme_get_wtime() - t0;
   if (stats) numFilterApplies += bs;
   free(aux);
}

/******************************************************************************
 * Use Chebyshev filter like in Zhou, "A Chebyshev-Davidson algorithm for large
 * symmetric eigenvalues", 2007.
 *
 * INPUT
 * -----
 *    m        #degrees of the expansion 
 *    a,b      [a,b] interval where function is ~0
 *    a0       min eig of A
 *    damping  0: no damping; 1: Jackson; 2: Lanczos sigma damping
 *
******************************************************************************/
static void Apply_filter_cheb(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int lowpass, int stats) {
   double a,b, a0;

   double e, c, sigma, sigma1, sigma_new;
   PRIMME_NUM *y_new, *x0, *xvec, *yvec, *r, *aux;
   PRIMME_NUM *xyvec, *xx0, *xr, *xy_new;

   double lowerBound, upperBound;
   int j,k,j0,bs=*blockSize;
   double t0;

   getBoundsTuned(filter, primme, &lowerBound, &upperBound);
   if (lowerBound <= filter->minEig && upperBound >= filter->maxEig) {
      Apply_original(x, y, blockSize, filter, primme, stats);
      return;
   }

   if (lowpass) {  // low-pass filter
      a0 = lowerBound; a = upperBound; b = filter->maxEig;
   } else {
      a0 = upperBound; a = lowerBound; b = filter->minEig;
   }
   aux = (PRIMME_NUM *)primme_calloc(primme->nLocal*3*bs+4*bs, sizeof(PRIMME_NUM), "aux");
   x0 = aux;
   y_new = &aux[primme->nLocal*bs];
   r = &y_new[primme->nLocal*bs];
   xyvec = &r[primme->nLocal*bs];
   xx0 = &xyvec[bs];
   xr = &xx0[bs];
   xy_new = &xr[bs];

   e = (b - a)/2.; assert(e != 0.);
   c = (b + a)/2.;
   t0 = primme_get_wtime();
   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;
   sigma = e/(a0 - c);
   sigma1 = sigma;
   filter->matvec(xvec, yvec, blockSize, primme);
   if (stats) primme->stats.numMatvecs++;
   for (j=0, j0=primme->nLocal*bs; j<j0; ++j) x0[j] = 0.;
   for (j=0; j<bs; j++) xx0[j] = 1.;
   for (j=0, j0=primme->nLocal*bs; j<j0; ++j) yvec[j] = (yvec[j] - c*xvec[j])*sigma1/e;
   ortho2(yvec, xvec, xyvec, primme->nLocal, bs);
   for (j=0, j0=primme->nLocal*bs; j<j0; ++j) r[j] = yvec[j]/sigma1;
   for (j=0; j<bs; j++) xr[j] = xyvec[j]/sigma1;
   for (k=1; k<filter->degrees; ++k) {
      sigma_new = 1./(2./sigma1 - sigma);
      if (!isfinite(sigma_new/e) || !isfinite(sigma*sigma_new)) break;
      assert(isfinite(yvec[0]));
      filter->matvec(yvec, y_new, blockSize, primme);
      assert(isfinite(y_new[0]));
      if (stats) primme->stats.numMatvecs++;
      for (j=0, j0=primme->nLocal*bs; j<j0; ++j)
         y_new[j] = 2.*(y_new[j] - c*yvec[j])*sigma_new/e + 2*sigma_new*r[j]*xyvec[j] - sigma*sigma_new*x0[j];
      ortho2(y_new, xvec, xy_new, primme->nLocal, bs);
      for (j=0, j0=primme->nLocal*bs; j<j0; ++j) {
         x0[j] = yvec[j];
         yvec[j] = y_new[j];
      }
      for (j=0; j<bs; j++) {
         xy_new[j] += -sigma*sigma_new*xx0[j] + 2*sigma_new*xr[j]*xyvec[j];
         xx0[j] = xyvec[j];
         xyvec[j] = xy_new[j];
      }
      sigma = sigma_new;
   }
   if (stats) elapsedTimeFilterMV += primme_get_wtime() - t0;
   if (stats) numFilterApplies++;
   free(aux);
}  

/******************************************************************************
 * Use Chebyshev filter like in Zhou, "A Chebyshev-Davidson algorithm for large
 * symmetric eigenvalues", 2007.
 *
 * INPUT
 * -----
 *    m        #degrees of the expansion 
 *    a,b      [a,b] interval where function is ~0
 *    a0       min eig of A
 *    damping  0: no damping; 1: Jackson; 2: Lanczos sigma damping
 *
******************************************************************************/
static void Apply_filter_cheb_interior(void *x, void *y, int *blockSize, filter_params *filter,
                                 primme_params *primme, int stats) {
// TODO:
//   double a,b, a0;
//
//   double e, c, sigma, sigma1, sigma_new;
//   double *y_new, *x0, *xvec, *yvec, *aux, *x1;
//
//   double lowerBound, upperBound, target;
//   int i,j,k,j0;
//   CSRMatrix *matrix;
//   double t0;
//
//   matrix = (CSRMatrix *)primme->matrix;
//   getBoundsTuned(filter, primme, &lowerBound, &upperBound);
//   if (primme->numTargetShifts) target = primme->targetShifts[0];
//   else target = (driver.minEig + driver.maxEig)/2.0;
//
//   if (lowerBound <= driver.minEig && upperBound >= driver.maxEig) {
//      if (filter->prodIfFullRange) {
//         t0 = primme_get_wtime();
//         for (j=0; j<*blockSize; j++) {
//            amux_(&primme->n, &xvec[primme->nLocal*j], &yvec[primme->nLocal*j], matrix->AElts, matrix->JA, matrix->IA);
//            primme->stats.numMatvecs++;
//         }
//         elapsedTimeAMV += primme_get_wtime() - t0;
//      } else {
//         for(i=0; i<*blockSize*primme->nLocal; i++) yvec[i] = xvec[i]; 
//      }
//   }
//
//   a0 = (lowerBound - target)*(lowerBound - target);
//   a = (upperBound - target)*(upperBound - target);
//   b = (driver.maxEig - target)*(driver.maxEig - target);
//   aux = (double *)primme_calloc(primme->nLocal*3, sizeof(double), "aux");
//   x0 = aux;
//   x1 = &x0[primme->nLocal];
//   y_new = &x1[primme->nLocal];
//
//   e = (b - a)/2.;
//   c = (b + a)/2.;
//   t0 = primme_get_wtime();
//   for (i=0; i<*blockSize; i++) {
//      xvec = &((double *)x)[primme->nLocal*i];
//      yvec = &((double *)y)[primme->nLocal*i];
//      sigma = e/(a0 - c);
//      sigma1 = sigma;
//      // yvec <- (A - t)*(A - t)*xvec
//      amux_(&primme->n, xvec, x1, matrix->AElts, matrix->JA, matrix->IA);
//      for (j=0, j0=primme->nLocal; j<j0; ++j) x1[j] -= target*xvec[j];
//      amux_(&primme->n, x1, yvec, matrix->AElts, matrix->JA, matrix->IA);
//      for (j=0, j0=primme->nLocal; j<j0; ++j) yvec[j] -= target*x1[j];
//      primme->stats.numMatvecs+= 2;
//      for (j=0, j0=primme->nLocal; j<j0; ++j) x0[j] = xvec[j];
//      for (j=0, j0=primme->nLocal; j<j0; ++j) yvec[j] = (yvec[j] - c*xvec[j])*sigma1/e;
//      for (k=1; k<filter->degrees; ++k) {
//         sigma_new = 1./(2./sigma1 - sigma);
//         // y_new <- (A - t)*(A - t)*yvec
//         amux_(&primme->n, yvec, x1, matrix->AElts, matrix->JA, matrix->IA);
//         for (j=0, j0=primme->nLocal; j<j0; ++j) x1[j] -= target*yvec[j];
//         amux_(&primme->n, x1, y_new, matrix->AElts, matrix->JA, matrix->IA);
//         for (j=0, j0=primme->nLocal; j<j0; ++j) y_new[j] -= target*x1[j];
//         primme->stats.numMatvecs+= 2;
//         for (j=0, j0=primme->nLocal; j<j0; ++j)
//            y_new[j] = 2.*(y_new[j] - c*yvec[j])*sigma_new/e - sigma*sigma_new*x0[j];
//         for (j=0, j0=primme->nLocal; j<j0; ++j) {
//            x0[j] = yvec[j];
//            yvec[j] = y_new[j];
//         }
//         sigma = sigma_new;
//      }
//   }
//   elapsedTimeFilterMV += primme_get_wtime() - t0;
//   numFilterApplies++;
//   free(aux);
}  

void Apply_filter_modelcheb(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int stats) {

   int j;
   double lowerBound, upperBound;
   PRIMME_NUM *xvec, *yvec, *aux;

   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;

   getBoundsTuned(filter, primme, &lowerBound, &upperBound);
   aux = (PRIMME_NUM *)primme_calloc(primme->nLocal*3, sizeof(PRIMME_NUM), "aux");
   for (j=0; j<*blockSize; j++) {
      modelChebAv(primme, &xvec[primme->nLocal*j], filter->degrees, filter->minEig, filter->maxEig, lowerBound, upperBound, &yvec[primme->nLocal*j], aux, filter, stats);
   }
   free(aux);
}

/******************************************************************************
 * Computes p(A)*x being p a polynomial expressed Cheb. coefficients.
 *
 * INPUT
 * -----
 *    primme      primme_params
 *    xvec        input nLocal-vector
 *    m           #degrees of the expansion
 *    mu          (m+1)-vector with the expansion coefficients
 *    a,b         extreme values of the spectrum of A
 *    aux         auxiliary vector of 3*nLocal elements
 *
 * OUTPUT
 * ------
 *    yvec        output nLocal-vector
******************************************************************************/
void modelChebAv(primme_params *primme, PRIMME_NUM *xvec, int m, double a, double b, double lb, double ub, PRIMME_NUM *yvec, PRIMME_NUM *aux, filter_params *filter, int stats) {
   int k,k0, j,j0, one=1;
   double p, cx, c, w, s;
   PRIMME_NUM *vk, *vkm1, *vkp1, vkp1_j;
   double ymaxk, ymaxkm1, ymaxkaux, ymax;
   double t0;

   vk = aux;
   vkp1 = &aux[primme->nLocal];
   vkm1 = &aux[primme->nLocal*2];
   cx = (lb + ub)/2.;
   c = (a+b)/2.;
   w = (b-a)/2.;
   s = max((1.-(cx-c)/w)*(1.-(cx-c)/w), (-1.-(cx-c)/w)*(-1.-(cx-c)/w));
   p = (ub - cx)/w/sqrt(s);
   ymax = (1.+p*p)/(1.-p*p);

   for (k=0, k0=primme->nLocal; k<k0; ++k) {
      yvec[k] = xvec[k];
      vkm1[k] = 0.;
   }
   ymaxk = 1;
   ymaxkm1 = 0;
   t0 = primme_get_wtime();
   for(k=0; k<m; k+=2) {
      /* vkp1 <= scal*[(A-cx)*(A-cx) - c]*yvec - vkm1 */
      filter->matvec(yvec, vk, &one, primme);
      for (j=0, j0=primme->nLocal; j<j0; ++j) vk[j] = (vk[j] - cx*yvec[j])/w;
      filter->matvec(vk, vkp1, &one, primme);
      primme->stats.numMatvecs += 2;
      for (j=0, j0=primme->nLocal; j<j0; ++j) {
         vkp1_j = (k==0?1.:2.)*((1. + p*p)*yvec[j] - 2.*(vkp1[j] - cx*vk[j])/w/s)/(1.-p*p) - vkm1[j];
         vkm1[j] = yvec[j];
         yvec[j] = vkp1_j;
      }
      ymaxkaux = (k==0?1.:2.)*ymax*ymaxk - ymaxkm1;
      ymaxkm1 = ymaxk;
      ymaxk = ymaxkaux;
   } 
   for (j=0, j0=primme->nLocal; j<j0; ++j) yvec[j] /= ymaxk;
   if (stats) elapsedTimeFilterMV += primme_get_wtime() - t0;
   if (stats) numFilterApplies++;
}





#ifdef USE_FILTLAN
/******************************************************************************
 * Use filtered conjugate residual polynomials algorithm implemented in
 * FILTLAN.
 *
 * INPUT
 * -----
 *    m        #degrees of the expansion 
 *    a,b      [a,b] interval where function is ~0
 *    a0       min eig of A
 *    damping  0: no damping; 1: Jackson; 2: Lanczos sigma damping
 *
 * OUTPUT
 * ------
 *    mu,      (m+1)-vector with the expansion coefficients
******************************************************************************/
#include "filtlan.h"
static void primme_spmv(double *xvec, double *yvec, void *ctxt);
typedef struct primme_spmv_params { primme_params *primme; filter_params *filter; int stats; } primme_spmv_params;
static void Apply_filter_filtlan(void *x, void *y, int *blockSize, filter_params *filter,
                            primme_params *primme, int stats) {
   double *xvec, *yvec;
   double lowerBound, upperBound, frame[4], intervalWeights[5] = {100, 1, 1, 1, 100};
   int i;
   primme_spmv_params ctxt = {primme, filter, stats};

   getBoundsTuned(filter, primme, &lowerBound, &upperBound);
   frame[0] = filter->minEig; frame[3] = filter->maxEig;
   frame[1] = lowerBound; frame[2] = upperBound;
   
   for (i=0; i<*blockSize; i++) {
      xvec = &((double *)x)[primme->nLocal*i];
      yvec = &((double *)y)[primme->nLocal*i];
      FilteredCRMatrixPolynomialVectorProduct_c(frame, 10, filter->degrees, intervalWeights, 1e-10, primme_spmv, &ctxt, primme->n, xvec, yvec);
   }
}

static void primme_spmv(double *xvec, double *yvec, void *ctxt) {
   primme_params *primme = ((primme_spmv_params*)ctxt)->primme;
   filter_params *filter = ((primme_spmv_params*)ctxt)->filter;
   int stats = ((primme_spmv_params*)ctxt)->stats;
   int blockSize = 1;
   double t0;
   
   t0 = primme_get_wtime();
   filter->matvec(xvec, yvec, &blockSize, primme);
   if (stats) elapsedTimeFilterMV += primme_get_wtime() - t0;
   if (stats) numFilterApplies++;
   if (stats) primme->stats.numMatvecs++;
} 
#endif /* FILTLAN */

#ifdef USE_FEAST
/******************************************************************************
 * Use FEAST single iteration as a filter
 *
******************************************************************************/
extern void zfeast_hrci_(int*,int*,PRIMME_NUM*,PRIMME_NUM*,PRIMME_NUM*,PRIMME_NUM*,PRIMME_NUM*,int*,double*,int*,double*,double*,int*,double*,PRIMME_NUM*,int*,double*,int*);
extern void feastinit_(int*);

static void Apply_filter_feast(void *x, void *y, int *blockSize, filter_params *filter,
                               primme_params *primme, int stats) {

   double complex *xvec=(double complex*)x, *yvec=(double complex*)y;
   double *oldShiftsForPreconditioner;
   double lowerBound, upperBound;
   int i;
   int  fpm[64],ijob,info,ncv,loop,nconv;
   double epsout;
   complex double Ze,shifts,*work1,*work2,*aux;
   double t0;

   double complex *Aq, *Bq;
   double *rnorms, *evals;
   
   if (!blockSize) return;
   getBoundsTuned(filter, primme, &lowerBound, &upperBound);
   ncv = *blockSize;
   work1 = (PRIMME_NUM *)primme_calloc(primme->n*ncv, sizeof(PRIMME_NUM), "work1");
   work2 = (PRIMME_NUM *)primme_calloc(primme->n*ncv, sizeof(PRIMME_NUM), "work2");
   aux = (PRIMME_NUM *)primme_calloc(primme->n*ncv, sizeof(PRIMME_NUM), "aux");

#ifdef FEAST_SOLVER
   Aq =       (PRIMME_NUM *)primme_calloc(ncv*ncv, sizeof(PRIMME_NUM), "work2");
   Bq =       (PRIMME_NUM *)primme_calloc(ncv*ncv, sizeof(PRIMME_NUM), "work2");
   evals =    (double *)primme_calloc(ncv, sizeof(double), "work2");
   rnorms =   (double *)primme_calloc(ncv, sizeof(double), "work2");
#else
   Aq =       NULL;
   Bq =       NULL;
   evals =    NULL;
   rnorms =   NULL;
#endif

   /* parameters */
   feastinit_(fpm);
   fpm[0] = (primme->printLevel>3)? 1: 0;                      /* runtime comments */
   fpm[7] = filter->degrees;                                   /* contour points */
#ifndef FEAST_SOLVER
   fpm[13] = 1;                     /* single iteration */
   fpm[4] = 1;                      /* provide initial guess */
#endif

   t0 = primme_get_wtime();
   for (i=0;i<ncv*primme->n;i++) yvec[i] = xvec[i];
   ijob = -1;           /* first call to reverse communication interface */
   do {
      zfeast_hrci_(&ijob, &primme->n, &Ze, work1, work2, Aq, Bq, fpm, &epsout, &loop, &lowerBound, &upperBound, &ncv, evals, y, &nconv, rnorms, &info);
      if (ijob == 10) {
         /* set new quadrature point */
         shifts = Ze;
         primme->rebuildPreconditioner = 1;
      } else if (ijob == 20) {
         shifts = conj(Ze);
         primme->rebuildPreconditioner = 1;
      } else if (ijob == 11 || ijob == 21) {
         /* linear solve (A-sigma*B)\work2, overwrite work2 */
         oldShiftsForPreconditioner = primme->ShiftsForPreconditioner;
         primme->ShiftsForPreconditioner = (double*)&shifts;
         primme->numberOfShiftsForPreconditioner = -1; /* one complex shift */
         filter->precond(work2, aux, &fpm[22], primme);
         for (i=0;i<fpm[22]*primme->n;i++) work2[i] = -aux[i];
         primme->ShiftsForPreconditioner = oldShiftsForPreconditioner;
         if (stats) primme->stats.numPreconds += fpm[22];
      } else if (ijob == 30) {
         filter->matvec(&yvec[(fpm[23]-1)*primme->n], &work1[(fpm[23]-1)*primme->n], &fpm[24], primme);
         if (stats) primme->stats.numMatvecs += fpm[24];
      } else if (ijob == 40) {
         for (i=0;i<fpm[24]*primme->n;i++) work1[(fpm[23]-1)*primme->n+i] = yvec[(fpm[23]-1)*primme->n+i];
      } else if (ijob != -2 && ijob != 0) {
         fprintf(stderr, "Internal error in FEAST reverse communication interface (ijob=%d)",ijob);
         return;
      }
   } while (ijob != 0 && info == 0);
   if (stats) elapsedTimeFilterMV += primme_get_wtime() - t0;
   assert(info == 4 || info == 0 || info == 3);
   free(work1);
   free(work2);
   free(aux);

#ifdef FEAST_SOLVER
   free(Aq);
   free(Bq);
   free(rnorms);
   free(evals);
#endif

   if (stats) numFilterApplies += *blockSize;
}
#endif /* USE_FEAST */


static void Apply_filter_augmented(void *x, void *y, int *blockSize, filter_params *filter,
                                   primme_params *primme, int stats) {

   int j,nLocal=primme->nLocal,n=primme->n,one=1;
   PRIMME_NUM *xvec, *yvec;
   driver_params *driver = (driver_params*)primme;
   double maxSval;

   /* Setup */
   if (!blockSize && !x && !y) {
      primme->nLocal *= 2;
      primme->n *= 2;
      filter->lowerBoundFix = driver->minEig;
      filter->upperBoundFix = driver->maxEig;
      maxSval = max(fabs(driver->maxEig), fabs(driver->minEig));
      driver->minEig = -maxSval;
      driver->maxEig = maxSval;
      return;
   }
   /* Un-setup */
   if (!blockSize && x && !y) {
      driver->minEig = filter->lowerBoundFix;
      driver->maxEig = filter->upperBoundFix;
      primme->nLocal /= 2;
      primme->n /= 2;
      return;
   }

   xvec = (PRIMME_NUM *)x;
   yvec = (PRIMME_NUM *)y;

   primme->nLocal = nLocal/2;
   primme->n = n/2;
   for (j=0; j<*blockSize; j++) {
      filter->matvec(&xvec[j*nLocal+nLocal/2], &yvec[j*nLocal], &one, primme);
      filter->matvec(&xvec[j*nLocal], &yvec[j*nLocal+nLocal/2], &one, primme);
   }
   primme->nLocal = nLocal;
   primme->n = n;

   if (stats) primme->stats.numMatvecs -= *blockSize; /* Removed the assumed blockSize MV prods */
   if (stats) primme->stats.numMatvecs += *blockSize * 2;
}

static void Apply_filter_normalequation(void *x, void *y, int *blockSize, filter_params *filter,
                                        primme_params *primme, int stats) {

   PRIMME_NUM *aux;
   driver_params *driver = (driver_params*)primme;
   int i;
   const double shift = (filter->lowerBoundFix + filter->upperBoundFix)/2.;

   /* Setup */
   if (!blockSize && !x && !y) {
      filter->matvec(x, y, blockSize, primme);
      filter->lowerBoundTuned = driver->minEig;
      filter->upperBoundTuned = driver->maxEig;
      driver->maxEig = max((filter->lowerBoundTuned-shift)*(filter->lowerBoundTuned-shift), (filter->upperBoundTuned-shift)*(filter->upperBoundTuned-shift));
      if (filter->lowerBoundTuned <= shift && filter->upperBoundTuned >= shift) driver->minEig = 0;
      else driver->minEig = min((filter->lowerBoundTuned-shift)*(filter->lowerBoundTuned-shift), (filter->upperBoundTuned-shift)*(filter->upperBoundTuned-shift));
      return;
   }
   /* Un-setup */
   if (!blockSize && x && !y) {
      driver->minEig = filter->lowerBoundTuned;
      driver->maxEig = filter->upperBoundTuned;
      filter->matvec(x, y, blockSize, primme);
      return;
   }

   aux = (PRIMME_NUM *)primme_calloc(*blockSize*primme->nLocal, sizeof(PRIMME_NUM), "aux");
   filter->matvec(x, aux, blockSize, primme);
   if (shift) for (i=0; i<primme->nLocal; i++) aux[i] -= shift*((PRIMME_NUM*)x)[i];
   filter->matvec(aux, y, blockSize, primme);
   if (shift) for (i=0; i<primme->nLocal; i++) ((PRIMME_NUM*)y)[i] -= shift*aux[i];
   free(aux);

   if (stats) primme->stats.numMatvecs -= *blockSize; /* Removed the assumed blockSize MV prods */
   if (stats) primme->stats.numMatvecs += *blockSize * 2;
}


static void Apply_filter_gamma5(void *x, void *y, int *blockSize, filter_params *filter,
                                        primme_params *primme, int stats) {

   driver_params *driver = (driver_params*)primme;
   int i,bs;
   const int nLocal = primme->nLocal;
   const double shift = filter->lowerBoundFix, mult = filter->upperBoundFix;

   if (!blockSize) {
      return;
   }

   assert(driver->Aperm || primme->nLocal == primme->n);
   filter->matvec(x, y, blockSize, primme);
   bs = *blockSize;
   for (i=0; i<bs*nLocal; i++)
      ((PRIMME_NUM*)y)[i] = (shift*((PRIMME_NUM*)x)[i] + mult*((PRIMME_NUM*)y)[i])
         *((driver->Aperm ? driver->Aperm[i%nLocal] : i%nLocal)%12 > 5 ? -1 : 1);

}

static void Apply_filter_gamma5sq(void *x, void *y, int *blockSize, filter_params *filter,
                                        primme_params *primme, int stats) {

   PRIMME_NUM *aux;

   if (!blockSize) {
      return;
   }

   aux = (PRIMME_NUM *)primme_calloc((*blockSize)*primme->nLocal, sizeof(PRIMME_NUM), "aux");
   Apply_filter_gamma5(x, aux, blockSize, filter, primme, stats);
   Apply_filter_gamma5(aux, y, blockSize, filter, primme, stats);
   free(aux);

   if (stats) primme->stats.numMatvecs -= (*blockSize); /* Removed the assumed blockSize MV prods */
   if (stats) primme->stats.numMatvecs += (*blockSize) * 2;

}


void plot_filter(int n, filter_params *filter, primme_params *primme, FILE *out) {
   int i;
   double *xy, lb, ub, chk;
   static double lst_chk=0;
   
   getBoundsTuned(filter, primme, &lb, &ub);
   chk = lb*ub*filter->degrees*filter->minEig*filter->maxEig;
   if (chk == lst_chk) return;
   lst_chk = chk;
   xy = (double *)primme_calloc(n*2, sizeof(double), "xy");
   for (i=0; i < n; i++) {
      xy[i] = (filter->maxEig - filter->minEig)/(double)(n-1)*(double)i + filter->minEig;
   }
   eval_filter(xy, &xy[n], n, filter, primme);
   fprintf(out, "FILTER === %d %d %d %d %g %g\n", filter->filter, filter->degrees, filter->lowerBound, filter->upperBound, lb, ub);
   for (i=0; i<n; i++) fprintf(out, "FILTER %g %g\n", xy[i], xy[n+i]);
   fprintf(out, "FILTER ...\n");
   free(xy);
}

static void eval_filter(double *x, double *y, int n, filter_params *filter, primme_params *primme) {
   primme_params primme0;
   filter_params filter0 = *filter;
   CSRMatrix matrix;
   int i, ONE=1;
   PRIMME_NUM *v, *y0;
   
   #ifndef USE_DOUBLECOMPLEX 
   matrix.AElts = x;
   y0 = y;
   #else
   matrix.AElts = (PRIMME_NUM*)primme_calloc(n, sizeof(PRIMME_NUM), "matrix.AElts");
   y0 = (PRIMME_NUM*)primme_calloc(n, sizeof(PRIMME_NUM), "y0");
   for (i=0; i < n; i++) matrix.AElts[i] = x[i];
   #endif
   matrix.IA = (int *)primme_calloc(n+1, sizeof(int), "matrix.IA");
   matrix.JA = (int *)primme_calloc(n, sizeof(int),"matrix.JA");
   for (i=0; i < n; i++) {
      matrix.IA[i] = i+1;
      matrix.JA[i] = i+1;
   }
   matrix.IA[n] = n+1;
   matrix.n = matrix.m = matrix.nnz = n;
   bzero(&primme0, sizeof(primme0));
   primme0.matrix = &matrix;
   primme0.n = primme0.nLocal = n;
   for (i=0; i<7; i++)
      primme0.RitzValuesForPreconditioner[i] = primme->RitzValuesForPreconditioner[i];
   primme0.printLevel = 0;
   filter0.matvec = CSRMatrixMatvec;
   filter0.precond = ApplyInvDiagPrecNative;
   v = (PRIMME_NUM *)primme_calloc(n, sizeof(PRIMME_NUM), "v");
   for (i=0; i < n; i++) v[i] = 1./n;
   Apply_filter(v, y0, &ONE, &filter0, &primme0, 0);
   free(v);
   free(matrix.JA);
   free(matrix.IA);
   #ifdef USE_DOUBLECOMPLEX 
   free(matrix.AElts);
   for (i=0; i < n; i++) y[i] = REAL_PART(y0[i]);
   free(y0);
   #endif
   for (i=0; i < n; i++) y[i] *= n;
}

static void ortho2(PRIMME_NUM *y, const PRIMME_NUM *x, PRIMME_NUM *o, int n, int cols) {
   PRIMME_NUM a;
   int i, j, k;

   for (k=0; k<cols; k++) {
      o[k] = 0.;
      for (i=0; i<2; i++) {
         o[k] += (a = COMPLEXV(SUF(Num_dot)(n, COMPLEXZ(y), 1, COMPLEXZ((PRIMME_NUM*)&x[k*n]), 1)));
         for (j=0; j<n; j++) y[k*n + j] -= x[k*n + j]*a;
      }
   }
}

/* Shift slightly the lower and upper bounds of the filter such as the value of the filter on both
   points is almost the same.
*/
static int level_filter_bounds(filter_params *filter, primme_params *primme) {
   double x[2], y[2], z0=0, z1, *z, zb;
   int i;

   getBounds(filter, primme, &filter->lowerBoundTuned, &filter->upperBoundTuned);
   x[0] = filter->lowerBoundTuned;
   x[1] = filter->upperBoundTuned;
   z = &filter->upperBoundTuned;
   zb = filter->lowerBoundTuned;
   eval_filter(x, y, 2, filter, primme);
   if (y[1] > y[0]) {
      x[0] = x[1];
      x[1] = filter->lowerBoundTuned;
      z = &filter->lowerBoundTuned;
      zb = filter->upperBoundTuned;
   }
   for (i=0; i<1000 && *z >= filter->minEig && *z <= filter->maxEig; i++) {
      z0 = *z;
      *z = zb + (*z-zb)*1.01;
      eval_filter(x, y, 2, filter, primme);
      if (y[1] >= y[0]) break;
   }
   *z = max(filter->minEig, min(filter->maxEig, *z)); 
   if (y[1] < y[0]) return 0;
   z1 = *z;
   for (i=0; i<100 && fabs(y[1]-y[0])/fabs(y[1]+y[0])*2>.1; i++) {
      *z = (z1 + z0)/2.;
      eval_filter(x, y, 2, filter, primme);
      if (y[1] >= y[0]) z1 = *z;
      else z0 = *z;
   }
   return 1;
}

/* Check that filter(x) >= fb iff x in [lb, ub], where fb=max(filter(lb), filter(ub)) */
/* Check that max_{x in [lb, ub]} filter(x)*(x - l) >= max_{x not in [lb, ub]} filter(x)*(x - l) where
     l = argmax_{x in [lb, ub]} filter(x)
*/
static double check_filter_split(double *x, double *y, int n, filter_params *filter, primme_params *primme, int setBounds) {
   int i, iMaxIn, offended=0;
   double fb, maxIn, maxInD, maxOutD, lb, ub;

   getBounds(filter, primme, &lb, &ub);
   assert(n > 2 && x[0] == lb && x[1] == ub);
   eval_filter(x, y, n, filter, primme);
   fb = min(y[0], y[1]);
   maxIn = -HUGE_VAL;
   for (i=0; i<n; i++) {
      if (y[i] >= fb) {
         if (x[i] < lb || x[i] > ub) {
            if (y[i] <= max(y[0], y[1])) offended++;
            else fb = y[i]*1.01;
         }
         else if (maxIn < fabs(y[i])) maxIn = fabs(y[i]), iMaxIn = i;
      }
   }
   if (setBounds) {
      primme->numBounds = 1;
      primme->bounds[0] = fb;
      primme->aNorm = maxIn;
   }
   if (fb > y[iMaxIn] || offended > 10*n/100) return -HUGE_VAL;
   maxInD = maxOutD = -HUGE_VAL;
   for (i=0; i<n; i++) {
      if (y[i] >= fb) {
         maxInD = max(maxInD, fabs(y[i]*(x[i] - x[iMaxIn])));
      } else {
         maxOutD = max(maxOutD, fabs(y[i]*(x[i] - x[iMaxIn])));
      }
   }
   return maxInD/maxOutD;
}

int tune_filter(filter_params *filter, primme_params *primme, int onlyIfStatic, int setBounds) {
   double chk, lb, ub;

   if (!onlyIfStatic || (filter->lowerBound == 8 && filter->upperBound == 8)) {
      getBounds(filter, primme, &lb, &ub);
      chk = lb+ub+filter->degrees+filter->minEig+filter->maxEig;
      if (filter->lastCheckCS != chk) {
         if (filter->checkEps != 0) check_filter(filter->checkEps, filter->degrees, 500, setBounds, filter, primme);
         else check_filter(HUGE_VAL, filter->degrees, filter->degrees, setBounds, filter, primme);
         filter->lastCheckCS = chk;
         return 1;
      }
   }
   return 0;
}

static int check_filter(double a, int d0, int d1, int setBounds, filter_params *filter, primme_params *primme) {
   int i, d, j, b=1;
   const int n = d1*100;
   double *xy, lb, ub, q;

   xy = (double *)primme_calloc(n*2+4, sizeof(double), "xy");
   getBounds(filter, primme, &lb, &ub);
   xy[0] = lb;
   xy[1] = ub;
   for (i=0, j=2; i < n/2; i++, j++) {
      xy[j] = (filter->maxEig - filter->minEig - ub + lb)/(double)(n/2+1)*(double)(i+1) + filter->minEig;
      if (xy[j] >= lb) break;
   }
   for (; i < n/2; i++, j++) {
      xy[j] = (filter->maxEig - filter->minEig - ub + lb)/(double)(n/2+1)*(double)(i+1) + filter->minEig + ub - lb;
      assert(xy[j] <= filter->maxEig && xy[j] >= ub);
   }
   for (i=0; i < (n+1)/2; i++) {
      xy[j++] = (ub - lb)/(double)((n+1)/2+1)*(double)(i+1) + lb;
   }
   assert(j == n+2);
   for (d=d0, b=1; d<d1; d0=d, b*=2, d+=b) {
      filter->degrees = d;
      if (!level_filter_bounds(filter, primme)) continue;
      if ((q = check_filter_split(xy, &xy[n+2], n+2, filter, primme, setBounds)) >= a) break;
   }
   if (d >= d1) {
      filter->degrees = d = d1;
      level_filter_bounds(filter, primme);
      if ((q = check_filter_split(xy, &xy[n+2], n+2, filter, primme, setBounds)) < a) {
         if (primme->printLevel >= 4)
            printf("check_filter: failed! degrees %d scale: %g\n", filter->degrees, q);
         free(xy);
         return 0;
      }
   }
   d1 = d;
   while (d0+1 < d1) {
      filter->degrees = d = (d0 + d1 + 1)/2;
      if (level_filter_bounds(filter, primme) && (q = check_filter_split(xy, &xy[n+2], n+2, filter, primme, setBounds)) >= a) d1 = d;
      else d0 = d;
   }
   if (d != d1) {
      filter->degrees = d1;
      level_filter_bounds(filter, primme);
      assert((q = check_filter_split(xy, &xy[n+2], n+2, filter, primme, setBounds)) >= a);
   }
   if (primme->printLevel >= 4)
      printf("check_filter: degrees %d scale: %g\n", filter->degrees, q);
   free(xy);
   return 1;
}
