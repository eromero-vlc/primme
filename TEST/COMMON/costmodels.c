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

#include "primme.h"
#include "filters.h"
#include "shared_utils.h"
#include "wtime.h"
#include "costmodels.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "num.h"
#ifdef USE_MPI
#  include <mpi.h>
#endif

int model_conv_one(int d0, int its0, double res0, double tol, int n, double (*costModel)(int,int,int,int,void*), void *ctx);

#define PURY(X) ((X)>0 && 1.0/(X)>0 ? (X) : 1)

int reconsider_degree_filter(filter_params *filter, primme_params *primme, double *work, int *iwork,
                              double (*costModel)(int,int,int,int,void*)) {
   int prevDegree = filter->degrees, newDegrees, reset=0;
   int 
      prevEigFirstIt = iwork[1],    /* Iteration index the first time the current eig was selected */
      prevIt = iwork[2],            /* Last iteration index the heuristic was done */
      decIts = iwork[3];            /* Min iterations to consider a change in the residual */
   double prevTime = work[0],       /* Last time the heuristic was done */
      prevResidual = work[1],       /* Residual norm the last time the heuristic was done */
      eigFirstRes = work[2],        /* Residual norm the first time an eig was selected */
      prevEigRes = work[3];         /* Residual norm in the previous iteration */
   const double currTime = primme->stats.elapsedTimeMatvec + primme->stats.elapsedTimeOrtho + primme->stats.elapsedTimePrecond + primme->stats.elapsedTimeSolveH; //primme_wTimer(0);
   const double currRes = PURY(primme->stats.currentEigResidual), logCurrRes = log(currRes);

   if (primme->procID == 0) {
      if (prevDegree < 0 || prevIt > primme->stats.numOuterIterations) {
         /* First time this is called */
         prevDegree = max(2, prevDegree); prevEigFirstIt = -1, prevIt = 0, decIts =  max(primme->minRestartSize, 10);
         prevTime = currTime, prevResidual = 0, eigFirstRes = currRes, prevEigRes = 1;
         reset = 1;
      }
      
      if (prevEigFirstIt != primme->stats.currentEigFirstIteration) {
         /* Different eigenvalue has been selected */
         if (prevResidual != 0)
            prevResidual = prevResidual + log(1./prevEigRes*currRes);
         else
            prevResidual = logCurrRes;
         //assert(prevResidual > 0 && 1/prevResidual > 0);
         prevEigFirstIt = primme->stats.currentEigFirstIteration;
         newDegrees = max(2, prevDegree);
         eigFirstRes = sqrt(eigFirstRes*primme->stats.currentEigFirstResidual);
         //decIts = max(100, (int)((primme->stats.currentEigFirstIteration - prevEigFirstIt)/log10(prevEigFirstRes/prevEigRes)));
         //reset = 1;
      }
      //assert(prevResidual > 0 && 1/prevResidual > 0);
      if ((currTime - prevTime > 1 || fabs(logCurrRes-prevResidual) > 200) && currRes == primme->stats.currentEigResidual && eigFirstRes != 1 && primme->stats.numRestarts > 1 && prevResidual != 0 && (logCurrRes <= prevResidual-log(100.0) || prevIt + decIts < primme->stats.numOuterIterations)) {
         /* Enough time/iterations/residual has passed */
         double diffRes = logCurrRes-prevResidual;
         if (diffRes >= 0) {
            /* If residual has not been reduced, increase degrees */
            newDegrees = 2*prevDegree;
            newDegrees = prevDegree;
            if (primme->printLevel > 3) printf("FILTERD: %d %d %g %g\n", primme->stats.numOuterIterations, newDegrees, logCurrRes, prevResidual);
            reset = 1;
         } else {
            /* Use cost model to set a new degree */
            assert(prevDegree > 1);
            assert(primme->stats.numOuterIterations-prevIt > 1);
            assert(currRes > 0);
            newDegrees = model_conv_one(prevDegree, primme->stats.numOuterIterations-prevIt,
                           diffRes,
                           min(primme->eps*primme->aNorm/eigFirstRes, 1),
                           primme->numEvals-primme->initSize, costModel, primme);
            if (newDegrees <= 1) newDegrees = prevDegree;
            if (newDegrees != prevDegree) reset = 1;
            if (primme->printLevel > 3) printf("FILTERD: %d %d\n", primme->stats.numOuterIterations, newDegrees);
         }
      } else {
         newDegrees = max(2, prevDegree);
      }
      if (reset || fabs(logCurrRes-prevResidual) > 200 /* avoid overflow in model_conv */) {
         prevEigFirstIt = primme->stats.currentEigFirstIteration;
         prevTime = currTime;
         prevIt = primme->stats.numOuterIterations;
         prevResidual = logCurrRes;
         //assert(prevResidual > 0 && 1/prevResidual > 0);
      }
      if (currRes == primme->stats.currentEigResidual) {
         prevEigRes = currRes;
         if (prevResidual == 0) prevResidual = logCurrRes;
      }

      iwork[1] = prevEigFirstIt; iwork[2] = prevIt; iwork[3] = decIts;
      work[0] = prevTime; work[1] = prevResidual; work[2] = eigFirstRes; work[3] = prevEigRes;
   }
   if (primme->numProcs > 1) {
      double nd0 = primme->procID == 0 ? newDegrees : 0, nd1;
      int one = 1;
      primme->globalSumDouble(&nd0, &nd1, &one, primme);
      newDegrees = (int)nd1;
   }
   filter->degrees = newDegrees;
   return prevDegree != newDegrees;
}

int model_conv_one(int d0, int its0, double logRes0, double tol, int n, double (*costModel)(int,int,int,int,void*), void *ctx) {
   double minModel=INFINITY, t, currModel=-1;
   int j,degree,minDegree=d0,its;
   int itsW=0;

   assert(d0 > 0 && its0 > 0 && tol > 0 && logRes0 <= 0 && tol <= 1 && n > 0);
   //printf("GAMMA: g: %e res0: %e its: %d dg: %d\n", pow((cosh(acosh(1./res0)/its0)+1.)/2., 1./d0), res0, its0, d0);
   //printf("MODEL ===\n");
   for (j=0, degree=2; j<200 && degree<2000; j++, degree++) {
      its = acosh(1./tol)/acosh(2.*pow((cosh(acosh(exp(-logRes0))/its0)+1.)/2., degree/(double)d0)-1.);
      if (its <= 0) continue;
      t = costModel(degree, its, degree != d0, n, ctx);
      if (degree == d0) currModel = t;
      //printf("MODEL: %d %d %e\n", degree, its, t);
      if (t < minModel) { minModel = t; minDegree = degree; j = 0; itsW=its; }
   }
   if (currModel < 0) {
      its = acosh(1./tol)/acosh(2.*pow((cosh(acosh(exp(-logRes0))/its0)+1.)/2., 1.0)-1.);
      if (its <= 0) currModel = minModel;
      else currModel = costModel(d0, its, 0, n, ctx);
   }
   printf("MODEL ... nD:%d nIts:%d s:%e d0:%d its0:%d tol:%e res0:%e n:%d\n", minDegree,itsW,currModel/minModel, d0, its0, tol, logRes0, n);
   return currModel/minModel >= 1.25 ? minDegree : d0;
}


double timeCostModelA(int degrees, int its, int init, int n, void *ctx) {
   primme_params *primme = (primme_params*)ctx;
   double mv = (primme->stats.elapsedTimeMatvec + primme->stats.elapsedTimePrecond)/(primme->stats.numMatvecs + primme->stats.numPreconds),
          ortho = primme->stats.elapsedTimeOrtho/primme->stats.numColumnsOrtho,
          avgCols = primme->initSize + primme->numOrthoConst + (primme->maxBasisSize + primme->minRestartSize + primme->restartingParams.maxPrevRetain)/2.,
          solveH = primme->stats.elapsedTimeSolveH/primme->stats.numOuterIterations,
          rest = (primme_wTimer(0) - primme->stats.elapsedTimeMatvec - primme->stats.elapsedTimeOrtho - primme->stats.elapsedTimePrecond - primme->stats.elapsedTimeSolveH)/primme->stats.numOuterIterations;
   int k = primme->minRestartSize+primme->initSize;

   assert(rest > 0);
   return degrees*mv*its*n + ortho*(avgCols+(n-1)/2.)*its*n + rest*its*n + solveH*its*n
      + (init ? (degrees*mv*k + ortho*(k-1)*k/2 + solveH + rest) : 0);
}

double timeCostModelPrecond(int degrees, int its, int init, int n, void *ctx) {
   primme_params *primme = (primme_params*)ctx;
   driver_params *driver = (driver_params*)ctx;
   double mv = (primme->stats.elapsedTimeMatvec + primme->stats.elapsedTimePrecond)/(primme->stats.numMatvecs + primme->stats.numPreconds),
          ortho = primme->stats.elapsedTimeOrtho/primme->stats.numColumnsOrtho,
          avgCols = primme->initSize + primme->numOrthoConst + (primme->maxBasisSize + primme->minRestartSize + primme->restartingParams.maxPrevRetain)/2.,
          solveH = primme->stats.elapsedTimeSolveH/primme->stats.numOuterIterations,
          rest = (primme_wTimer(0) - primme->stats.elapsedTimeMatvec - primme->stats.elapsedTimeOrtho - primme->stats.elapsedTimePrecond - primme->stats.elapsedTimeSolveH)/primme->stats.numOuterIterations;

   assert(rest > 0);
   return (degrees+max(1,driver->AFilter.degrees))*mv*its*n + ortho*(avgCols+(n-1)/2.)*its*n + rest*its*n + solveH*its*n;
}
