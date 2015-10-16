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
 * File: driver.c
 *
 * Purpose - driver that can read a matrix from a file and compute some
 *           eigenvalues using PRIMME.
 *
 *  Parallel driver for PRIMME. Calling format:
 *
 *             primme DriverConfigFileName SolverConfigFileName
 *
 *  DriverConfigFileName  includes the path and filename of the matrix
 *                            as well as preconditioning information (eg., 
 *                            ParaSails parameters).
 *                            Currently, for reading the input matrix,
 *                            full coordinate format (.mtx) and upper triangular 
 *                            coordinate format (.U) are supported.
 *
 *         Example file:  DriverConf
 *
 *  SolverConfigFileName  includes all d/zprimme required information
 *                            as stored in primme_params data structure.
 *
 *             Example files: FullConf  Full customization of PRIMME
 *                            LeanConf  Use a preset method and some customization
 *                            MinConf   Provide ONLY a preset method and numEvals.
 *
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>

#ifdef USE_MPI
#  include <mpi.h>
#endif
#ifdef USE_NATIVE
#  include "native.h"
#endif
#ifdef USE_PARASAILS
#  include "parasailsw.h"
#endif
#ifdef USE_PETSC
# include "petscw.h"
#endif

/* primme.h header file is required to run primme */
#include "primme.h"
#include "shared_utils.h"
/* wtime.h header file is included so primme's timimg functions can be used */
#include "wtime.h"

#include "filters.h"
#include "costmodels.h"

#define ASSERT_MSG(COND, RETURN, ...) { if (!(COND)) {fprintf(stderr, "Error in " __FUNCT__ ": " __VA_ARGS__); return (RETURN);} }

#define ADD_STATS(A,B) {\
   (A).numOuterIterations += (B).numOuterIterations; \
   (A).numRestarts        += (B).numRestarts; \
   (A).numMatvecs         += (B).numMatvecs; \
   (A).elapsedTimeMatvec  += (B).elapsedTimeMatvec; \
   (A).numPreconds        += (B).numPreconds; \
   (A).elapsedTimePrecond += (B).elapsedTimePrecond; \
   (A).elapsedTime        += (B).elapsedTime; \
   (A).numReorthos        += (B).numReorthos ; \
   (A).elapsedTimeOrtho   += (B).elapsedTimeOrtho;\
   (A).elapsedTimeSolveH  += (B).elapsedTimeSolveH;}

static int real_main (int argc, char *argv[]);
static int setMatrixAndPrecond(driver_params *driver, primme_params *primme, int **permutation);
#ifdef USE_MPI
static void broadCast(primme_params *primme, primme_preset_method *method, 
   driver_params *driver, int master, MPI_Comm comm);
static void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, 
                         primme_params *primme);
#endif
static void setFilters(driver_params *driver, primme_params *primme);
static void unsetFilters(double *evals, PRIMME_NUM *evecs, double *rnorms, primme_params *primme);
static int check_solution(const char *checkXFileName, primme_params *primme, double *evals,
                          PRIMME_NUM *evecs, double *rnorms, int *perm);
static int destroyMatrixAndPrecond(driver_params *driver, primme_params *primme, int *permutation);
static int writeBinaryEvecsAndPrimmeParams(const char *fileName, PRIMME_NUM *X, int *perm,
                                           primme_params *primme);
static int readBinaryEvecsAndPrimmeParams(const char *fileName, PRIMME_NUM *X, PRIMME_NUM **Xout,
                                          int n, int Xcols, int *Xcolsout, int nLocal,
                                          int *perm, primme_params *primme);
static int primmew(double *evals, PRIMME_NUM *evecs, double *rnorms, primme_params *primme);
static int spectrum_slicing(double *evals, PRIMME_NUM *evecs, double *rnorms, primme_params *primme,
                            int maxNumEvals, int *perm);
static int slice_tree(double *lb, double *ub, double lastMinEig, double lastMaxEig, double *queue, int lqueue, int *qtop);
static void Apply_transform_filter(void *x, void *y, int *blockSize, primme_params *primme);


int main (int argc, char *argv[]) {
   int ret;
#if defined(USE_PETSC)
   PetscInt ierr;
#endif

#if defined(USE_MPI) && !defined(USE_PETSC)
   MPI_Init(&argc, &argv);
#elif defined(USE_PETSC)
   PetscInitialize(&argc, &argv, NULL, NULL);
   ierr = PetscLogBegin(); CHKERRQ(ierr);
#endif

   ret = real_main(argc, argv);

#if defined(USE_MPI) && !defined(USE_PETSC)
   if (ret >= 0) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
   }
   else {
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }
#elif defined(USE_PETSC)
   ierr = PetscFinalize(); CHKERRQ(ierr);
#endif

   return ret;
}

/******************************************************************************/
#undef __FUNCT__
#define __FUNCT__ "real_main"
static int real_main (int argc, char *argv[]) {

   /* Timing vars */
   double wt1,wt2;
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
   double ut1,ut2,st1,st2;
#endif

   /* Files */
   char *DriverConfigFileName=NULL, *SolverConfigFileName=NULL;
   
   /* Driver and solver I/O arrays and parameters */
   double *evals, *rnorms;
   PRIMME_NUM *evecs;
   driver_params driver;
   primme_params *primme = &driver.primme;
   primme_preset_method method;
   int *permutation = NULL;

   /* Other miscellaneous items */
   int ret, retX=0;
   int i, *eperm = NULL;
   int master = 1;
   int procID = 0;

#ifdef USE_MPI
   MPI_Comm comm;
   int numProcs;
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &procID);

   comm = MPI_COMM_WORLD;
   master = (procID == 0);
#endif

   memset(&driver, 0, sizeof(driver));
   primme_initialize(primme);

   if (master) {
      /* ------------------------------------------------------------ */
      /* Get from command line the names for the 1 or 2 config files  */
      /* NOTE: PETSc arguments starts with '-' and they shouldn't be  */
      /*       considered as configuration files.                     */
      /* ------------------------------------------------------------ */
   
      if (argc == 2 || (argc > 2 && argv[2][0] == '-')) {
         DriverConfigFileName = argv[1];
         SolverConfigFileName = argv[1];
      } else if (argc >= 3) {
         DriverConfigFileName = argv[1];
         SolverConfigFileName = argv[2];
      } else {
         fprintf(stderr, "Invalid number of arguments.\n");
         return(-1);
      }
   
      /* ----------------------------- */
      /* Read in the driver parameters */
      /* ----------------------------- */
      if (read_driver_params(DriverConfigFileName, &driver) < 0) {
         fprintf(stderr, "Reading driver parameters failed\n");
         fflush(stderr);
         return(-1);
      }
   
      /* --------------------------------------- */
      /* Read in the PRIMME configuration file   */
      /* --------------------------------------- */
      if (read_solver_params(SolverConfigFileName, driver.outputFileName, 
                           primme, &method) < 0) {
         fprintf(stderr, "Reading solver parameters failed\n");
         return(-1);
      }
   }

#ifdef USE_MPI
   /* ------------------------------------------------- */
   /* Send read common primme members to all processors */ 
   /* Setup the primme members local to this processor  */ 
   /* ------------------------------------------------- */
   broadCast(primme, &method, &driver, master, comm);
#endif

   /* --------------------------------------- */
   /* Set up matrix vector and preconditioner */
   /* --------------------------------------- */
   if (setMatrixAndPrecond(&driver, primme, &permutation) != 0) return -1;

   if (master) {
      /* --------------------------------------- */
      /* Pick one of the default methods(if set) */
      /* --------------------------------------- */
      if (primme_set_method(method, primme) < 0 ) {
         fprintf(primme->outputFile, "No preset method. Using custom settings\n");
      }

      /* --------------------------------------- */
      /* Reread the PRIMME configuration file to */
      /* some override primme_set_method changes */
      /* --------------------------------------- */
       if (read_solver_params(SolverConfigFileName, driver.outputFileName, 
                              primme, &method) < 0) {
         fprintf(stderr, "Reading solver parameters failed\n");
         return(-1);
      }
   }

#ifdef USE_MPI
   /* ------------------------------------------------- */
   /* Send read common primme members to all processors */ 
   /* Setup the primme members local to this processor  */ 
   /* ------------------------------------------------- */
   broadCast(primme, &method, &driver, master, comm);
#endif

   /* --------------------------------------- */
   /* Optional: report memory requirements    */
   /* --------------------------------------- */

   ret = dprimme(NULL,NULL,NULL,primme);
   if (master) {
      fprintf(primme->outputFile,"PRIMME will allocate the following memory:\n");
      fprintf(primme->outputFile," processor %d, real workspace, %ld bytes\n",
                                      procID, primme->realWorkSize);
      fprintf(primme->outputFile," processor %d, int  workspace, %d bytes\n",
                                      procID, primme->intWorkSize);
   }

   /* --------------------------------------- */
   /* Display given parameter configuration   */
   /* Place this after the dprimme() to see   */
   /* any changes dprimme() made to PRIMME    */
   /* --------------------------------------- */

   if (master) {
      driver_display_params(driver, primme->outputFile); 
      primme_display_params(*primme);
      driver_display_method(method, primme->outputFile);
   }



   /* --------------------------------------------------------------------- */
   /*                            Run the d/zprimme solver                   */
   /* --------------------------------------------------------------------- */

   /* Allocate space for converged Ritz values and residual norms */

   evals = (double *)primme_calloc(primme->numEvals, sizeof(double), "evals");
   evecs = (PRIMME_NUM *)primme_calloc(primme->nLocal*primme->numEvals, 
                                sizeof(PRIMME_NUM), "evecs");
   rnorms = (double *)primme_calloc(primme->numEvals, sizeof(double), "rnorms");

   /* ------------------------ */
   /* Initial guess (optional) */
   /* ------------------------ */

   /* Read initial guess from a file */
   if (driver.initialGuessesFileName[0] && primme->initSize+primme->numOrthoConst > 0) {
      int cols, i=0;
      ASSERT_MSG(readBinaryEvecsAndPrimmeParams(driver.initialGuessesFileName, evecs, NULL, primme->n,
                                                min(primme->initSize+primme->numOrthoConst, primme->numEvals),
                                                &cols, primme->nLocal, permutation, primme) != 0, 1, "");
      primme->numOrthoConst = min(primme->numOrthoConst, cols);

      /* Perturb the initial guesses by a vector with some norm  */
      if (driver.initialGuessesPert > 0) {
         PRIMME_NUM *r = (PRIMME_NUM *)primme_calloc(primme->nLocal,sizeof(PRIMME_NUM), "random");
         double norm;
         int j;
         for (i=primme->numOrthoConst; i<min(cols, primme->initSize+primme->numOrthoConst); i++) {
            SUF(Num_larnv)(2, primme->iseed, primme->nLocal, COMPLEXZ(r));
            norm = sqrt(REAL_PARTZ(SUF(Num_dot)(primme->nLocal, COMPLEXZ(r), 1, COMPLEXZ(r), 1)));
            for (j=0; j<primme->nLocal; j++)
               evecs[primme->nLocal*i+j] += r[j]/norm*driver.initialGuessesPert;
         }
         free(r);
      }
      SUF(Num_larnv)(2, primme->iseed, (primme->initSize+primme->numOrthoConst-i)*primme->nLocal,
                     COMPLEXZ(&evecs[primme->nLocal*i]));
   } else if (primme->numOrthoConst > 0) {
      ASSERT_MSG(0, 1, "numOrthoConst > 0 but no value in initialGuessesFileName.\n");
   } else if (primme->initSize > 0) {
      SUF(Num_larnv)(2, primme->iseed, primme->initSize*primme->nLocal, COMPLEXZ(evecs));
   } else {
      SUF(Num_larnv)(2, primme->iseed, primme->nLocal, COMPLEXZ(evecs));
   }


   /* ------------- */
   /*  Call primme  */
   /* ------------- */

   wt1 = primme_get_wtime(); 
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
   primme_get_time(&ut1,&st1);
#endif

   if (primme->numBounds == 0) {
      ret = primmew(evals, evecs, rnorms, primme);
   } else {
      eperm = (int *)primme_calloc(primme->numEvals, sizeof(int), "eperm");
      ret = spectrum_slicing(evals, evecs, rnorms, primme, driver.maxLocked?driver.maxLocked:primme->numEvals, eperm);
   }

   wt2 = primme_get_wtime();
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
   primme_get_time(&ut2,&st2);
#endif

   if (driver.checkXFileName[0]) {
      retX = check_solution(driver.checkXFileName, primme, evals, evecs, rnorms, permutation);
   }

   /* --------------------------------------------------------------------- */
   /* Save evecs and primme params  (optional)                              */
   /* --------------------------------------------------------------------- */
   if (driver.saveXFileName[0]) {
      ASSERT_MSG(writeBinaryEvecsAndPrimmeParams(driver.saveXFileName, evecs, permutation, primme) == 0, 1, "");
   }

   /* --------------------------------------------------------------------- */
   /* Reporting                                                             */
   /* --------------------------------------------------------------------- */

   if (master) {
      primme_PrintStackTrace(*primme);

      fprintf(primme->outputFile, "Wallclock Runtime   : %-f\n", wt2-wt1);
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
      fprintf(primme->outputFile, "User Time           : %f seconds\n", ut2-ut1);
      fprintf(primme->outputFile, "Syst Time           : %f seconds\n", st2-st1);
#endif

      for (i=0; i < primme->initSize; i++) {
         fprintf(primme->outputFile, "Eval[%d]: %-22.15E rnorm: %-22.15E\n", i+1,
            evals[eperm?eperm[i]:i], rnorms[eperm?eperm[i]:i]); 
      }
      fprintf(primme->outputFile, " %d eigenpairs converged\n", primme->initSize);

      fprintf(primme->outputFile, "Tolerance    : %-22.15E\n", primme->aNorm*primme->eps);
      fprintf(primme->outputFile, "Iterations   : %-d\n", 
                                                    primme->stats.numOuterIterations); 
      fprintf(primme->outputFile, "Restarts     : %-d\n", primme->stats.numRestarts);
      fprintf(primme->outputFile, "Matvecs      : %-d\n", primme->stats.numMatvecs);
      fprintf(primme->outputFile, "Matvecs Time : %g\n",  primme->stats.elapsedTimeMatvec);
      fprintf(primme->outputFile, "Preconds     : %-d\n", primme->stats.numPreconds);
      fprintf(primme->outputFile, "Precond Time : %g\n",  primme->stats.elapsedTimePrecond);
      fprintf(primme->outputFile, "Reortho      : %-d\n", primme->stats.numReorthos);
      fprintf(primme->outputFile, "OrthoTime    : %g\n",  primme->stats.elapsedTimeOrtho);
      fprintf(primme->outputFile, "A MV Time    : %g\n",  elapsedTimeAMV);
      fprintf(primme->outputFile, "Filt Time    : %g\n",  elapsedTimeFilterMV);
      fprintf(primme->outputFile, "Filter app   : %-d\n", numFilterApplies);
      fprintf(primme->outputFile, "Solve H Time : %g\n",  primme->stats.elapsedTimeSolveH);
      if (primme->locking && primme->intWork && primme->intWork[0] == 1) {
         fprintf(primme->outputFile, "\nA locking problem has occurred.\n");
         fprintf(primme->outputFile,
            "Some eigenpairs do not have a residual norm less than the tolerance.\n");
         fprintf(primme->outputFile,
            "However, the subspace of evecs is accurate to the required tolerance.\n");
      }

      fprintf(primme->outputFile, "\n\n#,%d,%.1f\n\n", primme->stats.numMatvecs,
         wt2-wt1); 

      switch (primme->dynamicMethodSwitch) {
         case -1: fprintf(primme->outputFile,
               "Recommended method for next run: DEFAULT_MIN_MATVECS\n"); break;
         case -2: fprintf(primme->outputFile,
               "Recommended method for next run: DEFAULT_MIN_TIME\n"); break;
         case -3: fprintf(primme->outputFile,
               "Recommended method for next run: DYNAMIC (close call)\n"); break;
      }
   }

   fclose(primme->outputFile);
   destroyMatrixAndPrecond(&driver, primme, permutation);
   primme_Free(primme);
   free(evals);
   free(evecs);
   free(rnorms);

   if (ret != 0 && master) {
      fprintf(primme->outputFile, 
         "Error: dprimme returned with nonzero exit status: %d \n",ret);
      return -1;
   }

   if (retX != 0 && master) {
      fprintf(primme->outputFile, 
         "Error: found some issues in the solution return by dprimme\n");
      return -1;
   }

  return(0);
}
/******************************************************************************/
/* END OF MAIN DRIVER FUNCTION                                                */
/******************************************************************************/

/******************************************************************************/
/* Matvec, preconditioner and other utilities                                 */

/******************************************************************************/

#ifdef USE_MPI
/******************************************************************************
 * Function to broadcast the primme data structure to all processors
 *
 * EXCEPTIONS: procID and seed[] are not copied from processor 0. 
 *             Each process creates their own.
******************************************************************************/
static void broadCast(primme_params *primme, primme_preset_method *method, 
   driver_params *driver, int master, MPI_Comm comm){

   int i;

   MPI_Bcast(driver->outputFileName, 512, MPI_CHAR, 0, comm);
   MPI_Bcast(driver->matrixFileName, 1024, MPI_CHAR, 0, comm);
   MPI_Bcast(driver->initialGuessesFileName, 1024, MPI_CHAR, 0, comm);
   MPI_Bcast(driver->saveXFileName, 1024, MPI_CHAR, 0, comm);
   MPI_Bcast(driver->checkXFileName, 1024, MPI_CHAR, 0, comm);
   MPI_Bcast(&driver->initialGuessesPert, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->matrixChoice, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->PrecChoice, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->isymm, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->level, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->threshold, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->filter, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->shift, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->minEig, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->maxEig, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->maxLocked, 1, MPI_INT, 0, comm);

   MPI_Bcast(&driver->AFilter.filter, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->AFilter.degrees, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->AFilter.lowerBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->AFilter.upperBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->AFilter.lowerBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->AFilter.upperBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->AFilter.checkEps, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->precFilter.filter, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->precFilter.degrees, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->precFilter.lowerBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->precFilter.upperBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->precFilter.lowerBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->precFilter.upperBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->precFilter.checkEps, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->orthoFilter.filter, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->orthoFilter.degrees, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->orthoFilter.lowerBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->orthoFilter.upperBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->orthoFilter.lowerBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->orthoFilter.upperBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->transform.filter, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->transform.degrees, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->transform.lowerBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->transform.upperBound, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->transform.lowerBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->transform.upperBoundFix, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->transform.checkEps, 1, MPI_DOUBLE, 0, comm);

   MPI_Bcast(&(primme->numEvals), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->target), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->numTargetShifts), 1, MPI_INT, 0, comm);

   if (primme->numTargetShifts > 0 && !master) {
      primme->targetShifts = (double *)primme_calloc(
         primme->numTargetShifts, sizeof(double), "targetShifts");
   }
   for (i=0; i<primme->numTargetShifts; i++) {
      MPI_Bcast(&(primme->targetShifts[i]), 1, MPI_DOUBLE, 0, comm);
   }
   MPI_Bcast(&(primme->numBounds), 1, MPI_INT, 0, comm);

   if (primme->numBounds > 0 && !master) {
      primme->bounds = (double *)primme_calloc(
         primme->numBounds, sizeof(double), "bounds");
   }
   for (i=0; i<primme->numBounds; i++) {
      MPI_Bcast(&(primme->bounds[i]), 1, MPI_DOUBLE, 0, comm);
   }
   MPI_Bcast(&(primme->locking), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->dynamicMethodSwitch), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->initSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->numOrthoConst), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->maxBasisSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->minRestartSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->maxBlockSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->maxMatvecs), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->maxOuterIterations), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->aNorm), 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&(primme->eps), 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&(primme->printLevel), 1, MPI_INT, 0, comm);

   MPI_Bcast(&(primme->restartingParams.scheme), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->restartingParams.maxPrevRetain), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->applyPrecondTo), 1, MPI_INT, 0, comm);

   MPI_Bcast(&(primme->correctionParams.precondition), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->correctionParams.robustShifts), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->correctionParams.maxInnerIterations),1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.convTest), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->correctionParams.relTolBase), 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&(primme->correctionParams.projectors.LeftQ),  1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.projectors.LeftX),  1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.projectors.RightQ), 1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.projectors.RightX), 1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.projectors.SkewQ),  1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.projectors.SkewX),  1, MPI_INT, 0,comm);
   MPI_Bcast(&(primme->correctionParams.projectors.SkewX),  1, MPI_INT, 0,comm);

   MPI_Bcast(method, 1, MPI_INT, 0, comm);
}
#endif

static int setMatrixAndPrecond(driver_params *driver, primme_params *primme, int **permutation) {
   int numProcs=1;

#  if defined(USE_MPI) || defined(USE_PETSC)
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   primme->commInfo = (MPI_Comm *)primme_calloc(1, sizeof(MPI_Comm), "MPI_Comm");
   *(MPI_Comm*)primme->commInfo = MPI_COMM_WORLD;
#  endif

   if (driver->matrixChoice == driver_default) {
      if (numProcs <= 1) {
         driver->matrixChoice = driver_native;
      } else {
#        ifdef USE_PETSC
            driver->matrixChoice = driver_petsc;
#        else
            driver->matrixChoice = driver_parasails;
#        endif
      }
   }
   switch(driver->matrixChoice) {
   case driver_default:
      assert(0);
      break;
   case driver_native:
#if !defined(USE_NATIVE)
      fprintf(stderr, "ERROR: NATIVE is needed!\n");
      return -1;
#else
#  if defined(USE_MPI)
      if (numProcs != 1) {
         fprintf(stderr, "ERROR: MPI is not supported with NATIVE, use other!\n");
         return -1;
      }
      *(MPI_Comm*)primme->commInfo = MPI_COMM_WORLD;
#  endif
      {
         CSRMatrix *matrix;
         
         if (readMatrixNative(driver->matrixFileName, &matrix, &primme->aNorm) !=0 )
            return -1;
         primme->matrix = matrix;
         primme->matrixMatvec = CSRMatrixMatvec;
         primme->n = primme->nLocal = matrix->n;
         switch(driver->PrecChoice) {
         case driver_noprecond:
            primme->applyPreconditioner = NULL;
            break;
         case driver_jacobi:
            primme->applyPreconditioner = ApplyInvDiagPrecNative;
            break;
         case driver_jacobi_i:
            primme->applyPreconditioner = ApplyInvDavidsonDiagPrecNative;
            break; 
         case driver_ilut:
            primme->applyPreconditioner = ApplyILUTPrecNative;
            break;
         case driver_filter:
            break;
         }
         primme->preconditioner = NULL;
      }
#endif
      break;

   case driver_parasails:
#if !defined(USE_PARASAILS)
      fprintf(stderr, "ERROR: ParaSails is needed!\n");
      return -1;
      if (driver->PrecChoice != driver_ilut) {
         fprintf(stderr, "ERROR: ParaSails only supports ILUT preconditioner!\n");
         return -1;
      }
#else
      {
         Matrix *matrix;
         ParaSails *precond=NULL;
         int m, mLocal;
         readMatrixAndPrecondParaSails(driver->matrixFileName, driver->shift, driver->level,
               driver->threshold, driver->filter, driver->isymm, MPI_COMM_WORLD, &primme->aNorm,
               &primme->n, &m, &primme->nLocal, &mLocal, &primme->numProcs, &primme->procID, &matrix,
               (driver->PrecChoice == driver_ilut) ? &precond : NULL);
         *(MPI_Comm*)primme->commInfo = MPI_COMM_WORLD;
         primme->matrix = matrix;
         primme->matrixMatvec = ParaSailsMatrixMatvec;
         primme->preconditioner = precond;
         primme->applyPreconditioner = precond ? ApplyPrecParaSails : NULL;
      }
#endif
      break;

   case driver_petsc:
#ifndef USE_PETSC
      fprintf(stderr, "ERROR: PETSc is needed!\n");
      return -1;
#else
      {
         PetscErrorCode ierr;
         Mat *matrix,A;
         Vec *vec;
         int m, mLocal;
         PETScPrecondStruct *s;
         PC pc;
         if (readMatrixPetsc(driver->matrixFileName, &primme->n, &m, &primme->nLocal, &mLocal,
                         &primme->numProcs, &primme->procID, &matrix, &primme->aNorm, permutation) != 0)
            return -1;
         *(MPI_Comm*)primme->commInfo = PETSC_COMM_WORLD;
         primme->matrix = matrix;
         primme->matrixMatvec = PETScMatvec;
         if (driver->PrecChoice == driver_noprecond) {
            primme->preconditioner = NULL;
            primme->applyPreconditioner = NULL;
         }
         else if (driver->PrecChoice != driver_jacobi_i) {
            s = (PETScPrecondStruct *)primme_calloc(1, sizeof(PETScPrecondStruct), "PETScPrecondStruct");
            s->prevShift = 0;
            ierr = KSPCreate(PETSC_COMM_WORLD, &s->ksp); CHKERRQ(ierr);
            ierr = KSPSetType(s->ksp, KSPPREONLY); CHKERRQ(ierr);
            ierr = KSPGetPC(s->ksp,&pc); CHKERRQ(ierr);
            if (driver->PrecChoice == driver_jacobi) {
               ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
            }
            else if (driver->PrecChoice == driver_ilut) {
               if (primme->numProcs <= 1) {
                  ierr = PCSetType(pc, PCILU); CHKERRQ(ierr);
               }
               else {
                  #ifdef PETSC_HAVE_HYPRE
                     ierr = PCSetType(pc, PCHYPRE); CHKERRQ(ierr);
                     ierr = PCHYPRESetType(pc, "parasails"); CHKERRQ(ierr);
                  #else
                     ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);
                  #endif
               }
            }
            ierr = MatDuplicate(*matrix, MAT_COPY_VALUES, &A);CHKERRQ(ierr);
            ierr = KSPSetOperators(s->ksp, A, A); CHKERRQ(ierr);
            ierr = KSPSetFromOptions(s->ksp); CHKERRQ(ierr);
            primme->preconditioner = s;
            primme->applyPreconditioner = ApplyPCPrecPETSC;
         }
         else {
            vec = (Vec *)primme_calloc(1, sizeof(Vec), "Vec preconditioner");
            #if PETSC_VERSION_LT(3,6,0)
            ierr = MatGetVecs(*matrix, vec, NULL); CHKERRQ(ierr);
            #else
            ierr = MatCreateVecs(*matrix, vec, NULL); CHKERRQ(ierr);
            #endif
            ierr = MatGetDiagonal(*matrix, *vec); CHKERRQ(ierr);
            primme->preconditioner = vec;
            primme->applyPreconditioner = ApplyInvDavidsonDiagPrecPETSc;
         }
      }
#endif
      break;
   }

#if defined(USE_MPI)
   primme->globalSumDouble = par_GlobalSumDouble;
#endif

   /* Setup transform */
   if (driver->transform.filter) {
      driver->transform.matvec = primme->matrixMatvec;
      primme->matrixMatvec = Apply_transform_filter;
      primme->matrixMatvec(NULL, NULL, NULL, primme);
   }

   /* Build precond based on shift */
   if (driver->PrecChoice != driver_noprecond && driver->PrecChoice != driver_filter) {
      primme->rebuildPreconditioner = 1;
      primme->numberOfShiftsForPreconditioner = 1;
      primme->ShiftsForPreconditioner = &driver->shift;
      primme->applyPreconditioner(NULL, NULL, NULL, primme);
   }
   return 0;
}

static int destroyMatrixAndPrecond(driver_params *driver, primme_params *primme, int *permutation) {
   switch(driver->matrixChoice) {
   case driver_default:
      assert(0);
      break;
   case driver_native:
#if !defined(USE_NATIVE)
      fprintf(stderr, "ERROR: NATIVE is needed!\n");
      return -1;
#else
      freeCSRMatrix((CSRMatrix*)primme->matrix);

      switch(driver->PrecChoice) {
      case driver_noprecond:
         break;
      case driver_jacobi:
      case driver_jacobi_i:
         free(primme->preconditioner);
         break;
      case driver_ilut:
         if (primme->preconditioner) {
            free(((CSRMatrix*)primme->preconditioner)->AElts);
            free(((CSRMatrix*)primme->preconditioner)->IA);
            free(((CSRMatrix*)primme->preconditioner)->JA);
            free(primme->preconditioner);
         }
         break;
      case driver_filter:
         break;
      }
#endif
      break;

   case driver_parasails:
#if !defined(USE_PARASAILS)
      fprintf(stderr, "ERROR: ParaSails is needed!\n");
      return -1;
      if (driver->PrecChoice != driver_ilut) {
         fprintf(stderr, "ERROR: ParaSails only supports ILUT preconditioner!\n");
         return -1;
      }
#else
      /* TODO: destroy ParaSail matrices */

#endif
      break;

   case driver_petsc:
#ifndef USE_PETSC
      fprintf(stderr, "ERROR: PETSc is needed!\n");
      return -1;
#else
      {
         PetscErrorCode ierr;
         ierr = MatDestroy((Mat*)primme->matrix);CHKERRQ(ierr);
         if (primme->preconditioner) {
         }
         if (driver->PrecChoice == driver_noprecond) {
         }
         else if (driver->PrecChoice != driver_jacobi_i) {
            ierr = PCDestroy((PC*)primme->preconditioner);CHKERRQ(ierr);
            free(primme->preconditioner);
         }
         else {
            ierr = VecDestroy((Vec*)primme->preconditioner);CHKERRQ(ierr);
            free(primme->preconditioner);
         }
      }
#endif
      break;
   }
#if defined(USE_MPI)
   free(primme->commInfo);
#endif
   if (permutation) free(permutation);
   return 0;
}


#ifdef USE_MPI
/******************************************************************************
 * MPI globalSumDouble function
 *
******************************************************************************/
static void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, 
                         primme_params *primme) {
   MPI_Comm communicator = *(MPI_Comm *) primme->commInfo;

   MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "check_solution"
static int check_solution(const char *checkXFileName, primme_params *primme, double *evals,
                   PRIMME_NUM *evecs, double *rnorms, int *perm) {

   double eval0, rnorm0, prod;
   PRIMME_NUM *Ax, *r, *X=NULL, *h, *h0;
   int i, j, cols, retX=0, one=1;
   primme_params primme0;

   /* Read stored eigenvectors and primme_params */
   ASSERT_MSG(readBinaryEvecsAndPrimmeParams(checkXFileName, NULL, &X, primme->n, primme->n, &cols,
                                             primme->nLocal, perm, &primme0) == 0, -1, "");
   /* Check primme_params */
#  define CHECK_PRIMME_PARAM(F) \
        if (primme0. F != primme-> F ) { \
           fprintf(stderr, "Warning: discrepancy in primme." #F ", %d should be close to %d\n", primme-> F , primme0. F ); \
           retX = 1; \
        }
#  define CHECK_PRIMME_PARAM_DOUBLE(F) \
        if (fabs(primme0. F - primme-> F) > primme-> F * 1e-14) { \
           fprintf(stderr, "Warning: discrepancy in primme." #F ", %.16e should be close to %.16e\n", primme-> F , primme0. F ); \
           retX = 1; \
        }
#  define CHECK_PRIMME_PARAM_TOL(F, T) \
        if (abs(primme0. F - primme-> F ) > primme-> F * T /100+1) { \
           fprintf(stderr, "Warning: discrepancy in primme." #F ", %d should be close to %d\n", primme-> F , primme0. F ); \
           retX = 1; \
        }

   if (primme0.n) {
      CHECK_PRIMME_PARAM(n);
      CHECK_PRIMME_PARAM(numEvals);
      CHECK_PRIMME_PARAM(target);
      CHECK_PRIMME_PARAM(numTargetShifts);
      CHECK_PRIMME_PARAM(dynamicMethodSwitch);
      CHECK_PRIMME_PARAM(locking);
      CHECK_PRIMME_PARAM(numOrthoConst);
      CHECK_PRIMME_PARAM(maxBasisSize);
      CHECK_PRIMME_PARAM(minRestartSize);
      CHECK_PRIMME_PARAM(restartingParams.scheme);
      CHECK_PRIMME_PARAM(restartingParams.maxPrevRetain);
      CHECK_PRIMME_PARAM(correctionParams.precondition);
      CHECK_PRIMME_PARAM(correctionParams.robustShifts);
      CHECK_PRIMME_PARAM(correctionParams.maxInnerIterations);
      CHECK_PRIMME_PARAM(correctionParams.projectors.LeftQ);
      CHECK_PRIMME_PARAM(correctionParams.projectors.LeftX);
      CHECK_PRIMME_PARAM(correctionParams.projectors.RightQ);
      CHECK_PRIMME_PARAM(correctionParams.projectors.RightX);
      CHECK_PRIMME_PARAM(correctionParams.projectors.SkewQ);
      CHECK_PRIMME_PARAM(correctionParams.projectors.SkewX);
      CHECK_PRIMME_PARAM(correctionParams.convTest);
      CHECK_PRIMME_PARAM_DOUBLE(aNorm);
      CHECK_PRIMME_PARAM_DOUBLE(eps);
      CHECK_PRIMME_PARAM_DOUBLE(correctionParams.relTolBase);
      CHECK_PRIMME_PARAM(initSize);
      CHECK_PRIMME_PARAM_TOL(stats.numOuterIterations, 40);
   }

   h = (PRIMME_NUM *)primme_calloc(cols*2, sizeof(PRIMME_NUM), "h"); h0 = &h[cols];
   Ax = (PRIMME_NUM *)primme_calloc(primme->nLocal, sizeof(PRIMME_NUM), "Ax");
   r = (PRIMME_NUM *)primme_calloc(primme->nLocal, sizeof(PRIMME_NUM), "r");
   
   for (i=0; i < primme->initSize; i++) {
      /* Check |V(:,i)'A*V(:,i) - evals[i]| < |r|*|A| */
      primme->matrixMatvec(&evecs[primme->nLocal*i], Ax, &one, primme);
      eval0 = REAL_PART(primme_dot(&evecs[primme->nLocal*i], Ax, primme));
      if (fabs(evals[i] - eval0) > rnorms[i]*primme->aNorm && primme->procID == 0) {
         fprintf(stderr, "Warning: Eval[%d] = %-22.15E should be close to %-22.1E\n", i, evals[i], eval0);
         retX = 1;
      }
      /* Check |A*V(:,i) - (V(:,i)'A*V(:,i))*V(:,i)| < |r| */
      for (j=0; j<primme->nLocal; j++) r[j] = Ax[j] - evals[i]*evecs[primme->nLocal*i+j];
      rnorm0 = sqrt(REAL_PART(primme_dot(r, r, primme)));
      if (fabs(rnorms[i]-rnorm0) > 4*max(primme->aNorm,fabs(evals[i]))*MACHINE_EPSILON && primme->procID == 0) {
         fprintf(stderr, "Warning: Eval[%d] = %-22.15E, residual | %5E - %5E | <= %5E\n", i, evals[i], rnorms[i], rnorm0, 4*max(primme->aNorm,fabs(evals[i]))*MACHINE_EPSILON);
         retX = 1;
      }
      if (rnorm0 > primme->eps*primme->aNorm*sqrt((double)(i+1)) && primme->procID == 0) {
         fprintf(stderr, "Warning: Eval[%d] = %-22.15E, RR residual %5E is larger than tolerance %5E\n", i, evals[i], rnorm0, primme->eps*primme->aNorm*sqrt((double)(i+1)));
         retX = 1;
      }
      /* Check X'V(:,i) >= sqrt(1-2|r|), assuming residual of X is less than the tolerance */
      SUF(Num_gemv)("C", primme->nLocal, cols, COMPLEXZV(1.0), COMPLEXZ(X), primme->nLocal, COMPLEXZ(&evecs[primme->nLocal*i]), 1, COMPLEXZV(0.), COMPLEXZ(h), 1);
      if (primme->globalSumDouble) {
         int cols0 = cols*sizeof(PRIMME_NUM)/sizeof(double);
         primme->globalSumDouble(h, h0, &cols0, primme);
      }
      else h0 = h;
      prod = REAL_PARTZ(SUF(Num_dot)(cols, COMPLEXZ(h0), 1, COMPLEXZ(h0), 1));
      if (prod < sqrt(1.-2.*rnorms[i]) && primme->procID == 0) {
         fprintf(stderr, "Warning: Eval[%d] = %-22.15E not found on X, %5E > %5E\n", i, evals[i], prod, sqrt(1.-2.*rnorms[i]));
         retX = 1;
      }
   }
   free(h);
   free(X);
   free(r);
   free(Ax);

   return retX; 
}

#undef __FUNCT__
#define __FUNCT__ "readBinaryEvecsAndPrimmeParams"
static int readBinaryEvecsAndPrimmeParams(const char *fileName, PRIMME_NUM *X, PRIMME_NUM **Xout,
                                          int n, int Xcols, int *Xcolsout, int nLocal,
                                          int *perm, primme_params *primme_out) {

#  define FREAD(A, B, C, D) { ASSERT_MSG(fread(A, B, C, D) == (size_t)C, -1, "Unexpected end of file\n"); }

   FILE *f;
   PRIMME_NUM d;
   int i, j, cols;

   ASSERT_MSG((f = fopen(fileName, "rb")),
                  -1, "Could not open file %s\n", fileName);

   /* Check number size */
   /* NOTE: 2*IMAGINARY*IMAGINARY+1 is -1 in complex arith and 1 in real arith */
   FREAD(&d, sizeof(d), 1, f);
   ASSERT_MSG((int)(REAL_PART(d*(2.*IMAGINARY*IMAGINARY + 1.))) == (int)sizeof(d),
                  -1, "Mismatch arithmetic in file %s\n", fileName);
   /* Check matrix size */
   FREAD(&d, sizeof(d), 1, f);
   ASSERT_MSG(((int)REAL_PART(d)) == n,
                  -1, "Mismatch matrix size in file %s\n", fileName);

   /* Read X */
   FREAD(&d, sizeof(d), 1, f); cols = REAL_PART(d);
   if (Xcols > 0 && (X || Xout)) {
      if (!X) *Xout = X = (PRIMME_NUM*)malloc(sizeof(PRIMME_NUM)*min(cols, Xcols)*nLocal);
      if (Xcolsout) *Xcolsout = min(cols, Xcols);
      if (!perm) {
         for (i=0; i<min(cols, Xcols); i++) {
            fseek(f, (i*n + 3)*sizeof(d), SEEK_SET);
            FREAD(&X[nLocal*i], sizeof(d), nLocal, f);
         }
      }
      else {
         for (i=0; i<min(cols, Xcols); i++) {
            for (j=0; j<nLocal; j++) {
               fseek(f, (i*n + perm[j] + 3)*sizeof(d), SEEK_SET);
               FREAD(&X[nLocal*i+j], sizeof(d), 1, f);
            }
         }
      }
   }
   fseek(f, (cols*n + 3)*sizeof(d), SEEK_SET);

   /* Read primme_params */
   if (primme_out) {
      FREAD(&d, sizeof(d), 1, f);
      if ((int)REAL_PART(d) == (int)sizeof(*primme_out)) {
         FREAD(primme_out, sizeof(*primme_out), 1, f);
      }
      else
         primme_out->n = 0;
   }

   fclose(f);
   return 0;
}

#undef __FUNCT__
#define __FUNCT__ "writeBinaryEvecsAndPrimmeParams"
static int writeBinaryEvecsAndPrimmeParams(const char *fileName, PRIMME_NUM *X, int *perm,
                                           primme_params *primme) {

#  define FWRITE(A, B, C, D) { ASSERT_MSG(fwrite(A, B, C, D) == (size_t)C, -1, "Unexpected error writing on %s\n", fileName); }

   FILE *f;
   PRIMME_NUM d;
   int i, j;

   ASSERT_MSG((f = fopen(fileName, "wb")),
                  -1, "Could not open file %s\n", fileName);

   /* Write number size */
   if (primme->procID == 0) {
      /* NOTE: 2*IMAGINARY*IMAGINARY+1 is -1 in complex arith and 1 in real arith */
      d = (2.*IMAGINARY*IMAGINARY + 1.)*sizeof(d);
      FWRITE(&d, sizeof(d), 1, f);
      /* Write matrix size */
      d = primme->n;
      FWRITE(&d, sizeof(d), 1, f);
      /* Write number of columns */
      d = primme->initSize;
      FWRITE(&d, sizeof(d), 1, f);
   }

   /* Write X */
   if (!perm) {
      for (i=0; i<primme->initSize; i++) {
         fseek(f, (i*primme->n + 3)*sizeof(d), SEEK_SET);
         FWRITE(&X[primme->nLocal*i], sizeof(d), primme->nLocal, f);
      }
   }
   else {
      for (i=0; i<primme->initSize; i++) {
         for (j=0; j<primme->nLocal; j++) {
            fseek(f, (i*primme->n + perm[j] + 3)*sizeof(d), SEEK_SET);
            FWRITE(&X[primme->nLocal*i+j], sizeof(d), 1, f);
         }
      }
   }

   /* Write primme_params */
   if (primme->procID == 0) {
      fseek(f, sizeof(d)*(primme->n*primme->initSize + 3), SEEK_SET);
      d = sizeof(*primme);
      FWRITE(&d, sizeof(d), 1, f);
      FWRITE(primme, sizeof(*primme), 1, f);
   }

   fclose(f);
   return 0;
}


/******************************************************************************
 * Set filters
 *
 ******************************************************************************/

static void Apply_A_filter(void *x, void *y, int *blockSize,
                  primme_params *primme) {

   driver_params *driver = (driver_params*)primme;
   double ub, lb, ub0, lb0;
   int oldDegrees;
   static int iwork[4], newDegrees;
   static double work[4];
   static filter_params filter0 = {-10,0};
   static primme_target target;

   /* Save a copy of the original filter parameters */
   if (!blockSize && !x && !y) {
      filter0 = driver->AFilter;
      filter0.lowerBound = driver->AFilter.upperBound = 8; /* set fixed bounds */
      if (primme->numBounds) {
         filter0.lowerBoundFix = primme->bounds[0];
         filter0.upperBoundFix = primme->bounds[1];
      }
      newDegrees = filter0.degrees;
      target = primme->target;
      primme->target = primme_largest;
   }
   /* Restore original primme params */
   if (!blockSize && x && !y) {
      if (primme->numBounds) {
         primme->bounds[0] = filter0.lowerBoundFix;
         primme->bounds[1] = filter0.upperBoundFix;
      }
      primme->target = target;
      return;
   }
   assert(filter0.filter != -10);

   oldDegrees = filter0.degrees;
   if (oldDegrees != newDegrees && blockSize && primme->stats.numOuterIterations == 0) {
      filter0.degrees = newDegrees;
      oldDegrees = -1;
   }
      
   if (blockSize && driver->AFilter.degrees < 0) {
      if (reconsider_degree_filter(&filter0, primme, work, iwork, timeCostModelA)) {
         if (oldDegrees > 1) {
            newDegrees = filter0.degrees;
            filter0.degrees = oldDegrees;
            /* Force restart PRIMME with the new degree */
            primme->maxMatvecs = -1;
         } else {
            oldDegrees = newDegrees = filter0.degrees;
         }
      }
   }
   if (filter0.degrees > 0)
      if (tune_filter(&filter0, primme, blockSize == 0, primme->numBounds>0))
         oldDegrees = newDegrees = filter0.degrees;
   Apply_filter(x, y, blockSize, &filter0, primme, 1);
   if (primme->printLevel >= 5 && blockSize) plot_filter(1000, &filter0, primme, stderr);
   if (blockSize) primme->stats.numMatvecs -= *blockSize;
   if (primme->printLevel > 3 && primme->outputFile) {
      getBoundsTuned(&filter0, primme, &lb, &ub);
      getBounds(&filter0, primme, &lb0, &ub0);
      fprintf(primme->outputFile, "filter: A %d bounds: [ %g %g ] bounds0: [ %g %g] d:%d\n", filter0.filter, lb, ub, lb0, ub0, filter0.degrees);
   }
}

static void Apply_precon_filter(void *x, void *y, int *blockSize,
                  primme_params *primme) {

   driver_params *driver = (driver_params*)primme;
   double ub, lb, ub0, lb0;
   static int iwork[4];
   static double work[4];
   static filter_params filter0 = {-10,0};

   /* Save a copy of the original filter parameters */
   if (filter0.filter == -10) filter0 = driver->precFilter;

   if (blockSize && filter0.degrees < 0) {
      reconsider_degree_filter(&driver->precFilter, primme, work, iwork, timeCostModelPrecond);
   }
   if (driver->precFilter.degrees > 0)
      tune_filter(&driver->precFilter, primme, blockSize == 0, 0);
   Apply_filter(x, y, blockSize, &driver->precFilter, primme, 1);
   if (primme->printLevel >= 5 && blockSize) plot_filter(1000, &driver->precFilter, primme, stderr);
   if (blockSize) primme->stats.numPreconds -= *blockSize;
   if (primme->printLevel > 3 && primme->outputFile) {
      getBoundsTuned(&driver->AFilter, primme, &lb, &ub);
      getBounds(&driver->AFilter, primme, &lb0, &ub0);
      fprintf(primme->outputFile, "filter: A %d bounds: [ %g %g ] bounds0: [ %g %g] d:%d\n", driver->AFilter.filter, lb, ub, lb0, ub0, driver->AFilter.degrees);
   }

}

static void Apply_ortho_filter(void *x, void *y, int *blockSize,
                  primme_params *primme) {

   driver_params *driver = (driver_params*)primme;
   Apply_filter(x, y, blockSize, &driver->orthoFilter, primme, 1);
   if (primme->printLevel >= 5 && blockSize) plot_filter(1000, &driver->orthoFilter, primme, stderr);

}

static void Apply_transform_filter(void *x, void *y, int *blockSize,
                  primme_params *primme) {

   driver_params *driver = (driver_params*)primme;
   Apply_filter(x, y, blockSize, &driver->transform, primme, 1);
   if (primme->printLevel >= 5 && blockSize) plot_filter(1000, &driver->transform, primme, stderr);

}


static void setFilters(driver_params *driver, primme_params *primme) {

   driver->AFilter.matvec = driver->orthoFilter.matvec = driver->precFilter.matvec = primme->matrixMatvec;
   driver->AFilter.precond = driver->orthoFilter.precond = driver->precFilter.precond = primme->applyPreconditioner;
   driver->AFilter.minEig = driver->orthoFilter.minEig = driver->precFilter.minEig = driver->minEig;
   driver->AFilter.maxEig = driver->orthoFilter.maxEig = driver->precFilter.maxEig = driver->maxEig;
   if (driver->AFilter.filter) {
      primme->matrixMatvec = Apply_A_filter;
      driver->AFilter.prodIfFullRange = 1;
      primme->matrixMatvec(NULL, NULL, NULL, primme);
   }
   if (driver->orthoFilter.filter) {
      primme->applyOrtho = Apply_ortho_filter;
      driver->orthoFilter.prodIfFullRange = 0;
   } else {
      primme->applyOrtho = NULL;
   }
   if (driver->precFilter.filter) {
      primme->applyPreconditioner = Apply_precon_filter;
      driver->precFilter.prodIfFullRange = 1;
   }
   elapsedTimeAMV = elapsedTimeFilterMV = 0;
   numFilterApplies = 0;

}

static void unsetFilters(double *evals, PRIMME_NUM *evecs, double *rnorms, primme_params *primme) {
   driver_params *driver = (driver_params*)primme;

   if (driver->AFilter.filter > 0 && driver->AFilter.filter != 13) {
      /* Fix evalues and rnorms */
      PRIMME_NUM *Ax, *r;
      int one = 1, i,j;
      Ax = (PRIMME_NUM *)primme_calloc(primme->nLocal*2, sizeof(PRIMME_NUM), "Ax");
      r = Ax + primme->nLocal;
      for (i=0; i<primme->initSize; i++) {
         driver->AFilter.matvec(&evecs[primme->nLocal*i], Ax, &one, primme);
         evals[i] = REAL_PART(primme_dot(&evecs[primme->nLocal*i], Ax, primme));
         for (j=0; j<primme->nLocal; j++) r[j] = Ax[j] - evals[i]*evecs[primme->nLocal*i+j];
         rnorms[i] = sqrt(REAL_PART(primme_dot(r, r, primme)));
      }
      free(Ax);

      primme->matrixMatvec(evecs, NULL, NULL, primme);
      primme->matrixMatvec = driver->AFilter.matvec;
   }
   if (driver->precFilter.filter) {
      primme->applyPreconditioner = driver->precFilter.precond;
   }

}


static int primmew(double *evals, PRIMME_NUM *evecs, double *rnorms, primme_params *primme) {
   primme_stats stats0 = (primme_stats){0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   int numOrthoConst0, nconv, minRestartSize, ret;
   driver_params *driver = (driver_params*)primme;
   int master = 1;

#ifdef USE_MPI
   int procID;
   MPI_Comm_rank(MPI_COMM_WORLD, &procID);
   master = (procID == 0);
#endif


   numOrthoConst0 = primme->numOrthoConst;
   minRestartSize = primme->minRestartSize;
   nconv = 0;
   setFilters(driver, primme);
   while(1) {
      if (master) primme_display_params(*primme);

      ret = PREFIX(primme)(evals+primme->numOrthoConst-numOrthoConst0, COMPLEXZ(evecs),
                           rnorms+primme->numOrthoConst-numOrthoConst0, primme);
      ADD_STATS(stats0, primme->stats);
      nconv += primme->initSize;
      //primme->numEvals -= primme->initSize;
      if (primme->maxMatvecs != -1 || primme->initSize >= primme->maxBasisSize) break;
      primme->maxMatvecs = 0;
      //primme->numOrthoConst += primme->initSize;
      //primme->initSize = primme->minRestartSize;
      primme->initSize = primme->minRestartSize+primme->initSize;
      primme->minRestartSize = minRestartSize;
   }
   primme->stats = stats0;
   //primme->initSize = nconv;
   //primme->numEvals += nconv;

   unsetFilters(evals, evecs, rnorms, primme);

   return ret;
}


static int spectrum_slicing(double *evals, PRIMME_NUM *evecs, double *rnorms, primme_params *primme,
                            int maxNumEvals, int *perm) {
   driver_params *driver = (driver_params*)primme, driver0 = *driver;
   primme_params *primme0 = &driver0.primme;
   double norm, lastMinEig=HUGE_VAL, lastMaxEig=-HUGE_VAL, bounds[2], queue[200], r;
   int i, j, k, l, sliceNum, qtop=-100, numEvals, ret;

   assert(primme->numBounds == 2 && primme->target == primme_closest_abs && primme->bounds[0] < primme->bounds[1]);

   bounds[0] = primme->bounds[0];
   bounds[1] = primme->bounds[1];
   primme->stats = (primme_stats){0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   for (i=0; i<primme->numEvals; i++) perm[i] = i;
   r = (bounds[1] - bounds[0])/primme->numEvals;

   for(numEvals=sliceNum=0; numEvals < primme->numEvals; sliceNum++ ) {
      /* Set filter for this slice */
      if (!slice_tree(&bounds[0], &bounds[1], lastMinEig, lastMaxEig, queue, 200, &qtop)) {
         fprintf(primme->outputFile,"All range has been visited!\n");
         break;
      }
      primme0->numBounds = 2;
      primme0->bounds = bounds;

      primme0->numEvals = min(maxNumEvals, primme->numEvals - numEvals);
      primme0->initSize = 0;
      if (bounds[1] - bounds[0] > r*maxNumEvals) {
         double d = r*maxNumEvals, c = (bounds[1] + bounds[0])/2;
         bounds[0] = c - d/2;
         bounds[1] = c + d/2;
         
      }
      if (primme->numTargetShifts == 1) {
         primme0->targetShifts[0] = (bounds[0] + bounds[1])/2.;
      }
      ret = primmew(&evals[numEvals], &evecs[primme->nLocal*numEvals], &rnorms[numEvals], primme0);
      ADD_STATS(primme->stats, primme0->stats);

      /* Store new eigenpairs */
      lastMinEig=HUGE_VAL, lastMaxEig=-HUGE_VAL;
      for (i=j=numEvals; i<numEvals+primme0->initSize; i++) {
         lastMinEig = min(lastMinEig, evals[i]);
         lastMaxEig = max(lastMaxEig, evals[i]);
         if (primme->bounds[0] > evals[i]+rnorms[i] || evals[i]-rnorms[i] > primme->bounds[1]) continue;
         for (k=l=0; k<j; k++) {
            if (evals[perm[k]] < evals[i]) l = k+1;
            if (fabs(evals[k]-evals[i]) > rnorms[i]+rnorms[k]) continue;
            norm = REAL_PART(primme_dot(&evecs[primme->nLocal*i], &evecs[primme->nLocal*k], primme));
            if (fabs(norm) < max(rnorms[i],rnorms[k])) continue;
            if (lastMinEig == evals[i]) lastMinEig = evals[k];
            if (lastMaxEig == evals[i]) lastMaxEig = evals[k];
            break;
         }
         if (k >= j) {
            if (j != i) primme_copy(primme->nLocal, &evecs[primme->nLocal*i], &evecs[primme->nLocal*j]);
            evals[j] = evals[i];
            assert(perm[j] == j);
            for (k=j; k>l; k--) perm[k] = perm[k-1];
            perm[l] = j;
            j++;
         }
         else if (primme->printLevel >= 4) {
            printf("warning: discarded %g\n", evals[i]);
         }
      }
      /* None eigenvalue found means there is not more eigenvalues inside the bounds */
      if (j == numEvals) lastMinEig=bounds[0], lastMaxEig=bounds[1];

      if (primme->printLevel >= 4) 
         printf("SLICERES: %d numEvals %d from %g to %g discard %d\n", sliceNum, primme0->initSize, lastMinEig, lastMaxEig, primme0->initSize+numEvals-j);
      numEvals = j;
      for (i=0; i<numEvals-1; i++) assert(evals[perm[i]] <= evals[perm[i+1]]);
      primme->initSize = numEvals;

      if (ret != 0 && ret != -3 && ret > -100) {
         fprintf(primme->outputFile, 
                 "Error: dprimme returned with nonzero exit status %d\n", ret);
         return ret;
      }
   }

   return ret;
}

/* Tree Slice Strategy                                                         */
/* Slice are taken recursively: first, center; then left and right gaps.       */
static int slice_tree(double *lb, double *ub, double lastMinEig, double lastMaxEig, double *queue, int lqueue, int *qtop) {

   /* Initial case */
   if (*qtop <= -100) {
      *qtop = 0;
      queue[0] = *lb;
      queue[1] = *ub;
   }

   /* Regular case */
   else {
      const double qtopLeft = queue[*qtop], qtopRight = queue[*qtop+1], c = qtopLeft+qtopRight;
      int i;
      /* Generate right gap */
      if (qtopRight > lastMaxEig) {
         const double c0 = fabs(qtopRight + lastMaxEig - c);
         for (i=*qtop; i>1 && fabs(queue[i-2]+queue[i-1]-c)<c0; i-=2)
            queue[i] = queue[i-2], queue[i+1] = queue[i-1];
         queue[i] = lastMaxEig;
         queue[i+1] = qtopRight;
         *qtop += 2;
      }
      /* Generate left gap */
      if (qtopLeft < lastMinEig) {
         const double c0 = fabs(qtopLeft + lastMinEig - c);
         for (i=*qtop; i>1 && fabs(queue[i-2]+queue[i-1]-c)<c0; i-=2)
            queue[i] = queue[i-2], queue[i+1] = queue[i-1];
         queue[i] = qtopLeft;
         queue[i+1] = lastMinEig;
         *qtop += 2;
      }
      *qtop -= 2;
   }
   assert(*qtop < lqueue && *qtop%2 == 0);

   /* End case */
   if (*qtop < 0) return 0;

   *lb = queue[*qtop];
   *ub = queue[*qtop+1];
   return 1;
}
