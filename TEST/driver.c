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
 *
 * --------------------------------------------------------------------------
 *
 *  Parallel driver for dprimme. Calling format:
 *
 *             par_dprimme DriverConfigFileName SolverConfigFileName
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
 *  SolverConfigFileName  includes all dprimme required information
 *                            as stored in primme data structure.
 *
 *             Example files: FullConf  Full customization of primme
 *                            LeanConf  Use a preset method and some customization
 *                            MinConf   Provide ONLY a preset method and numEvals.
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
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


int real_main (int argc, char *argv[]);
static int setMatrixAndPrecond(driver_params *driver, primme_params *primme);
#ifdef USE_MPI
static void broadCast(primme_params *primme, primme_preset_method *method, 
   driver_params *driver, int master, MPI_Comm comm);
static void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, 
                         primme_params *primme);
#endif


int main (int argc, char *argv[]) {
   int ret;
#if defined(USE_PETSC)
   PetscInt ierr;
#endif

#if defined(USE_MPI) && !defined(USE_PETSC)
   MPI_Init(&argc, &argv);
#elif defined(USE_PETSC)
   PetscInitialize(&argc, &argv, NULL, NULL);
#endif

   ret = real_main(argc, argv);

#if defined(USE_MPI) && !defined(USE_PETSC)
   if (ret >= 0) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
   }
   else
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#elif defined(USE_PETSC)
   ierr = PetscFinalize(); CHKERRQ(ierr);
#endif

   return ret;
}

/******************************************************************************/
int real_main (int argc, char *argv[]) {

   /* Timing vars */
   double ut1,ut2,st1,st2,wt1,wt2;

   /* Files */
   char *DriverConfigFileName, *SolverConfigFileName;
   
   /* Driver and solver I/O arrays and parameters */
   double *evals, *evecs, *rnorms;
   driver_params driver;
   primme_params primme;
   primme_preset_method method;

   /* Other miscellaneous items */
   int ret;
   int i;
   int master = 1;
   int procID = 0;
#ifdef USE_MPI
   MPI_Comm comm;
   int numProcs;
#endif

#ifdef USE_MPI
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &procID);
   comm = MPI_COMM_WORLD;
   master = (procID == 0);
#endif

   primme_initialize(&primme);

   if (master) {
      /* ------------------------------------------------------- */
      /* Get from command line the names for the 2 config files  */
      /* ------------------------------------------------------- */
   
      if (argc >= 3) {
         DriverConfigFileName = argv[1];
         SolverConfigFileName = argv[2];
      }
      else {
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
      /* Read in the primme configuration file   */
      /* --------------------------------------- */
      if (read_solver_params(SolverConfigFileName, driver.outputFileName, 
                           &primme, &method) < 0) {
         fprintf(stderr, "Reading solver parameters failed\n");
         return(-1);
      }
   }

#ifdef USE_MPI
   /* ------------------------------------------------- */
   // Send read common primme members to all processors
   // Setup the primme members local to this processor  
   /* ------------------------------------------------- */
   broadCast(&primme, &method, &driver, master, comm);
#endif

   /* --------------------------------------- */
   /* Set up matrix vector and preconditioner */
   /* --------------------------------------- */
   if (setMatrixAndPrecond(&driver, &primme) != 0) return -1;

   /* --------------------------------------- */
   /* Pick one of the default methods(if set) */
   /* --------------------------------------- */
   if (primme_set_method(method, &primme) < 0 ) {
      fprintf(primme.outputFile, "No preset method. Using custom settings\n");
   }

   /* --------------------------------------- */
   /* Optional: report memory requirements    */
   /* --------------------------------------- */

   ret = dprimme(NULL,NULL,NULL,&primme);
   if (master) {
      fprintf(primme.outputFile,"PRIMME will allocate the following memory:\n");
      fprintf(primme.outputFile," processor %d, real workspace, %ld bytes\n",
                                      procID, primme.realWorkSize);
      fprintf(primme.outputFile," processor %d, int  workspace, %d bytes\n",
                                      procID, primme.intWorkSize);
   }

   /* --------------------------------------- */
   /* Display given parameter configuration   */
   /* Place this after the dprimme() to see   */
   /* any changes dprimme() made to primme    */
   /* --------------------------------------- */

   if (master) {
      fprintf(primme.outputFile," Matrix: %s\n",driver.matrixFileName);
      primme_display_params(primme);
   }

   /* --------------------------------------------------------------------- */
   /*                            Run the dprimme solver                           */
   /* --------------------------------------------------------------------- */

   /* Allocate space for converged Ritz values and residual norms */

   evals = (double *)primme_calloc(primme.numEvals, sizeof(double), "evals");
   evecs = (double *)primme_calloc(primme.nLocal*primme.numEvals, 
                                sizeof(double), "evecs");
   rnorms = (double *)primme_calloc(primme.numEvals, sizeof(double), "rnorms");

   /* ------------------------ */
   /* Initial guess (optional) */
   /* ------------------------ */
   for (i=0;i<primme.nLocal;i++) evecs[i]=1/sqrt(primme.n);

   /* ------------- */
   /*  Call primme  */
   /* ------------- */

   wt1 = primme_get_wtime(); 
   primme_get_time(&ut1,&st1);

   ret = dprimme(evals, evecs, rnorms, &primme);

   wt2 = primme_get_wtime();
   primme_get_time(&ut2,&st2);

   /* --------------------------------------------------------------------- */
   /* Reporting                                                             */
   /* --------------------------------------------------------------------- */

   if (master) {
      primme_PrintStackTrace(primme);
   }

   fprintf(primme.outputFile, "Wallclock Runtime   : %-f\n", wt2-wt1);
   fprintf(primme.outputFile, "User Time           : %f seconds\n", ut2-ut1);
   fprintf(primme.outputFile, "Syst Time           : %f seconds\n", st2-st1);

   if (master) {
      for (i=0; i < primme.numEvals; i++) {
         fprintf(primme.outputFile, "Eval[%d]: %-22.15E rnorm: %-22.15E\n", i+1,
            evals[i], rnorms[i]); 
      }
      fprintf(primme.outputFile, " %d eigenpairs converged\n", primme.initSize);

      fprintf(primme.outputFile, "Tolerance : %-22.15E\n", 
                                                            primme.aNorm*primme.eps);
      fprintf(primme.outputFile, "Iterations: %-d\n", 
                                                    primme.stats.numOuterIterations); 
      fprintf(primme.outputFile, "Restarts  : %-d\n", primme.stats.numRestarts);
      fprintf(primme.outputFile, "Matvecs   : %-d\n", primme.stats.numMatvecs);
      fprintf(primme.outputFile, "Preconds  : %-d\n", primme.stats.numPreconds);
      if (primme.locking && primme.intWork && primme.intWork[0] == 1) {
         fprintf(primme.outputFile, "\nA locking problem has occurred.\n");
         fprintf(primme.outputFile,
            "Some eigenpairs do not have a residual norm less than the tolerance.\n");
         fprintf(primme.outputFile,
            "However, the subspace of evecs is accurate to the required tolerance.\n");
      }

      fprintf(primme.outputFile, "\n\n#,%d,%.1f\n\n", primme.stats.numMatvecs,
         wt2-wt1); 

      switch (primme.dynamicMethodSwitch) {
         case -1: fprintf(primme.outputFile,
               "Recommended method for next run: DEFAULT_MIN_MATVECS\n"); break;
         case -2: fprintf(primme.outputFile,
               "Recommended method for next run: DEFAULT_MIN_TIME\n"); break;
         case -3: fprintf(primme.outputFile,
               "Recommended method for next run: DYNAMIC (close call)\n"); break;
      }
   }

   if (ret != 0 && master) {
      fprintf(primme.outputFile, 
         "Error: dprimme returned with nonzero exit status: %d \n",ret);
      return -1;
   }


   fflush(primme.outputFile);
   fclose(primme.outputFile);
   primme_Free(&primme);

   fflush(stdout);
   fflush(stderr);
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
   MPI_Bcast(&driver->matrixChoice, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->PrecChoice, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->isymm, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->level, 1, MPI_INT, 0, comm);
   MPI_Bcast(&driver->threshold, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->filter, 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&driver->shift, 1, MPI_DOUBLE, 0, comm);

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

static int setMatrixAndPrecond(driver_params *driver, primme_params *primme) {
#if defined(USE_MPI)
   primme->commInfo = (MPI_Comm *)primme_calloc(1, sizeof(MPI_Comm), "MPI_Comm");
#endif
   switch(driver->matrixChoice) {
   case driver_native:
#if !defined(USE_NATIVE)
      fprintf(stderr, "ERROR: NATIVE is needed!\n");
      return -1;
#else
#  if defined(USE_MPI)
      {
         int numProcs;
         
         MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
         if (numProcs != 1) {
            fprintf(stderr, "ERROR: MPI is not supported with NATIVE, use other!\n");
            return -1;
         }
         *(MPI_Comm*)primme->commInfo = MPI_COMM_WORLD;
      }
#  endif
      {
         CSRMatrix *matrix, *prec;
         double *diag;
         
         if (readMatrixNative(driver->matrixFileName, &matrix, &primme->aNorm) !=0 )
            return -1;
         primme->matrix = matrix;
         primme->matrixMatvec = CSRMatrixMatvec;
         primme->n = primme->nLocal = matrix->n;
         switch(driver->PrecChoice) {
         case driver_noprecond:
            primme->preconditioner = NULL;
            primme->applyPreconditioner = NULL;
            break;
         case driver_jacobi:
            createInvDiagPrecNative(matrix, driver->shift, &diag);
            primme->preconditioner = diag;
            primme->applyPreconditioner = ApplyInvDiagPrecNative;
            break;
         case driver_jacobi_i:
            createInvDavidsonDiagPrecNative(matrix, &diag);
            primme->preconditioner = diag;
            primme->applyPreconditioner = ApplyInvDavidsonDiagPrecNative;
         case driver_ilut:
            createILUTPrecNative(matrix, driver->shift, driver->level, driver->threshold,
                                 driver->filter, &prec);
            primme->preconditioner = prec;
            primme->applyPreconditioner = ApplyILUTPrecNative;
            break;
         }
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
         readMatrixAndPrecondParaSails(driver->matrixFileName, driver->shift, driver->level,
               driver->threshold, driver->filter, driver->isymm, MPI_COMM_WORLD, &primme->aNorm,
               &primme->n, &primme->nLocal, &primme->numProcs, &primme->procID, &matrix,
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
         Mat *matrix;
         PC *pc;
         Vec *vec;
         if (readMatrixPetsc(driver->matrixFileName, &primme->n, &primme->nLocal,
                         &primme->numProcs, &primme->procID, &matrix, &primme->aNorm, 0) != 0)
            return -1;
         *(MPI_Comm*)primme->commInfo = PETSC_COMM_WORLD;
         primme->matrix = matrix;
         primme->matrixMatvec = PETScMatvec;
         if (driver->PrecChoice == driver_noprecond) {
            primme->preconditioner = NULL;
            primme->applyPreconditioner = NULL;
         }
         else if (driver->PrecChoice != driver_jacobi_i) {
            pc = (PC *)primme_calloc(1, sizeof(PC), "pc");
            ierr = PCCreate(PETSC_COMM_WORLD, pc); CHKERRQ(ierr);
            if (driver->PrecChoice == driver_jacobi) {
               ierr = PCSetType(*pc, PCJACOBI); CHKERRQ(ierr);
            }
            else if (driver->PrecChoice == driver_ilut) {
               ierr = PCSetType(*pc, PCICC); CHKERRQ(ierr);
            }
            ierr = PCSetOperators(*pc, *matrix, *matrix); CHKERRQ(ierr);
            ierr = PCSetFromOptions(*pc); CHKERRQ(ierr);
            ierr = PCSetUp(*pc); CHKERRQ(ierr);
            primme->preconditioner = pc;
            primme->applyPreconditioner = ApplyPCPrecPETSC;
         }
         else {
            vec = (Vec *)primme_calloc(1, sizeof(Vec), "Vec preconditioner");
            ierr = MatCreateVecs(*matrix, vec, NULL); CHKERRQ(ierr);
            ierr = MatGetDiagonal(*matrix, *vec); CHKERRQ(ierr);
            primme->preconditioner = vec;
            primme->applyPreconditioner = ApplyPCPrecPETSC;
         }
      }
#endif
      break;
   }

#if defined(USE_MPI)
   primme->globalSumDouble = par_GlobalSumDouble;
#endif
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

