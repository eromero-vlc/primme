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
 * File: shared_utils.c
 * 
 * Purpose - Functions to read and print driver_params and primme_params.
 * 
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include "shared_utils.h"


/******************************************************************************
 *
 * Reads the solver parameters for configuring the primme structure of 
 * dprimme(). "method" is not part of the struct primme, but if specified
 * primme will be setup accordingly by primme_set_method().
 *
******************************************************************************/
int read_solver_params(char *configFileName, char *outputFileName,
                primme_params *primme, const char* primmeprefix,
                primme_preset_method *method, const char* methodstr) {

   int line, ret, i;
   char ident[2048], *field;
   char op[128];
   char stringValue[128];
   FILE *configFile;

   if ((configFile = fopen(configFileName, "r")) == NULL) {
      fprintf(stderr,"Error(read_solver_params): Could not open config file\n");
      fprintf(stderr, "config file: %s\n", configFileName);
      return(-1);
   }

   line = 1;
   while (EOF != fscanf(configFile, "%s", ident)) {
      if (strncmp(ident, "//", 2) == 0) {
         if (fgets(ident, 2048, configFile) == NULL) {
            break;
         }
         line++;
         continue;
      }
      else {
         ret = fscanf(configFile, "%s", op);
      }

      if (ret == EOF) {
         fprintf(stderr, "ERROR(read_solver_params): Unexpected end of file\n");
         return(-1);
      }

      if (strcmp(op, "=") == 0) {
         if (strcmp(ident, methodstr) == 0) {
            ret = fscanf(configFile, "%s", stringValue);
            if (ret == 1) {
               ret = 0;
               #define READ_METHOD(V) if (strcmp(stringValue, #V) == 0) {*method = V; ret=1;}
               READ_METHOD(DEFAULT_METHOD);
               READ_METHOD(DYNAMIC);
               READ_METHOD(DEFAULT_MIN_TIME);
               READ_METHOD(DEFAULT_MIN_MATVECS);
               READ_METHOD(Arnoldi);
               READ_METHOD(GD);
               READ_METHOD(GD_plusK);
               READ_METHOD(GD_Olsen_plusK);
               READ_METHOD(JD_Olsen_plusK);
               READ_METHOD(RQI);
               READ_METHOD(JDQR);
               READ_METHOD(JDQMR);
               READ_METHOD(JDQMR_ETol);
               READ_METHOD(SUBSPACE_ITERATION);
               READ_METHOD(LOBPCG_OrthoBasis);
               READ_METHOD(LOBPCG_OrthoBasis_Window);
               #undef READ_METHOD
            }
            if (ret == 0) {
               printf("Invalid %s value\n", methodstr);
               return -1;
            }
            line++;
            continue;
         }
         else if (strncmp(ident, primmeprefix, strlen(primmeprefix)) != 0) {
            if (fgets(ident, 2048, configFile) == NULL) {
               break;
            }
            line++;
            continue;
         }
         field = ident + strlen(primmeprefix);

         #define READ_FIELD(V, P) if (strcmp(field, #V) == 0) \
            ret = fscanf(configFile, #P, &primme-> V);
         #define READ_FIELD_OP(V, P) if (strcmp(field, #V) == 0) { \
            ret = fscanf(configFile, "%s", stringValue); \
            if (ret == 1) { ret=0; P } \
            if (ret == 0) printf("Invalid " #V " value\n"); \
         }
         #define OPTION(F, V) if (strcmp(stringValue, #V) == 0) { primme-> F = V; ret = 1; }
         #define READ_FIELDParams(S, V, P) if (strcmp(field, #S "." #V) == 0) \
            ret = fscanf(configFile, #P, &primme-> S ## Params . V);
         #define READ_FIELD_OPParams(S, V, P) if (strcmp(field, #S "." #V) == 0) { \
            ret = fscanf(configFile, "%s", stringValue); \
            if (ret == 1) { ret=0; P } \
            if (ret == 0) printf("Invalid " #S "." #V " value\n"); \
         }
         #define OPTIONParams(S, F, V) if (strcmp(stringValue, #V) == 0) { primme-> S ## Params . F = V; ret = 1; }
  
         READ_FIELD(printLevel, %d);
         READ_FIELD(numEvals, %d);
         READ_FIELD(aNorm, %le);
         READ_FIELD(eps, %le);
         READ_FIELD(maxBasisSize, %d);
         READ_FIELD(minRestartSize, %d);
         READ_FIELD(maxBlockSize, %d);
         READ_FIELD(maxOuterIterations, %d);
         READ_FIELD(maxMatvecs, %d);
         READ_FIELD_OP(target,
            OPTION(target, primme_smallest)
            OPTION(target, primme_largest)
            OPTION(target, primme_closest_geq)
            OPTION(target, primme_closest_leq)
            OPTION(target, primme_closest_abs)
         );
         READ_FIELD_OPParams(projection, projection,
            OPTIONParams(projection, projection, primme_proj_default)
            OPTIONParams(projection, projection, primme_proj_RR)
            OPTIONParams(projection, projection, primme_proj_refined)
            OPTIONParams(projection, projection, primme_proj_harmonic)
         );

         READ_FIELD_OP(initBasisMode,
            OPTION(initBasisMode, primme_init_default)
            OPTION(initBasisMode, primme_init_krylov)
            OPTION(initBasisMode, primme_init_random)
            OPTION(initBasisMode, primme_init_user)
         );

         READ_FIELD(numTargetShifts, %d);
         if (strcmp(field, "targetShifts") == 0) {
            ret = 1;
            if (primme->numTargetShifts > 0) {
               primme->targetShifts = (double *)primme_calloc(
                  primme->numTargetShifts, sizeof(double), "targetShifts");
               for (i=0; i<primme->numTargetShifts; i++) {
                  ret = fscanf(configFile, "%le", &primme->targetShifts[i]);
                  if (ret != 1) break;
               }
            }
            if (ret == 1) {
               if (fgets(ident, 2048, configFile) == NULL) {
                  break;
               }
            }
         }

         READ_FIELD(numBounds, %d);
         if (strcmp(field, "bounds") == 0) {
            ret = 1;
            if (primme->numBounds > 0) {
               primme->bounds = (double *)primme_calloc(
                  primme->numBounds, sizeof(double), "bounds");
               for (i=0; i<primme->numBounds; i++) {
                  ret = fscanf(configFile, "%le", &primme->bounds[i]);
                  if (ret != 1) break;
               }
            }
            if (ret == 1) {
               if (fgets(ident, 2048, configFile) == NULL) {
                  break;
               }
            }
         }
  
         READ_FIELD(dynamicMethodSwitch, %d);
         READ_FIELD(locking, %d);
         READ_FIELD(initSize, %d);
         READ_FIELD(numOrthoConst, %d);

         if (strcmp(field, "iseed") == 0) {
            ret = 1;
            for (i=0;i<4; i++) {
               ret = fscanf(configFile, "%d", &primme->iseed[i]);
               if (ret != 1) break;
            }
            if (ret == 1) {
               if (fgets(ident, 2048, configFile) == NULL) {
                  break;
               }
            }
         }

         READ_FIELD_OPParams(restarting, scheme,
            OPTIONParams(restarting, scheme, primme_thick)
            OPTIONParams(restarting, scheme, primme_dtr)
         );

         READ_FIELDParams(restarting, maxPrevRetain, %d);

         READ_FIELDParams(correction, precondition, %d);
         READ_FIELDParams(correction, robustShifts, %d);
         READ_FIELDParams(correction, maxInnerIterations, %d);
         READ_FIELDParams(correction, relTolBase, %lf);

         READ_FIELD_OPParams(correction, convTest,
            OPTIONParams(correction, convTest, primme_full_LTolerance)
            OPTIONParams(correction, convTest, primme_decreasing_LTolerance)
            OPTIONParams(correction, convTest, primme_adaptive_ETolerance)
            OPTIONParams(correction, convTest, primme_adaptive)
         );

         READ_FIELDParams(correction, projectors.LeftQ , %d);
         READ_FIELDParams(correction, projectors.LeftX , %d);
         READ_FIELDParams(correction, projectors.RightQ, %d);
         READ_FIELDParams(correction, projectors.SkewQ , %d);
         READ_FIELDParams(correction, projectors.RightX, %d);
         READ_FIELDParams(correction, projectors.SkewX , %d);

         READ_FIELD_OP(applyPrecondTo,
            OPTION(applyPrecondTo, primme_r)
            OPTION(applyPrecondTo, primme_x)
            OPTION(applyPrecondTo, primme_lastv)
         );

         if (ret == 0) {
            fprintf(stderr, 
               "ERROR(read_solver_params): Invalid parameter '%s'\n", ident);
            return(-1);
         }
         line++;

         #undef READ_FIELD
         #undef READ_FIELD_OP
         #undef OPTION
         #undef READ_FIELDParams
         #undef READ_FIELD_OPParams
         #undef OPTIONParams
      }
      else {
         fprintf(stderr, 
            "ERROR(read_solver_params): Invalid operator on %d\n", line);
         return(-1);
      }

      if (ret != 1) {
         fprintf(stderr, 
         "ERROR(read_solver_params): Could not read value on line %d\n", line);
         return(-1);
      }
   }

   /* Set up the output file in primme, from the filename read in driverConfig */
   if (primme->procID == 0) {
      if (outputFileName[0] && strcmp(outputFileName, "stdout") != 0) {
         if ((primme->outputFile = fopen(outputFileName, "w+")) == NULL) {
            fprintf(stderr, 
                   "ERROR(read_solver_params): Could not open output file\n");
         }
      }
      else {
         primme->outputFile = stdout;
      }
   }
   else {
      primme->outputFile = stdout;
   }

   fclose(configFile);
   return (0);
}

/******************************************************************************
 *
 * Reads the parameters necessary for the test driver
 * eg., matrix, preconditioning parameters and choice, output files, etc
 * This function does not read any solver parameters.
 *
******************************************************************************/
int read_driver_params(char *configFileName, driver_params *driver) {

   int line, ret;
   char ident[2048];
   char op[128];
   char stringValue[128];
   FILE *configFile;
   primme_params primme0;


   primme0 = driver->primme;
   memset(driver, 0, sizeof(*driver));
   driver->primme = primme0;
   if ((configFile = fopen(configFileName, "r")) == NULL) {
      fprintf(stderr,"Error(read_driver_params): Could not open config file\n");
      fprintf(stderr,"Driver config file: %s\n", configFileName);
      return(-1);
   }

   line = 1;
   while (EOF != fscanf(configFile, "%s", ident)) {
      if (strncmp(ident, "//", 2) == 0) {
         if (fgets(ident, 2048, configFile) == NULL) {
            break;
         }
         line++;
         continue;
      }
      else {
         ret = fscanf(configFile, "%s", op);
      }

      if (ret == EOF) {
         fprintf(stderr, "ERROR(read_driver_params): Unexpected end of file\n");
         return(-1);
      }

      if (strcmp(op, "=") == 0) {
         /* Matrix, partitioning and I/O params  */
         if (strcmp(ident, "driver.outputFile") == 0) {
            ret = fscanf(configFile, "%s", driver->outputFileName);
         }
         else if (strcmp(ident, "driver.partId") == 0) {
            ret = fscanf(configFile, "%s", driver->partId);
         }
         else if (strcmp(ident, "driver.partDir") == 0) {
            ret = fscanf(configFile, "%s", driver->partDir);
         }
         else if (strcmp(ident, "driver.matrixFile") == 0) {
            ret = fscanf(configFile, "%s", driver->matrixFileName);
         }
         else if (strcmp(ident, "driver.initialGuessesFile") == 0) {
            ret = fscanf(configFile, "%s", driver->initialGuessesFileName);
         }
         else if (strcmp(ident, "driver.initialGuessesPert") == 0) {
            ret = fscanf(configFile, "%le", &driver->initialGuessesPert);
         }
         else if (strcmp(ident, "driver.saveXFile") == 0) {
            ret = fscanf(configFile, "%s", driver->saveXFileName);
         }
         else if (strcmp(ident, "driver.checkXFile") == 0) {
            ret = fscanf(configFile, "%s", driver->checkXFileName);
         }
         else if (strcmp(ident, "driver.matrixChoice") == 0) {
            ret = fscanf(configFile, "%s", stringValue);
            if (ret == 1) {
               if (strcmp(stringValue, "default") == 0) {
                  driver->matrixChoice = driver_default;
               }
               else if (strcmp(stringValue, "native") == 0) {
                  driver->matrixChoice = driver_native;
               }
               else if (strcmp(stringValue, "parasails") == 0) {
                  driver->matrixChoice = driver_parasails;
               }
               else if (strcmp(stringValue, "petsc") == 0) {
                  driver->matrixChoice = driver_petsc;
               }
               else if (strcmp(stringValue, "rsb") == 0) {
                  driver->matrixChoice = driver_rsb;
               }
               else {
                  fprintf(stderr, 
                     "ERROR(read_driver_params): Invalid parameter '%s'\n", ident);
               }
            }
         }
         /* Preconditioning parameters */
         else if (strcmp(ident, "driver.PrecChoice") == 0) {
            ret = fscanf(configFile, "%s", stringValue);
            if (ret == 1) {
               if (strcmp(stringValue, "noprecond") == 0) {
                  driver->PrecChoice = driver_noprecond;
               }
               else if (strcmp(stringValue, "jacobi") == 0) {
                  driver->PrecChoice = driver_jacobi;
               }
               else if (strcmp(stringValue, "davidsonjacobi") == 0) {
                  driver->PrecChoice = driver_jacobi_i;
               }
               else if (strcmp(stringValue, "ilut") == 0) {
                  driver->PrecChoice = driver_ilut;
               }
               else {
                  fprintf(stderr, 
                     "ERROR(read_driver_params): Invalid parameter '%s'\n", ident);
               }
            }
         }
         else if (strcmp(ident, "driver.shift") == 0) {
            ret = fscanf(configFile, "%le", &driver->shift);
         }
         else if (strcmp(ident, "driver.isymm") == 0) {
            ret = fscanf(configFile, "%d", &driver->isymm);
         }
         else if (strcmp(ident, "driver.level") == 0) {
            ret = fscanf(configFile, "%d", &driver->level);
         }
         else if (strcmp(ident, "driver.threshold") == 0) {
            ret = fscanf(configFile, "%lf", &driver->threshold);
         }
         else if (strcmp(ident, "driver.filter") == 0) {
            ret = fscanf(configFile, "%lf", &driver->filter);
         }
         // Filter options
         else if (strcmp(ident, "driver.minEig") == 0) {
            ret = fscanf(configFile, "%lf", &driver->minEig);
         }
         else if (strcmp(ident, "driver.maxEig") == 0) {
            ret = fscanf(configFile, "%lf", &driver->maxEig);
         }
         else if (strcmp(ident, "driver.PrecFilter") == 0) {
            ret = fscanf(configFile, "%d", &driver->precFilter.filter);
         }
         else if (strcmp(ident, "driver.PrecFilterDegrees") == 0) {
            ret = fscanf(configFile, "%d", &driver->precFilter.degrees);
         }
         else if (strcmp(ident, "driver.PrecFilterLowerBound") == 0) {
            ret = fscanf(configFile, "%d", &driver->precFilter.lowerBound);
         }
         else if (strcmp(ident, "driver.PrecFilterUpperBound") == 0) {
            ret = fscanf(configFile, "%d", &driver->precFilter.upperBound);
         }
         else if (strcmp(ident, "driver.PrecFilterLBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->precFilter.lowerBoundFix);
         }
         else if (strcmp(ident, "driver.PrecFilterUBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->precFilter.upperBoundFix);
         }
         else if (strcmp(ident, "driver.PrecFilterCheckEps") == 0) {
            ret = fscanf(configFile, "%lf", &driver->precFilter.checkEps);
         }
         else if (strcmp(ident, "driver.AFilter") == 0) {
            ret = fscanf(configFile, "%d", &driver->AFilter.filter);
         }
         else if (strcmp(ident, "driver.AFilterDegrees") == 0) {
            ret = fscanf(configFile, "%d", &driver->AFilter.degrees);
         }
         else if (strcmp(ident, "driver.AFilterLowerBound") == 0) {
            ret = fscanf(configFile, "%d", &driver->AFilter.lowerBound);
         }
         else if (strcmp(ident, "driver.AFilterUpperBound") == 0) {
            ret = fscanf(configFile, "%d", &driver->AFilter.upperBound);
         }
         else if (strcmp(ident, "driver.AFilterLBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->AFilter.lowerBoundFix);
         }
         else if (strcmp(ident, "driver.AFilterUBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->AFilter.upperBoundFix);
         }
         else if (strcmp(ident, "driver.AFilterCheckEps") == 0) {
            ret = fscanf(configFile, "%lf", &driver->AFilter.checkEps);
         }
         else if (strcmp(ident, "driver.Transform") == 0) {
            ret = fscanf(configFile, "%d", &driver->transform.filter);
         }
         else if (strcmp(ident, "driver.TransformLBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->transform.lowerBoundFix);
         }
         else if (strcmp(ident, "driver.TransformUBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->transform.upperBoundFix);
         }
          else if (strcmp(ident, "driver.OrthoFilter") == 0) {
            ret = fscanf(configFile, "%d", &driver->orthoFilter.filter);
         }
         else if (strcmp(ident, "driver.OrthoFilterDegrees") == 0) {
            ret = fscanf(configFile, "%d", &driver->orthoFilter.degrees);
         }
         else if (strcmp(ident, "driver.OrthoFilterLowerBound") == 0) {
            ret = fscanf(configFile, "%d", &driver->orthoFilter.lowerBound);
         }
         else if (strcmp(ident, "driver.OrthoFilterUpperBound") == 0) {
            ret = fscanf(configFile, "%d", &driver->orthoFilter.upperBound);
         }
         else if (strcmp(ident, "driver.OrthoFilterLBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->orthoFilter.lowerBoundFix);
         }
         else if (strcmp(ident, "driver.OrthoFilterUBFix") == 0) {
            ret = fscanf(configFile, "%lf", &driver->orthoFilter.upperBoundFix);
         }
         else if (strcmp(ident, "driver.maxLocked") == 0) {
            ret = fscanf(configFile, "%d", &driver->maxLocked);
         }
         else if (strncmp(ident, "driver.", 7) == 0) {
            fprintf(stderr, 
              "ERROR(read_driver_params): Invalid parameter '%s'\n", ident);
            return(-1);
         }
         else {
            if (fgets(ident, 2048, configFile) == NULL) {
               break;
            }
         }

         line++;
      }
      else {
         fprintf(stderr, "ERROR(read_driver_params): Invalid operator on %d\n",
                 line);
         return(-1);
      }

      if (ret != 1) {
         fprintf(stderr, 
          "ERROR(read_driver_params): Could not read value on line %d\n", line);
         return(-1);
      }
   }
   fclose(configFile);
   return (0);
}

void driver_display_params(driver_params driver, FILE *outputFile) {

char *helpFilter[] = {"no filter",
                      "Chebyshev no damping",
                      "Chebyshev Jackson damping",
                      "Chebyshev sigma-Lanczos damping",
                      "Chebyshev acceleration low-pass",
                      "Chebyshev acceleration high-pass",
                      "Conjugate residual filter from FILTLAN",
                      "Chebyshev acceleration interior",
                      "Chebyshev delta no damping",
                      "Chebyshev delta Jackson damping",
                      "Chebyshev delta sigma-Lanczos damping",
                      "Chebyshev based on (A-shift I)^2",
                      "FEAST",
                      "Augmented matrix [0 A';A 0]",
                      "Normal equation (A - shift*I)^2",
                      "gamma5*(lb*I + ub*A)",
                      "[gamma5*(lb*I + ub*A)]^2"};
char *helpBoundFilter[] = {"extreme eigenvalue",
                           "last converged",
                           "last frozen",
                           "target Ritz val[0]",
                           "target Ritz val[0] - err",
                           "target Ritz val[blockSize-1]",
                           "target Ritz val[blockSize-1]+err",
                           "median of Ritz vals",
                           "fix value"};
const char *strPrecChoice[] = {"noprecond", "jacobi", "davidsonjacobi", "ilut"};
const char *strMatrixChoice[] = {"default", "native", "petsc", "parasails", "rsb"};
 
fprintf(outputFile, "// ---------------------------------------------------\n"
                    "//                 driver configuration               \n"
                    "// ---------------------------------------------------\n");

fprintf(outputFile, "driver.partId        = %s\n", driver.partId);
fprintf(outputFile, "driver.partDir       = %s\n", driver.partDir);
fprintf(outputFile, "driver.matrixFile    = %s\n", driver.matrixFileName);
fprintf(outputFile, "driver.matrixChoice  = %s\n", strMatrixChoice[driver.matrixChoice]);
fprintf(outputFile, "driver.initialGuessesFile = %s\n", driver.initialGuessesFileName);
fprintf(outputFile, "driver.initialGuessesPert = %e\n", driver.initialGuessesPert);
fprintf(outputFile, "driver.saveXFile     = %s\n", driver.saveXFileName);
fprintf(outputFile, "driver.checkXFile    = %s\n", driver.checkXFileName);
fprintf(outputFile, "driver.PrecChoice    = %s\n", strPrecChoice[driver.PrecChoice]);
fprintf(outputFile, "driver.shift         = %e\n", driver.shift);
fprintf(outputFile, "driver.isymm         = %d\n", driver.isymm);
fprintf(outputFile, "driver.level         = %d\n", driver.level);
fprintf(outputFile, "driver.threshold     = %f\n", driver.threshold);
fprintf(outputFile, "driver.filter        = %f\n\n", driver.filter);
fprintf(outputFile, "driver.minEig        = %lf\n", driver.minEig);
fprintf(outputFile, "driver.maxEig        = %lf\n\n", driver.maxEig);

fprintf(outputFile, "driver.Transform          = %d   // %s\n", driver.transform.filter,
         helpFilter[driver.transform.filter]);
fprintf(outputFile, "driver.TransformLBFix     = %lf\n", driver.transform.lowerBoundFix);
fprintf(outputFile, "driver.TransformUBFix     = %lf\n", driver.transform.upperBoundFix);

fprintf(outputFile, "driver.PrecFilter            = %d   // %s\n", driver.precFilter.filter,
         helpFilter[driver.precFilter.filter]);
fprintf(outputFile, "driver.PrecFilterDegrees     = %d\n", driver.precFilter.degrees);
fprintf(outputFile, "driver.PrecFilterLowerBound  = %d   // %s\n", driver.precFilter.lowerBound,
         helpBoundFilter[driver.precFilter.lowerBound]);
fprintf(outputFile, "driver.PrecFilterUpperBound  = %d   // %s\n", driver.precFilter.upperBound,
         helpBoundFilter[driver.precFilter.upperBound]);
fprintf(outputFile, "driver.PrecFilterLBFix       = %lf\n", driver.precFilter.lowerBoundFix);
fprintf(outputFile, "driver.PrecFilterUBFix       = %lf\n", driver.precFilter.upperBoundFix);
fprintf(outputFile, "driver.PrecFilterCheckEps    = %lf\n\n", driver.precFilter.checkEps);

fprintf(outputFile, "driver.AFilter            = %d   // %s\n", driver.AFilter.filter,
         helpFilter[driver.AFilter.filter]);
fprintf(outputFile, "driver.AFilterDegrees     = %d\n", driver.AFilter.degrees);
fprintf(outputFile, "driver.AFilterLowerBound  = %d   // %s\n", driver.AFilter.lowerBound,
         helpBoundFilter[driver.AFilter.lowerBound]);
fprintf(outputFile, "driver.AFilterUpperBound  = %d   // %s\n", driver.AFilter.upperBound,
         helpBoundFilter[driver.AFilter.upperBound]);
fprintf(outputFile, "driver.AFilterLBFix       = %lf\n", driver.AFilter.lowerBoundFix);
fprintf(outputFile, "driver.AFilterUBFix       = %lf\n", driver.AFilter.upperBoundFix);
fprintf(outputFile, "driver.AFilterCheckEps    = %lf\n\n", driver.AFilter.checkEps);

fprintf(outputFile, "driver.OrthoFilter            = %d   // %s\n", driver.orthoFilter.filter,
         helpFilter[driver.orthoFilter.filter]);
fprintf(outputFile, "driver.OrthoFilterDegrees     = %d\n", driver.orthoFilter.degrees);
fprintf(outputFile, "driver.OrthoFilterLowerBound  = %d   // %s\n", driver.orthoFilter.lowerBound,
         helpBoundFilter[driver.orthoFilter.lowerBound]);
fprintf(outputFile, "driver.OrthoFilterUpperBound  = %d   // %s\n", driver.orthoFilter.upperBound,
         helpBoundFilter[driver.orthoFilter.upperBound]);
fprintf(outputFile, "driver.OrthoFilterLBFix       = %lf\n", driver.orthoFilter.lowerBoundFix);
fprintf(outputFile, "driver.OrthoFilterUBFix       = %lf\n", driver.orthoFilter.upperBoundFix);
fprintf(outputFile, "driver.maxLocked          = %d\n", driver.maxLocked);
}

void driver_display_method(primme_preset_method method, const char* methodstr, FILE *outputFile) {

   const char *strMethod[] = {
      "DEFAULT_METHOD",
      "DYNAMIC",
      "DEFAULT_MIN_TIME",
      "DEFAULT_MIN_MATVECS",
      "Arnoldi",
      "GD",
      "GD_plusK",
      "GD_Olsen_plusK",
      "JD_Olsen_plusK",
      "RQI",
      "JDQR",
      "JDQMR",
      "JDQMR_ETol",
      "SUBSPACE_ITERATION",
      "LOBPCG_OrthoBasis",
      "LOBPCG_OrthoBasis_Window"};

   fprintf(outputFile, "%s               = %s\n", methodstr, strMethod[method]);

}

void driver_display_methodsvd(primme_svds_preset_method method, const char* methodstr, FILE *outputFile) {

   const char *strMethod[] = {
      "primme_svds_default",
      "primme_svds_hybrid",
      "primme_svds_normalequations",
      "primme_svds_augmented"};

   fprintf(outputFile, "%s               = %s\n", methodstr, strMethod[method]);

}


int read_solver_params_svds(char *configFileName, char *outputFileName,
                primme_svds_params *primme_svds, const char* primmeprefix,
                primme_svds_preset_method *method, const char* methodstr,
                primme_preset_method *primme_method,
                primme_preset_method *primme_methodStage2) {

   int line, ret, i;
   char ident[2048], *field;
   char op[128];
   char stringValue[128];
   FILE *configFile;

   if ((configFile = fopen(configFileName, "r")) == NULL) {
      fprintf(stderr,"Error(read_solver_params_svds): Could not open config file\n");
      fprintf(stderr, "config file: %s\n", configFileName);
      return(-1);
   }

   line = 1;
   while (EOF != fscanf(configFile, "%s", ident)) {
      if (strncmp(ident, "//", 2) == 0) {
         if (fgets(ident, 2048, configFile) == NULL) {
            break;
         }
         line++;
         continue;
      }
      else {
         ret = fscanf(configFile, "%s", op);
      }

      if (ret == EOF) {
         fprintf(stderr, "ERROR(read_solver_params_svds): Unexpected end of file\n");
         return(-1);
      }

      if (strcmp(op, "=") == 0) {
         if (strcmp(ident, methodstr) == 0) {
            ret = fscanf(configFile, "%s", stringValue);
            if (ret == 1) {
               ret = 0;
               #define READ_METHOD(V) if (strcmp(stringValue, #V) == 0) {*method = V; ret=1;}
               READ_METHOD(primme_svds_default);
               READ_METHOD(primme_svds_hybrid);
               READ_METHOD(primme_svds_normalequations);
               READ_METHOD(primme_svds_augmented);
               #undef READ_METHOD
            }
            if (ret == 0) {
               printf("Invalid %s value\n", methodstr);
               return -1;
            }
            line++;
            continue;
         }
         else if (strncmp(ident, primmeprefix, strlen(primmeprefix)) != 0) {
            if (fgets(ident, 2048, configFile) == NULL) {
               break;
            }
            line++;
            continue;
         }
         field = ident + strlen(primmeprefix);

         #define READ_FIELD(V, P) if (strcmp(field, #V) == 0) \
            ret = fscanf(configFile, #P, &primme_svds-> V);
         #define READ_FIELD_OP(V, P) if (strcmp(field, #V) == 0) { \
            ret = fscanf(configFile, "%s", stringValue); \
            if (ret == 1) { ret=0; P } \
            if (ret == 0) printf("Invalid " #V " value\n"); \
         }
         #define OPTION(F, V) if (strcmp(stringValue, #V) == 0) { primme_svds-> F = V; ret = 1; }
  
         READ_FIELD(printLevel, %d);
         READ_FIELD(numSvals, %d);
         READ_FIELD(aNorm, %le);
         READ_FIELD(eps, %le);
         READ_FIELD(maxBasisSize, %d);
         READ_FIELD(maxBlockSize, %d);
         READ_FIELD(maxMatvecs, %d);

         READ_FIELD_OP(target,
            OPTION(target, primme_svds_smallest)
            OPTION(target, primme_svds_largest)
            OPTION(target, primme_svds_closest_abs)
         );

         READ_FIELD(numTargetShifts, %d);
         if (strcmp(field, "targetShifts") == 0) {
            ret = 1;
            if (primme_svds->numTargetShifts > 0) {
               primme_svds->targetShifts = (double *)primme_calloc(
                  primme_svds->numTargetShifts, sizeof(double), "targetShifts");
               for (i=0; i<primme_svds->numTargetShifts; i++) {
                  ret = fscanf(configFile, "%le", &primme_svds->targetShifts[i]);
                  if (ret != 1) break;
               }
            }
            if (ret == 1) {
               if (fgets(ident, 2048, configFile) == NULL) {
                  break;
               }
            }
         }
 
         READ_FIELD(locking, %d);
         READ_FIELD(initSize, %d);
         READ_FIELD(numOrthoConst, %d);

         if (strcmp(field, "iseed") == 0) {
            ret = 1;
            for (i=0;i<4; i++) {
               ret = fscanf(configFile, "%d", &primme_svds->iseed[i]);
               if (ret != 1) break;
            }
            if (ret == 1) {
               if (fgets(ident, 2048, configFile) == NULL) {
                  break;
               }
            }
         }

         READ_FIELD(precondition, %d);

         READ_FIELD_OP(method,
            OPTION(method, primme_svds_op_none)
            OPTION(method, primme_svds_op_AtA)
            OPTION(method, primme_svds_op_AAt)
            OPTION(method, primme_svds_op_augmented)
         );

         READ_FIELD_OP(methodStage2,
            OPTION(method, primme_svds_op_none)
            OPTION(method, primme_svds_op_AtA)
            OPTION(method, primme_svds_op_AAt)
            OPTION(method, primme_svds_op_augmented)
         );

         if (ret == 0) {
            fprintf(stderr, 
               "ERROR(read_solver_params_svds): Invalid parameter '%s'\n", ident);
            return(-1);
         }
         line++;

         #undef READ_FIELD
         #undef READ_FIELD_OP
         #undef OPTION
       }
      else {
         fprintf(stderr, 
            "ERROR(read_solver_params_svds): Invalid operator on %d\n", line);
         return(-1);
      }

      if (ret != 1) {
         fprintf(stderr, 
         "ERROR(read_solver_params_svds): Could not read value on line %d\n", line);
         return(-1);
      }
   }

   /* Set up the output file in primme_svds, from the filename read in driverConfig */
   if (primme_svds->procID == 0) {
      if (outputFileName[0] && strcmp(outputFileName, "stdout") != 0) {
         if ((primme_svds->outputFile = fopen(outputFileName, "w+")) == NULL) {
            fprintf(stderr, 
                   "ERROR(read_solver_params_svds): Could not open output file\n");
         }
      }
      else {
         primme_svds->outputFile = stdout;
      }
   }
   else {
      primme_svds->outputFile = stdout;
   }

   fclose(configFile);

   read_solver_params(configFileName, outputFileName, &primme_svds->primme,
                      "primme.", primme_method, "primme.method");
   read_solver_params(configFileName, outputFileName, &primme_svds->primmeStage2,
                      "primmeStage2.", primme_methodStage2, "primme.methodStage2");

   return (0);
}

#ifdef USE_MPI

/******************************************************************************
 * Function to broadcast the primme data structure to all processors
 *
 * EXCEPTIONS: procID and seed[] are not copied from processor 0. 
 *             Each process creates their own.
******************************************************************************/
void broadCast(primme_params *primme, primme_preset_method *method, 
   driver_params *driver, int master, MPI_Comm comm){

   int i;

   if (driver) {
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
   }

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
   MPI_Bcast(&(primme->initBasisMode), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme->applyPrecondTo), 1, MPI_INT, 0, comm);

   MPI_Bcast(&(primme->projectionParams.projection), 1, MPI_INT, 0, comm);
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

/******************************************************************************
 * Function to broadcast the primme svds data structure to all processors
 *
 * EXCEPTIONS: procID and seed[] are not copied from processor 0. 
 *             Each process creates their own.
******************************************************************************/
void broadCast_svds(primme_svds_params *primme_svds, primme_svds_preset_method *method,
   primme_preset_method *primmemethod, primme_preset_method *primmemethodStage2,
   driver_params *driver, int master, MPI_Comm comm){

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

   MPI_Bcast(&(primme_svds->numSvals), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->target), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->numTargetShifts), 1, MPI_INT, 0, comm);

   if (primme_svds->numTargetShifts > 0 && !master) {
      primme_svds->targetShifts = (double *)primme_calloc(
         primme_svds->numTargetShifts, sizeof(double), "targetShifts");
   }
   MPI_Bcast(primme_svds->targetShifts, primme_svds->numTargetShifts, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&(primme_svds->locking), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->initSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->numOrthoConst), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->maxBasisSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->maxBlockSize), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->maxMatvecs), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->aNorm), 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&(primme_svds->eps), 1, MPI_DOUBLE, 0, comm);
   MPI_Bcast(&(primme_svds->printLevel), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->method), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->methodStage2), 1, MPI_INT, 0, comm);
   MPI_Bcast(&(primme_svds->precondition), 1, MPI_INT, 0, comm);

   MPI_Bcast(method, 1, MPI_INT, 0, comm);
   broadCast(&primme_svds->primme, primmemethod,  NULL, master, comm);
   broadCast(&primme_svds->primmeStage2, primmemethodStage2,  NULL, master, comm);
}

#ifdef USE_PETSC
#include <petscmat.h>
#endif

/******************************************************************************
 * MPI globalSumDouble function
 *
******************************************************************************/
void par_GlobalSumDouble(void *sendBuf, void *recvBuf, int *count, 
                         primme_params *primme) {
   MPI_Comm communicator = *(MPI_Comm *) primme->commInfo;

#ifdef USE_PETSC
   extern PetscLogEvent PRIMME_GLOBAL_SUM;
   PetscLogEventBegin(PRIMME_GLOBAL_SUM,0,0,0,0);
   assert(MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator)
      == MPI_SUCCESS);
   PetscLogEventEnd(PRIMME_GLOBAL_SUM,0,0,0,0);
#else
   MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator);
#endif
}

void par_GlobalSumDoubleSvds(void *sendBuf, void *recvBuf, int *count, 
                         primme_svds_params *primme_svds) {
   MPI_Comm communicator = *(MPI_Comm *) primme_svds->commInfo;

#ifdef USE_PETSC
   extern PetscLogEvent PRIMME_GLOBAL_SUM;
   PetscLogEventBegin(PRIMME_GLOBAL_SUM,0,0,0,0);
   assert(MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator)
      == MPI_SUCCESS);
   PetscLogEventEnd(PRIMME_GLOBAL_SUM,0,0,0,0);
#else
   MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator);
#endif
}


#endif /* USE_MPI */
