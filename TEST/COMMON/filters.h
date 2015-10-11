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

#ifndef FILTERS_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "primme.h"

typedef struct filter_params {
   int filter;
   int degrees;
   int lowerBound;
   int upperBound;
   double lowerBoundFix, upperBoundFix, lowerBoundTuned, upperBoundTuned;
   int prodIfFullRange;
   double minEig, maxEig;
   void (*matvec)
      ( void *x,  void *y, int *blockSize, struct primme_params *primme);
   void (*precond)
      ( void *x,  void *y, int *blockSize, struct primme_params *primme);
   double checkEps, lastCheckCS;
} filter_params;


void Apply_filter(void *x, void *y, int *blockSize, filter_params *filter,
                  primme_params *primme, int stats);
void plot_filter(int n, filter_params *filter, primme_params *primme, FILE *out);
void Setup_filter_augmented(filter_params *filter, primme_params *primme);
int tune_filter(filter_params *filter, primme_params *primme, int onlyIfStatic);
void getBounds(filter_params *filter, primme_params *primme, double *lb, double *ub);

// Please, don't use global variables!
extern double elapsedTimeAMV, elapsedTimeFilterMV;
extern int numFilterApplies;

#define FILTERS_H
#endif
