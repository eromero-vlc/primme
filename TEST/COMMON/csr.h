/*  PRIMME PReconditioned Iterative MultiMethod Eigensolver
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
 */

#ifndef CSR_H

typedef struct {
   int *JA;
   int *IA;
   double *AElts;
   int n;
   int nnz;
} CSRMatrix;

int readMatrixNative(const char* matrixFileName, CSRMatrix **matrix_, double *fnorm);
double frobeniusNorm(int n, int *IA, double *AElts);
void shiftCSRMatrix(double shift, int n, int *IA, int *JA, double *AElts);

#define CSR_H
#endif
