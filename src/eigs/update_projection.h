/*******************************************************************************
 * Copyright (c) 2018, College of William & Mary
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
 *   NOTE: THIS FILE IS AUTOMATICALLY GENERATED. PLEASE DON'T MODIFY
 ******************************************************************************/


#ifndef update_projection_H
#define update_projection_H
int update_projection_hprimme(dummy_type_hprimme *X, PRIMME_INT ldX, dummy_type_hprimme *Y,
      PRIMME_INT ldY, dummy_type_sprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_kprimme(dummy_type_kprimme *X, PRIMME_INT ldX, dummy_type_kprimme *Y,
      PRIMME_INT ldY, dummy_type_cprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_sprimme(dummy_type_sprimme *X, PRIMME_INT ldX, dummy_type_sprimme *Y,
      PRIMME_INT ldY, dummy_type_sprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_cprimme(dummy_type_cprimme *X, PRIMME_INT ldX, dummy_type_cprimme *Y,
      PRIMME_INT ldY, dummy_type_cprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
#if !defined(CHECK_TEMPLATE) && !defined(update_projection_Sprimme)
#  define update_projection_Sprimme CONCAT(update_projection_,SCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(update_projection_Rprimme)
#  define update_projection_Rprimme CONCAT(update_projection_,REAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(update_projection_SHprimme)
#  define update_projection_SHprimme CONCAT(update_projection_,HOST_SCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(update_projection_RHprimme)
#  define update_projection_RHprimme CONCAT(update_projection_,HOST_REAL_SUF)
#endif
int update_projection_dprimme(dummy_type_dprimme *X, PRIMME_INT ldX, dummy_type_dprimme *Y,
      PRIMME_INT ldY, dummy_type_dprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_zprimme(dummy_type_zprimme *X, PRIMME_INT ldX, dummy_type_zprimme *Y,
      PRIMME_INT ldY, dummy_type_zprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_magma_sprimme(dummy_type_magma_sprimme *X, PRIMME_INT ldX, dummy_type_magma_sprimme *Y,
      PRIMME_INT ldY, dummy_type_sprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_magma_cprimme(dummy_type_magma_cprimme *X, PRIMME_INT ldX, dummy_type_magma_cprimme *Y,
      PRIMME_INT ldY, dummy_type_cprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_magma_dprimme(dummy_type_magma_dprimme *X, PRIMME_INT ldX, dummy_type_magma_dprimme *Y,
      PRIMME_INT ldY, dummy_type_dprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
int update_projection_magma_zprimme(dummy_type_magma_zprimme *X, PRIMME_INT ldX, dummy_type_magma_zprimme *Y,
      PRIMME_INT ldY, dummy_type_zprimme *Z, PRIMME_INT ldZ, PRIMME_INT nLocal,
      int numCols, int blockSize, int isSymmetric, primme_context ctx);
#endif
