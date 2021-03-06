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


#ifndef correction_H
#define correction_H
int solve_correction_hprimme(dummy_type_hprimme *V, PRIMME_INT ldV, dummy_type_hprimme *W,
      PRIMME_INT ldW, dummy_type_hprimme *BV, PRIMME_INT ldBV, dummy_type_hprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_hprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_hprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_sprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_kprimme(dummy_type_kprimme *V, PRIMME_INT ldV, dummy_type_kprimme *W,
      PRIMME_INT ldW, dummy_type_kprimme *BV, PRIMME_INT ldBV, dummy_type_kprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_kprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_kprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_cprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_sprimme(dummy_type_sprimme *V, PRIMME_INT ldV, dummy_type_sprimme *W,
      PRIMME_INT ldW, dummy_type_sprimme *BV, PRIMME_INT ldBV, dummy_type_sprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_sprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_sprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_sprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_cprimme(dummy_type_cprimme *V, PRIMME_INT ldV, dummy_type_cprimme *W,
      PRIMME_INT ldW, dummy_type_cprimme *BV, PRIMME_INT ldBV, dummy_type_cprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_cprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_cprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_cprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Sprimme)
#  define solve_correction_Sprimme CONCAT(solve_correction_,SCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Rprimme)
#  define solve_correction_Rprimme CONCAT(solve_correction_,REAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SHprimme)
#  define solve_correction_SHprimme CONCAT(solve_correction_,HOST_SCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RHprimme)
#  define solve_correction_RHprimme CONCAT(solve_correction_,HOST_REAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SXprimme)
#  define solve_correction_SXprimme CONCAT(solve_correction_,XSCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RXprimme)
#  define solve_correction_RXprimme CONCAT(solve_correction_,XREAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Shprimme)
#  define solve_correction_Shprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,USE_ARITH(h,k)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Rhprimme)
#  define solve_correction_Rhprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,h),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Ssprimme)
#  define solve_correction_Ssprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Rsprimme)
#  define solve_correction_Rsprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Sdprimme)
#  define solve_correction_Sdprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,USE_ARITH(d,z)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Rdprimme)
#  define solve_correction_Rdprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,d),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Sqprimme)
#  define solve_correction_Sqprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,USE_ARITH(q,w)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_Rqprimme)
#  define solve_correction_Rqprimme CONCAT(solve_correction_,CONCAT(CONCAT(STEM_C,q),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SXhprimme)
#  define solve_correction_SXhprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(h,k)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RXhprimme)
#  define solve_correction_RXhprimme CONCAT(solve_correction_,CONCAT(CONCAT(,h),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SXsprimme)
#  define solve_correction_SXsprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RXsprimme)
#  define solve_correction_RXsprimme CONCAT(solve_correction_,CONCAT(CONCAT(,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SXdprimme)
#  define solve_correction_SXdprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(d,z)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RXdprimme)
#  define solve_correction_RXdprimme CONCAT(solve_correction_,CONCAT(CONCAT(,d),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SXqprimme)
#  define solve_correction_SXqprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(q,w)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RXqprimme)
#  define solve_correction_RXqprimme CONCAT(solve_correction_,CONCAT(CONCAT(,q),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SHhprimme)
#  define solve_correction_SHhprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RHhprimme)
#  define solve_correction_RHhprimme CONCAT(solve_correction_,CONCAT(CONCAT(,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SHsprimme)
#  define solve_correction_SHsprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RHsprimme)
#  define solve_correction_RHsprimme CONCAT(solve_correction_,CONCAT(CONCAT(,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SHdprimme)
#  define solve_correction_SHdprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(d,z)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RHdprimme)
#  define solve_correction_RHdprimme CONCAT(solve_correction_,CONCAT(CONCAT(,d),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_SHqprimme)
#  define solve_correction_SHqprimme CONCAT(solve_correction_,CONCAT(CONCAT(,USE_ARITH(q,w)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(solve_correction_RHqprimme)
#  define solve_correction_RHqprimme CONCAT(solve_correction_,CONCAT(CONCAT(,q),primme))
#endif
int solve_correction_dprimme(dummy_type_dprimme *V, PRIMME_INT ldV, dummy_type_dprimme *W,
      PRIMME_INT ldW, dummy_type_dprimme *BV, PRIMME_INT ldBV, dummy_type_dprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_dprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_dprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_dprimme *Mfact, int *ipivot, dummy_type_dprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_dprimme *ritzVals,
      dummy_type_dprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_dprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_zprimme(dummy_type_zprimme *V, PRIMME_INT ldV, dummy_type_zprimme *W,
      PRIMME_INT ldW, dummy_type_zprimme *BV, PRIMME_INT ldBV, dummy_type_zprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_zprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_zprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_zprimme *Mfact, int *ipivot, dummy_type_dprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_dprimme *ritzVals,
      dummy_type_dprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_dprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_magma_sprimme(dummy_type_magma_sprimme *V, PRIMME_INT ldV, dummy_type_magma_sprimme *W,
      PRIMME_INT ldW, dummy_type_magma_sprimme *BV, PRIMME_INT ldBV, dummy_type_magma_sprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_magma_sprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_magma_sprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_sprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_magma_cprimme(dummy_type_magma_cprimme *V, PRIMME_INT ldV, dummy_type_magma_cprimme *W,
      PRIMME_INT ldW, dummy_type_magma_cprimme *BV, PRIMME_INT ldBV, dummy_type_magma_cprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_magma_cprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_magma_cprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_cprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_magma_dprimme(dummy_type_magma_dprimme *V, PRIMME_INT ldV, dummy_type_magma_dprimme *W,
      PRIMME_INT ldW, dummy_type_magma_dprimme *BV, PRIMME_INT ldBV, dummy_type_magma_dprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_magma_dprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_magma_dprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_dprimme *Mfact, int *ipivot, dummy_type_dprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_dprimme *ritzVals,
      dummy_type_dprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_dprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_magma_zprimme(dummy_type_magma_zprimme *V, PRIMME_INT ldV, dummy_type_magma_zprimme *W,
      PRIMME_INT ldW, dummy_type_magma_zprimme *BV, PRIMME_INT ldBV, dummy_type_magma_zprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_magma_zprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_magma_zprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_zprimme *Mfact, int *ipivot, dummy_type_dprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_dprimme *ritzVals,
      dummy_type_dprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_dprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_magma_hprimme(dummy_type_magma_hprimme *V, PRIMME_INT ldV, dummy_type_magma_hprimme *W,
      PRIMME_INT ldW, dummy_type_magma_hprimme *BV, PRIMME_INT ldBV, dummy_type_magma_hprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_magma_hprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_magma_hprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_sprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
int solve_correction_magma_kprimme(dummy_type_magma_kprimme *V, PRIMME_INT ldV, dummy_type_magma_kprimme *W,
      PRIMME_INT ldW, dummy_type_magma_kprimme *BV, PRIMME_INT ldBV, dummy_type_magma_kprimme *evecs,
      PRIMME_INT ldevecs, dummy_type_magma_kprimme *Bevecs, PRIMME_INT ldBevecs, dummy_type_magma_kprimme *evecsHat,
      PRIMME_INT ldevecsHat, dummy_type_cprimme *Mfact, int *ipivot, dummy_type_sprimme *lockedEvals,
      int numLocked, int numConvergedStored, dummy_type_sprimme *ritzVals,
      dummy_type_sprimme *prevRitzVals, int *numPrevRitzVals, int *flags, int basisSize,
      dummy_type_sprimme *blockNorms, int *iev, int blockSize, int *touch, double startTime,
      primme_context ctx);
#endif
