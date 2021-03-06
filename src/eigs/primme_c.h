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


#ifndef primme_c_H
#define primme_c_H
int wrapper_hprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_kprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_sprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_cprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Sprimme)
#  define wrapper_Sprimme CONCAT(wrapper_,SCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Rprimme)
#  define wrapper_Rprimme CONCAT(wrapper_,REAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SHprimme)
#  define wrapper_SHprimme CONCAT(wrapper_,HOST_SCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RHprimme)
#  define wrapper_RHprimme CONCAT(wrapper_,HOST_REAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SXprimme)
#  define wrapper_SXprimme CONCAT(wrapper_,XSCALAR_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RXprimme)
#  define wrapper_RXprimme CONCAT(wrapper_,XREAL_SUF)
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Shprimme)
#  define wrapper_Shprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,USE_ARITH(h,k)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Rhprimme)
#  define wrapper_Rhprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,h),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Ssprimme)
#  define wrapper_Ssprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Rsprimme)
#  define wrapper_Rsprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Sdprimme)
#  define wrapper_Sdprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,USE_ARITH(d,z)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Rdprimme)
#  define wrapper_Rdprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,d),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Sqprimme)
#  define wrapper_Sqprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,USE_ARITH(q,w)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_Rqprimme)
#  define wrapper_Rqprimme CONCAT(wrapper_,CONCAT(CONCAT(STEM_C,q),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SXhprimme)
#  define wrapper_SXhprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(h,k)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RXhprimme)
#  define wrapper_RXhprimme CONCAT(wrapper_,CONCAT(CONCAT(,h),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SXsprimme)
#  define wrapper_SXsprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RXsprimme)
#  define wrapper_RXsprimme CONCAT(wrapper_,CONCAT(CONCAT(,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SXdprimme)
#  define wrapper_SXdprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(d,z)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RXdprimme)
#  define wrapper_RXdprimme CONCAT(wrapper_,CONCAT(CONCAT(,d),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SXqprimme)
#  define wrapper_SXqprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(q,w)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RXqprimme)
#  define wrapper_RXqprimme CONCAT(wrapper_,CONCAT(CONCAT(,q),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SHhprimme)
#  define wrapper_SHhprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RHhprimme)
#  define wrapper_RHhprimme CONCAT(wrapper_,CONCAT(CONCAT(,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SHsprimme)
#  define wrapper_SHsprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(s,c)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RHsprimme)
#  define wrapper_RHsprimme CONCAT(wrapper_,CONCAT(CONCAT(,s),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SHdprimme)
#  define wrapper_SHdprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(d,z)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RHdprimme)
#  define wrapper_RHdprimme CONCAT(wrapper_,CONCAT(CONCAT(,d),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_SHqprimme)
#  define wrapper_SHqprimme CONCAT(wrapper_,CONCAT(CONCAT(,USE_ARITH(q,w)),primme))
#endif
#if !defined(CHECK_TEMPLATE) && !defined(wrapper_RHqprimme)
#  define wrapper_RHqprimme CONCAT(wrapper_,CONCAT(CONCAT(,q),primme))
#endif
int wrapper_dprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_zprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_magma_sprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_magma_cprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_magma_dprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_magma_zprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_magma_hprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
int wrapper_magma_kprimme(primme_op_datatype input_type, void *evals, void *evecs,
      void *resNorms, primme_context ctx);
#endif
