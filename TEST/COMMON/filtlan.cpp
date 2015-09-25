#include <stdlib.h>  // for exit, atof, atoi, atol, srand, rand, etc.
#include <string.h>  // for memcpy, strcmp, strncmp, etc.
#include <iostream>  // for cout, cerr, endl, etc. (under namespace std)
#include <cassert>

#include "matkit.h"
#include "polyfilt.h"
#include "filtlan.h"

class SimpleSparseMatrix {
public: 
   void (*spmv)(double*, double *, void *);
   void *par;
   Real leftShift;      // makes leftShift*I - Op
   Real rightShift;     // makes Op - righShift*I

   SimpleSparseMatrix(void (*spmv_)(double*, double *, void *), void *par_, Real leftShift_=0,
                        Real rightShift_=0) :
      spmv(spmv_), par(par_), leftShift(leftShift_), rightShift(rightShift_) {};

   Vector operator*(const Vector &v) const {
      Real *ys = new Real[v.Length()];
      spmv((double*)v.Store(), ys, par);
      Vector y(v.Length(), ys);
      if (leftShift != 0) y = leftShift*v - y;
      if (rightShift != 0) y -= rightShift*v;
      return y;
   }
};

Vector FilteredConjugateResidualSimpleMatrixPolynomialVectorProduct(const SimpleSparseMatrix &A, const Vector &x0, const Vector &b,
                                                 const Matrix &baseFilter, const Vector &intv, const Vector &intervalWeights,
                                                 mkIndex niter, Real tol) {
    if (intv.Length() <= 1u) {
        // exception, no intervals, no update on x0
        return x0;
    }

    // initialize polynomial ppol to be 1 (i.e. multiplicative identity) in all intervals
    // number of intervals
    mkIndex nintv = intv.Length()-1u;
    // allocate memory
    Real *pp = new Real[2u*nintv];
    // set the polynomial be 1 for all intervals
    Real *sp = pp;
    mkIndex jj = nintv;
    while (jj--) {
        *sp++ = 1.0;
        *sp++ = 0.0;
    }
    Matrix ppol(2, nintv, pp);  // the initial p-polynomial (corresponding to the A-conjugate vector p in CG)

    // corrected CR in polynomial space
    Matrix rpol = ppol;         // rpol is the r-polynomial (corresponding to the residual vector r in CG)
    Matrix cpol = ppol;         // cpol is the "corrected" residual polynomial
    Matrix appol = PiecewisePolynomialInChebyshevBasisMultiplyX(ppol, intv);
    Matrix arpol = appol;
    Real rho00 = PiecewisePolynomialInnerProductInChebyshevBasis(rpol, arpol, intervalWeights);  // initial rho
    Real rho = rho00;

    // corrected CR in vector space
    Vector x = x0;
    #ifdef USE_CSRMV
        Vector r = b-(x*A);
    #else
        Vector r = b-(A*x);
    #endif
    Vector p = r;
    #ifdef USE_CSRMV
        Vector ap = p*A;
    #else
        Vector ap = A*p;
    #endif
    // alp0, alp, and bet will be from the polynomial space in the iteration

    // to trace the residual errors, execute the lines starting with "//*"
    //* Vector err(iter+1);
    //* err(1) = rho;
    for (mkIndex i=0; i<niter; i++) {
        // iteration in the polynomial space
        Real den = PiecewisePolynomialInnerProductInChebyshevBasis(appol, appol, intervalWeights);
        Real alp0 = rho / den;
        Real alp = alp0 - PiecewisePolynomialInnerProductInChebyshevBasis(baseFilter, appol, intervalWeights) / den;
        rpol = Matrix::xsum(rpol, (-alp0)*appol);
        cpol = Matrix::xsum(cpol, (-alp)*appol);
        arpol = PiecewisePolynomialInChebyshevBasisMultiplyX(rpol, intv);
        Real rho0 = rho;
        rho = PiecewisePolynomialInnerProductInChebyshevBasis(rpol, arpol, intervalWeights);

        // update x in the vector space
        x += alp*p;
        //* err(i+2) = (b-A*x).Norm2();
        if (rho < tol*rho00)
            break;

        // finish the iteration in the polynomial space
        Real bet = rho / rho0;
        ppol = Matrix::xsum(rpol, bet*ppol);
        appol = Matrix::xsum(arpol, bet*appol);

        // finish the iteration in the vector space
        r -= alp0*ap;
        p = r + bet*p;
        #ifdef USE_CSRMV
            ap = r*A + bet*ap;  // the only matrix-vector product in the loop
        #else
            ap = A*r + bet*ap;  // the only matrix-vector product in the loop
        #endif
    }
    return x;
}

void FilteredCRMatrixPolynomialVectorProduct_c(double frame_[4], int baseDegree, int polyDegree, double *intervalWeights_, double tol,
         void (*spmv)(double*, double *, void *), void *par,
         int n, double *xvec, double *yvec) {

   assert(sizeof(Real) == sizeof(double));

   double *xs = new double[n];
   for (int i=0; i<n; ++i) xs[i] = xvec[i];
   Vector x(n, xs);

   SimpleSparseMatrix A(spmv, par);

   Vector frame(4), intervalWeights(5);
   for (int i=0; i<4; ++i) frame(i+1) = frame_[i];
   for (int i=0; i<5; ++i) intervalWeights(i+1) = intervalWeights_[i];
   IntervalOptions opts;
   Vector intervals, intervals2;
   PolynomialFilterInfo filterInfo;
   if (frame(1) == frame(2)) {
      // low pass filter, convert it to high pass filter
      A.leftShift = frame(4);
      Vector frame2 = frame(4) - frame.reverse();
      filterInfo = GetIntervals(intervals, frame2, polyDegree, baseDegree, opts);
      // translate the interval back for return
      intervals2 = frame(4) - intervals.reverse();
   } else {
      // it can be a mid-pass filter or a high-pass filter
      if (frame(1) == 0.0) {
         filterInfo = GetIntervals(intervals, frame, polyDegree, baseDegree, opts);
         // not translation of intervals
         intervals2 = intervals;
      } else {
         A.rightShift = frame(1);
         Vector frame2 = frame - frame(1);
         filterInfo = GetIntervals(intervals, frame2, polyDegree, baseDegree, opts);
         // shift the intervals back for return
         intervals2 = intervals + frame(1);
      }
   }
   static const int HighLowFlags[] = { 1, -1, 0, -1, 1 };
   Matrix baseFilter = HermiteBaseFilterInChebyshevBasis(intervals, HighLowFlags, baseDegree);

   Vector z = Vector::zeros(n);
   Vector y = A*FilteredConjugateResidualSimpleMatrixPolynomialVectorProduct(A, z, x, baseFilter, intervals, intervalWeights, polyDegree, tol);
   for (int i=0; i<n; ++i) yvec[i] = y.Store()[i];
}

