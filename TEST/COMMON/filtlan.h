
#ifdef Cplusplus
extern "C" {
#endif /* Cplusplus */
   void FilteredCRMatrixPolynomialVectorProduct_c(double frame_[4], int baseDegrees, int polyDegrees, double *intervalWeight_, double tol,
         void (*spmv)(double*, double *, void *), void *par,
         int n, double *xvec, double *yvec);
#ifdef Cplusplus
}
#endif /* Cplusplus */
