#ifndef C_KCIPT_SDCIT2_H
#define C_KCIPT_SDCIT2_H

void c_sdcit2(const double * const K_XZ, const double * const K_Y, const double * const K_Z, const double * const D_Z_, const int n,
              const int b, const int seed, const int n_threads,
              double *const mmsd, double *const error_mmsd, double *const null, double *const error_null);


#endif //C_KCIPT_SDCIT2_H
