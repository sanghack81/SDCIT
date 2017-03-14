//
// Created by Sanghack Lee on 3/14/17.
//

#ifndef C_KCIPT_SDCIT_H
#define C_KCIPT_SDCIT_H


void c_sdcit(const double *K_XZ, const double *K_Y, const double *D_Z_, const int n,
             const int b, const int seed, const int n_threads,
             double *const mmsd, double *const null);


#endif //C_KCIPT_SDCIT_H
