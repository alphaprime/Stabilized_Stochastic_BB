/*
 * ste_sgd_mex_ifo.c - STE for ordinal embedding with SGD in MATLAB External Interfaces
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

double *base_arr;
static int compar (const void *a, const void *b);
void object_gradient(double *X, int *index, double *grad, int N, int dim);
//void evaluat_error(double *X, double *test, double *label, double *error, int error_type, int num_test, int N, int dim, int s);
void evaluat_error(double *X, double *train, double *test, double *label, double *train_error, double *test_error, int error_type, int num_train, int num_test, int N, int dim, int s);

static int compar (const void *a, const void *b)
{
	int aa = *((int *) a), bb = *((int *) b);
	if (base_arr[aa] < base_arr[bb])
		return -1;
	if (base_arr[aa] == base_arr[bb])
		return 0;
	if (base_arr[aa] > base_arr[bb])
		return 1;
}

void object_gradient(double *X, int *index, double *grad, int N, int dim)
{
	int d;
	double dij, dik, tmp;
	dij = dik = tmp = 0;
	for (d = 0; d < dim; ++d)
	{
		dij += pow(((*(X+(*(index+0))+d*N))-(*(X+(*(index+1))+d*N))), 2);
		dik += pow(((*(X+(*(index+0))+d*N))-(*(X+(*(index+2))+d*N))), 2);
		*(grad+0+d*3) = 0;
		*(grad+1+d*3) = 0;
		*(grad+2+d*3) = 0;	
	}
	tmp = 2*exp(dij-dik)/(1+exp(dij-dik));
	for (d = 0; d < dim; ++d)
	{
		*(grad+0+d*3) = tmp*(*(X+(*(index+2))+d*N)-*(X+(*(index+1))+d*N));
		*(grad+1+d*3) = tmp*(*(X+(*(index+1))+d*N)-*(X+(*(index+0))+d*N));
		*(grad+2+d*3) = tmp*(*(X+(*(index+0))+d*N)-*(X+(*(index+2))+d*N));
	}
}

void evaluat_error(double *X, double *train, double *test, double *label, double *train_error, double *test_error, int error_type, int num_train, int num_test, int N, int dim, int s)
{
	int i, j, n, d, no_viol, *index;
	double *D, *tmp;
	no_viol = 0;
	D = malloc(N*N*sizeof(*D));
	tmp = malloc(N*sizeof(*tmp));
	index = malloc(N*sizeof(*index));
	*(train_error+s) = 0;
	*(test_error+s) = 0;
	for (i = 0; i < N-1; ++i)
	{
		*(D+i+i*N) = 0;
		for (j = i+1; j < N; ++j)
		{
			*(D+i+j*N) = 0;
			*(D+j+i*N) = 0;
			for (d = 0; d < dim; ++d)
			{
				*(D+i+j*N) += pow(((*(X+i+d*N))-(*(X+j+d*N))), 2);
			}
			*(D+i+j*N) = sqrt(*(D+i+j*N));
			*(D+j+i*N) = *(D+i+j*N);
		}
	}
	if (error_type == 1)
	{
		for (n = 0; n < num_test; ++n)
		{
			if ((*(D+(int)(*(test+n+0*num_test))+((int)(*(test+n+1*num_test)))*N))>(*(D+(int)(*(test+n+0*num_test))+((int)(*(test+n+2*num_test)))*N)))
			{
				no_viol += 1;
			}
		}
		*(test_error+s) = (double) no_viol/num_test;
		no_viol = 0;
		for (n = 0; n < num_train; ++n)
		{
			if ((*(D+(int)(*(train+n+0*num_train))+((int)(*(train+n+1*num_train)))*N))>(*(D+(int)(*(train+n+0*num_train))+((int)(*(train+n+2*num_train)))*N)))
			{
				no_viol += 1;
			}
		}
		*(train_error+s) = (double) no_viol/num_train;
	}
	else
	{
		for (i = 0; i < N; ++i)
		{
			for (j = 0; j < N; ++j)
			{
				*(tmp+j) = *(D+i+j*N);
				*(index+j) = j;
			}
			base_arr = tmp;
			qsort(index, N, sizeof(int), compar);
			if (*(label+*(index+1)) != *(label+i))
			{
				no_viol += 1;
			}	
		}
		*(test_error+s) = (double) no_viol/N;
	}
	free(D);
	free(tmp);
	free(index);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int i, n, d, s, N, t, num_train, num_test, dim, p, count, sgd_iter, svrg_iter, error_type, scheduling_type, *index, *label;
	double relative_residual, eta, eta_int, eta_par, sum_residel, sum, *duration, *X_int, *X_new, *X_old, *train, *test, *stoch_grd, *train_error, *test_error;
	clock_t start, finish;

	/*Input*/
	X_int = mxGetPr(prhs[0]);
	train = mxGetPr(prhs[1]);
	test = mxGetPr(prhs[2]);
	label = mxGetPr(prhs[3]);
	N = (int) mxGetScalar(prhs[4]);
	dim = (int) mxGetScalar(prhs[5]);
	num_train = (int) mxGetScalar(prhs[6]);
	num_test = (int) mxGetScalar(prhs[7]);
	eta_int= mxGetScalar(prhs[8]);
	eta_par = mxGetScalar(prhs[9]);
	scheduling_type = (int) mxGetScalar(prhs[10]);
	sgd_iter = (int) mxGetScalar(prhs[11]);
	svrg_iter = (int) mxGetScalar(prhs[12]);
	error_type = (int) mxGetScalar(prhs[13]);

	/*Output*/
	plhs[0] = mxCreateDoubleMatrix(N, dim, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);

	X_new = mxGetPr(plhs[0]);
	train_error = mxGetPr(plhs[1]);
	test_error = mxGetPr(plhs[2]);
	duration = mxGetPr(plhs[3]);
	
	X_old = malloc(N*dim*sizeof(*X_old));
	stoch_grd = malloc(3*dim*sizeof(*stoch_grd));
	index = malloc(3*sizeof(*index));

	i = n = d = s = p = t = 0;
	sum_residel = sum = eta = 0;
	relative_residual = 1;
	count = sgd_iter/svrg_iter;

	for (n = 0; n < N; ++n)
	{
		for (d = 0; d < dim; ++d)
		{
			*(X_old+n+d*N) = *(X_int+n+d*N);
			*(X_new+n+d*N) = *(X_old+n+d*N);
		}
	}

	while (s < sgd_iter && relative_residual > 1e-5)
	{
		start = clock();
		*(index+0) = 0;
		*(index+1) = 0;
		*(index+2) = 0;
		for (i = 0; i < num_train; ++i)
		{
			p = 0 + (int) (rand() / (double) (RAND_MAX + 1)*(num_train-1 - 0 + 1));
			*(index+0) = (int) (*(train+p+0*num_train));
			*(index+1) = (int) (*(train+p+1*num_train));
			*(index+2) = (int) (*(train+p+2*num_train));
			object_gradient(X_new, index, stoch_grd, N, dim);
			if (scheduling_type == 1)
			{
				eta = eta_int/(1+eta_par*floor((double)((s*num_train+i+1)/num_train)));
			}
			else
			{
				eta = eta_int*pow(eta_par, floor((double)((s*num_train+i+1)/num_train)));
			}
			for (d = 0; d < dim; ++d)
			{
				*(X_new+(*(index+0))+d*N) -= eta * (*(stoch_grd+0+d*3));
				*(X_new+(*(index+1))+d*N) -= eta * (*(stoch_grd+1+d*3));
				*(X_new+(*(index+2))+d*N) -= eta * (*(stoch_grd+2+d*3));
			}
		}
		finish = clock();
		*(duration+t) += (double)(finish - start) / CLOCKS_PER_SEC;
		for (n = 0; n < N; ++n)
		{
			for (d = 0; d < dim; ++d)
			{
				sum += pow((*(X_old+n+d*N)), 2);
				sum_residel += pow(((*(X_new+n+d*N))-(*(X_old+n+d*N))), 2);
				*(X_old+n+d*N) = *(X_new+n+d*N);
			}
		}
		relative_residual = sqrt(sum_residel)/sqrt(sum);
		sum_residel = 0;
		sum = 0;
		if (s%count == 0)
		{
			evaluat_error(X_new, train, test, label, train_error, test_error, error_type, num_train, num_test, N, dim, t);
			//if (*(test_error+t) <= 0.2)
			//{
			//	break;
			//}
			t += 1;
		}
		s += 1;
	}
	free(X_old);
	free(stoch_grd);
	free(index);
}