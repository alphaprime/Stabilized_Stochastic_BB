/*
 * gnmds_svrg_mex_ifo.c - GNMDS for ordinal embedding with SVRG in MATLAB External Interfaces
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

double *base_arr;
static int compar (const void *a, const void *b);
void average_gradient(double *X, double *train, double *avg_grad, int num_triple, int N, int dim);
void object_gradient(double *X, int *index, double *grad, int N, int dim);
//void evaluat_error(double *X, double *test, double *label, double *error, int error_type, int num_test, int N, int dim, int s);
void evaluat_error(double *X, double *train, double *test, double *label, double *train_error, double *test_error, double *X_norm, int error_type, int num_train, int num_test, int N, int dim, int s);

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

void average_gradient(double *X, double *train, double *avg_grad, int num_triple, int N, int dim)
{	
	int i, d, *index;
	double *grad;
	grad = malloc(3*dim*sizeof(*grad));
	index = malloc(3*sizeof(*index));
	*(index+0) = 0;
	*(index+1) = 0;
	*(index+2) = 0;
	for (i = 0; i < num_triple; ++i)
	{
		*(index+0) = (int)*(train+i+0*num_triple);
		*(index+1) = (int)*(train+i+1*num_triple);
		*(index+2) = (int)*(train+i+2*num_triple);
		object_gradient(X, index, grad, N, dim);
		for (d = 0; d < dim; ++d)
		{
			*(avg_grad+(*(index+0))+d*N) += (*(grad+0+d*3));
			*(avg_grad+(*(index+1))+d*N) += (*(grad+1+d*3));
			*(avg_grad+(*(index+2))+d*N) += (*(grad+2+d*3));
		}
	}
	for (i = 0; i < N; ++i)
	{
		for (d = 0; d < dim; ++d)
		{
			*(avg_grad+i+d*N) /= (double) num_triple;
		}
	}
	free(grad);
	free(index);
}

void object_gradient(double *X, int *index, double *grad, int N, int dim)
{
	int d;
	double dij, dik;
	dij = dik = 0;
	for (d = 0; d < dim; ++d)
	{
		dij += pow(((*(X+(*(index+0))+d*N))-(*(X+(*(index+1))+d*N))), 2);
		dik += pow(((*(X+(*(index+0))+d*N))-(*(X+(*(index+2))+d*N))), 2);
		*(grad+0+d*3) = 0;
		*(grad+1+d*3) = 0;
		*(grad+2+d*3) = 0;	
	}
	if (dij+1-dik > 0)
	{
		for (d = 0; d < dim; ++d)
		{
			*(grad+0+d*3) = 2*(*(X+(*(index+2))+d*N)-*(X+(*(index+1))+d*N));
			*(grad+1+d*3) = 2*(*(X+(*(index+1))+d*N)-*(X+(*(index+0))+d*N));
			*(grad+2+d*3) = 2*(*(X+(*(index+0))+d*N)-*(X+(*(index+2))+d*N));
		}
	}
}

void evaluat_error(double *X, double *train, double *test, double *label, double *train_error, double *test_error, double *X_norm, int error_type, int num_train, int num_test, int N, int dim, int s)
{
	int i, j, n, d, no_viol, *index;
	double *D, *tmp;
	no_viol = 0;
	D = malloc(N*N*sizeof(*D));
	tmp = malloc(N*sizeof(*tmp));
	index = malloc(N*sizeof(*index));
	*(train_error+s) = 0;
	*(test_error+s) = 0;
	*(X_norm+s) = 0;
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
		for (d = 0; d < dim; ++d)
		{
			*(X_norm+s) += pow((*(X+i+d*N)), 2);
		}
	}
	*(X_norm+s) = sqrt(*(X_norm+s));
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
	int i, n, d, s, N, num_train, num_test, dim, p, svrg_iter, frq_iter, error_type, *index, *label;
	double relative_residual, eta, sum_residel, sum, *duration, *X_int, *X_new, *X_snapshot, *train, *test, *avg_grad, *stoch_grd, *snapshot_grd, *train_error, *test_error, *X_norm;
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
	eta = mxGetScalar(prhs[8]);
	frq_iter = (int) mxGetScalar(prhs[9]);
	svrg_iter = (int) mxGetScalar(prhs[10]);
	error_type = (int) mxGetScalar(prhs[11]);

	/*Output*/
	plhs[0] = mxCreateDoubleMatrix(N, dim, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);
	plhs[4] = mxCreateDoubleMatrix(1, svrg_iter, mxREAL);

	X_new = mxGetPr(plhs[0]);
	train_error = mxGetPr(plhs[1]);
	test_error = mxGetPr(plhs[2]);
	duration = mxGetPr(plhs[3]);
	X_norm = mxGetPr(plhs[4]);

	X_snapshot = malloc(N*dim*sizeof(*X_snapshot));
	avg_grad = malloc(N*dim*sizeof(*avg_grad));
	stoch_grd = malloc(3*dim*sizeof(*stoch_grd));
	snapshot_grd = malloc(3*dim*sizeof(*snapshot_grd));
	index = malloc(3*sizeof(*index));

	i = n = d = s = p = 0;
	sum_residel = sum = 0;
	relative_residual = 1;
	
	for (n = 0; n < N; ++n)
	{
		for (d = 0; d < dim; ++d)
		{
			*(X_snapshot+n+d*N) = *(X_int+n+d*N);
		}
	}
	

	while (s < svrg_iter)
	{
		start = clock();
		for (n = 0; n < N; ++n)
		{
			for (d = 0; d < dim; ++d)
			{
				*(X_new+n+d*N) = *(X_snapshot+n+d*N);
				*(avg_grad+n+d*N) = 0;
			}
		}
		*(index+0) = 0;
		*(index+1) = 0;
		*(index+2) = 0;
		average_gradient(X_snapshot, train, avg_grad, num_train, N, dim);
		for (i = 0; i < frq_iter; ++i)
		{
			//p = rand() % (num_train + 1 - 1) + 1;
			p = 0 + (int) (rand() / (double) (RAND_MAX + 1)*(num_train-1 - 0 + 1));
			*(index+0) = (int) (*(train+p+0*num_train));
			*(index+1) = (int) (*(train+p+1*num_train));
			*(index+2) = (int) (*(train+p+2*num_train));
			object_gradient(X_new, index, stoch_grd, N, dim);
			object_gradient(X_snapshot, index, snapshot_grd, N, dim);
			for (d = 0; d < dim; ++d)
			{
				*(X_new+(*(index+0))+d*N) -= eta * ((*(stoch_grd+0+d*3))-(*(snapshot_grd+0+d*3)));
				*(X_new+(*(index+1))+d*N) -= eta * ((*(stoch_grd+1+d*3))-(*(snapshot_grd+1+d*3)));
				*(X_new+(*(index+2))+d*N) -= eta * ((*(stoch_grd+2+d*3))-(*(snapshot_grd+2+d*3)));
			}
			for (n = 0; n < N; ++n)
			{
				for (d = 0; d < dim; ++d)
				{
					*(X_new+n+d*N) -= eta * (*(avg_grad+n+d*N));
				}
			}
		}
		finish = clock();
		*(duration+s) = (double)(finish - start) / CLOCKS_PER_SEC;
		for (n = 0; n < N; ++n)
		{
			for (d = 0; d < dim; ++d)
			{
				sum += pow((*(X_snapshot+n+d*N)), 2);
				sum_residel += pow(((*(X_new+n+d*N))-(*(X_snapshot+n+d*N))), 2);
				*(X_snapshot+n+d*N) = *(X_new+n+d*N);
			}
		}
		relative_residual = sqrt(sum_residel)/sqrt(sum);
		sum_residel = 0;
		sum = 0;
		evaluat_error(X_new, train, test, label, train_error, test_error, X_norm, error_type, num_train, num_test, N, dim, s);
		//if (*(test_error+s) <= 0.2)
		//{
		//	break;
		//}
		s += 1;
	}
	free(X_snapshot);
	free(avg_grad);
	free(stoch_grd);
	free(snapshot_grd);
	free(index);
}