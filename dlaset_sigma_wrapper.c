/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @generated d Sat Aug 13 15:02:01 2016
 *
 */

#include <lapacke.h>
#include "dplasma.h"

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

static int
dplasma_dlaset_sigma_operator( dague_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    int tempmm, tempnn, ldam, i;
    double *alpha = (double*)args;
    double *A = (double*)_A;
    (void)eu;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );

    if (m == n) {
        LAPACKE_dlaset_work(
            LAPACK_COL_MAJOR, lapack_const( uplo ), tempmm, tempnn,
            alpha[0], alpha[0], A, ldam);

        /* D(i)=1 - (i-1)/(N-1)*(1 - 1/COND) */
        if ( m == 0 ) {
            A[0] = 1.;
            i = 1;
        }
        else {
            i = 0;
        }
        if ( descA->n > 1 ) {
            double tmp = 1. / (alpha[1]);
            double alp = ( 1. - tmp ) / ((double)( descA->n - 1 ));
            for(; i < min(tempmm, tempnn); i++){
                A[i+i*ldam] = (double)( (double)(descA->n-(descA->nb*n+i+1)) * alp + tmp );
            }
        }
    } else {
        LAPACKE_dlaset_work(
            LAPACK_COL_MAJOR, 'A', tempmm, tempnn,
            alpha[0], alpha[0], A, ldam);
    }
    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_dlaset_New - Generates the handle that set the elements of the matrix
 * A on the diagonal to beta and the off-diagonals eklements to alpha.
 *
 * See dplasma_map_New() for further information.
 *
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, A has been set accordingly.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_dlaset_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_dlaset
 * @sa dplasma_dlaset_Destruct
 * @sa dplasma_claset_New
 * @sa dplasma_dlaset_New
 * @sa dplasma_slaset_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_dlaset_sigma_New( PLASMA_enum uplo,
                          double alpha,
                          double beta,
                          tiled_matrix_desc_t *A )
{
    double *params = (double*)malloc(2 * sizeof(double));

    params[0] = alpha;
    params[1] = beta;

    return dplasma_map_New( uplo, A, dplasma_dlaset_sigma_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_dlaset_Destruct - Free the data structure associated to an handle
 *  created with dplasma_dlaset_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_dlaset_New
 * @sa dplasma_dlaset
 *
 ******************************************************************************/
void
dplasma_dlaset_sigma_Destruct( dague_handle_t *handle )
{
    dplasma_map_Destruct(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_dlaset - Set the elements of the matrix
 * A on the diagonal to beta and the off-diagonals eklements to alpha.
 *
 * See dplasma_map() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is refrenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, A has been set accordingly.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_dlaset_New
 * @sa dplasma_dlaset_Destruct
 * @sa dplasma_claset
 * @sa dplasma_dlaset
 * @sa dplasma_slaset
 *
 ******************************************************************************/
int
dplasma_dlaset_sigma( dague_context_t *dague,
                      PLASMA_enum uplo,
                      double alpha,
                      double beta,
                      tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_dlaset = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_dlaset", "illegal value of type");
        return -1;
    }

    dague_dlaset = dplasma_dlaset_sigma_New(uplo, alpha, beta, A);

    if ( dague_dlaset != NULL ) {
        dague_enqueue(dague, (dague_handle_t*)dague_dlaset);
        dague_context_wait( dague );
        dplasma_dlaset_sigma_Destruct( dague_dlaset );
    }
    return 0;
}
