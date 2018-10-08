/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include <dplasma.h>
#include <dague/private_mempool.h>
#include <dague/arena.h>
#include "zgeqrf.h"
#include "zgeqrf_full.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_full_New - Generates the dague handle that computes the generation
 *  of an M-by-N matrix Q with orthonormal columns, which is defined as the
 *  first N columns of a product of K elementary reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_New().
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague handle which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_full_Destruct
 * @sa dplasma_zgeqrf_full
 * @sa dplasma_cgeqrf_full_New
 * @sa dplasma_dgeqrf_full_New
 * @sa dplasma_sgeqrf_full_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgeqrf_full_New( int optqr, int optid,
                         tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                         tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                         tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 )
{
    dague_zgeqrf_full_handle_t* handle;
    int ib = T1->mb;

    if ( Q1->n > Q1->m ) {
        dplasma_error("dplasma_zgeqrf_full_New", "illegal size of Q (N should be smaller or equal to M)");
        return NULL;
    }
    if ( A1->n > Q1->n ) {
        dplasma_error("dplasma_zgeqrf_full_New", "illegal size of A (K should be smaller or equal to N)");
        return NULL;
    }
    if ( (T1->nt < A1->nt) || (T1->mt < A1->mt) ) {
        dplasma_error("dplasma_zgeqrf_full_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    handle = dague_zgeqrf_full_new( (dague_ddesc_t*)A1, (dague_ddesc_t*)A2,
                                    (dague_ddesc_t*)T1, (dague_ddesc_t*)T2,
                                    (dague_ddesc_t*)Q1, (dague_ddesc_t*)Q2,
                                    optqr, optid, NULL, NULL );

    handle->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_tau, T1->nb * sizeof(dague_complex64_t) );

    handle->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_work, ib * T1->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_full_DEFAULT_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_UpperLower, 1, A1->mb, A1->nb, A1->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    /* Lower triangular part of tile without diagonal */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_full_LOWER_TILE_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_Lower, 0, A1->mb, A1->nb, A1->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    /* Upper triangular part of tile with diagonal */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_full_UPPER_TILE_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_Upper, 1, A1->mb, A1->nb, A1->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    /* Little T */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_full_LITTLE_T_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_UpperLower, 1, T1->mb, T1->nb, T1->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    return (dague_handle_t*)handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_full_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgeqrf_full_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_full_New
 * @sa dplasma_zgeqrf_full
 *
 ******************************************************************************/
void
dplasma_zgeqrf_full_Destruct( dague_handle_t *handle )
{
    dague_zgeqrf_full_handle_t *dague_zungqr = (dague_zgeqrf_full_handle_t *)handle;

    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zgeqrf_full_DEFAULT_ARENA   ] );
    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zgeqrf_full_LOWER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zgeqrf_full_UPPER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zgeqrf_full_LITTLE_T_ARENA  ] );

    dague_private_memory_fini( dague_zungqr->p_work );
    free( dague_zungqr->p_work );

    dague_private_memory_fini( dague_zungqr->p_tau );
    free( dague_zungqr->p_tau );

    dague_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_full - Generates an M-by-N matrix Q with orthonormal columns,
 *  which is defined as the first N columns of a product of K elementary
 *  reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_full_New
 * @sa dplasma_zgeqrf_full_Destruct
 * @sa dplasma_cgeqrf_full
 * @sa dplasma_dgeqrf_full
 * @sa dplasma_sgeqrf_full
 * @sa dplasma_zgeqrf
 *
 ******************************************************************************/
int
dplasma_zgeqrf_full( dague_context_t *dague, int optqr, int optid,
                     tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                     tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                     tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 )
{
    dague_handle_t *dague_zungqr;

    if (dague == NULL) {
        dplasma_error("dplasma_zgeqrf_full", "dplasma not initialized");
        return -1;
    }
    if ( Q1->n > Q1->m) {
        dplasma_error("dplasma_zgeqrf_full", "illegal number of columns in Q (N)");
        return -2;
    }
    if ( A1->n > Q1->n) {
        dplasma_error("dplasma_zgeqrf_full", "illegal number of columns in A (K)");
        return -3;
    }
    if ( A1->m != Q1->m ) {
        dplasma_error("dplasma_zgeqrf_full", "illegal number of rows in A");
        return -5;
    }

    if (dplasma_imin(Q1->m, dplasma_imin(Q1->n, A1->n)) == 0)
        return 0;

    dague_zungqr = dplasma_zgeqrf_full_New(optqr, optid,
                                           A1, A2, T1, T2, Q1, Q2 );

    if ( dague_zungqr != NULL ){
        dague_enqueue(dague, dague_zungqr);
        dague_context_wait(dague);
        dplasma_zgeqrf_full_Destruct( dague_zungqr );
    }
    return 0;
}
