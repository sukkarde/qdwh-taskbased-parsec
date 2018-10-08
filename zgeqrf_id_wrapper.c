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
#include "zgeqrf_id.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_id_setrecursive - Set the recursive size parameter to enable
 *  recursive DAGs.
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to modify.
 *          On exit, the modified handle.
 *
 * @param[in] hnb
 *          The tile size to use for the smaller recursive call.
 *          hnb must be > 0, otherwise nothing is changed.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_id_New
 * @sa dplasma_zgeqrf_id
 *
 ******************************************************************************/
void
dplasma_zgeqrf_id_setrecursive( dague_handle_t *handle, int hnb )
{
    dague_zgeqrf_id_handle_t *dague_zgeqrf_id = (dague_zgeqrf_id_handle_t *)handle;

    if (hnb > 0) {
        dague_zgeqrf_id->smallnb = hnb;
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_id_New - Generates the handle that computes the QR factorization
 * a complex M-by-N matrix A: A = Q * R.
 *
 * The method used in this algorithm is a tile QR algorithm with a flat
 * reduction tree.  It is recommended to use the super tiling parameter (SMB) to
 * improve the performance of the factorization.
 * A high SMB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SMB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgeqrf_id_param_New() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgeqrf_id_param_New() parameterized with systolic tree if
 *     computation load per node is very low.
 *
 * WARNING: The computations are not done by this call.
 *
 * If you want to enable the recursive DAGs, don't forget to set the recursive
 * tile size and to synchonize the handle ids after the computations since those
 * are for now local. You can follow the code of dplasma_zgeqrf_id_rec() as an
 * example to do this.
 *
 * Hierarchical DAG Scheduling for Hybrid Distributed Systems; Wu, Wei and
 * Bouteiller, Aurelien and Bosilca, George and Faverge, Mathieu and Dongarra,
 * Jack. 29th IEEE International Parallel & Distributed Processing Symposium,
 * May 2015. (https://hal.inria.fr/hal-0107835)
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeqrf_id_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_id
 * @sa dplasma_zgeqrf_id_Destruct
 * @sa dplasma_cgeqrf_id_New
 * @sa dplasma_dgeqrf_id_New
 * @sa dplasma_sgeqrf_id_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgeqrf_id_New( int optid,
                       tiled_matrix_desc_t *A1,
                       tiled_matrix_desc_t *A2,
                       tiled_matrix_desc_t *T2 )
{
    dague_zgeqrf_id_handle_t* handle;
    int ib = T2->mb;

    handle = dague_zgeqrf_id_new( (dague_ddesc_t*)A1,
                                  (dague_ddesc_t*)A2,
                                  (dague_ddesc_t*)T2,
                                  ib, optid,
                                  NULL, NULL );

    handle->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_tau, T2->nb * sizeof(dague_complex64_t) );

    handle->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_work, ib * T2->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_id_DEFAULT_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_UpperLower, 1, A1->mb, A1->nb, A1->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    /* Lower triangular part of tile without diagonal */
    /* dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_id_LOWER_TILE_ARENA], */
    /*                         dague_datatype_double_complex_t, */
    /*                         matrix_Lower, 0, A1->mb, A1->nb, A1->mb, */
    /*                         DAGUE_ARENA_ALIGNMENT_SSE, -1 ); */

    /* Upper triangular part of tile with diagonal */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_id_UPPER_TILE_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_Upper, 1, A1->mb, A1->nb, A1->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    /* Little T */
    dague_matrix_add2arena( handle->arenas[DAGUE_zgeqrf_id_LITTLE_T_ARENA],
                            dague_datatype_double_complex_t,
                            matrix_UpperLower, 1, T2->mb, T2->nb, T2->mb,
                            DAGUE_ARENA_ALIGNMENT_SSE, -1 );

    return (dague_handle_t*)handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_id_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgeqrf_id_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_id_New
 * @sa dplasma_zgeqrf_id
 *
 ******************************************************************************/
void
dplasma_zgeqrf_id_Destruct( dague_handle_t *handle )
{
    dague_zgeqrf_id_handle_t *dague_zgeqrf_id = (dague_zgeqrf_id_handle_t *)handle;

    dague_matrix_del2arena( dague_zgeqrf_id->arenas[DAGUE_zgeqrf_id_DEFAULT_ARENA   ] );
    //dague_matrix_del2arena( dague_zgeqrf_id->arenas[DAGUE_zgeqrf_id_LOWER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zgeqrf_id->arenas[DAGUE_zgeqrf_id_UPPER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zgeqrf_id->arenas[DAGUE_zgeqrf_id_LITTLE_T_ARENA  ] );

    dague_private_memory_fini( dague_zgeqrf_id->p_work );
    dague_private_memory_fini( dague_zgeqrf_id->p_tau  );
    free( dague_zgeqrf_id->p_work );
    free( dague_zgeqrf_id->p_tau  );

    dague_handle_free(handle);
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_id - Computes the QR factorization a M-by-N matrix A:
 * A = Q * R.
 *
 * The method used in this algorithm is a tile QR algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SMB) to
 * improve the performance of the factorization.
 * A high SMB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SMB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgeqrf_id_param() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgeqrf_id_param() parameterized with systolic tree if computation
 *     load per node is very low.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_id_New
 * @sa dplasma_zgeqrf_id_Destruct
 * @sa dplasma_cgeqrf_id
 * @sa dplasma_dgeqrf_id
 * @sa dplasma_sgeqrf_id
 *
 ******************************************************************************/
int
dplasma_zgeqrf_id( dague_context_t *dague,
                   int optid,
                   tiled_matrix_desc_t *A1,
                   tiled_matrix_desc_t *A2,
                   tiled_matrix_desc_t *T2 )
{
    dague_handle_t *dague_zgeqrf = NULL;

    if ( (A2->mt != T2->mt) || (A2->nt != T2->nt) ) {
        dplasma_error("dplasma_zgeqrf_id", "T2 doesn't have the same number of tiles as A2");
        return -101;
    }

    dague_zgeqrf = dplasma_zgeqrf_id_New(optid, A1, A2, T2);

    if ( dague_zgeqrf != NULL ) {
        dague_enqueue(dague, (dague_handle_t*)dague_zgeqrf);
        dague_context_wait(dague);
        dplasma_zgeqrf_id_Destruct( dague_zgeqrf );
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_id_rec - Computes the QR factorization a M-by-N matrix A:
 * A = Q * R with recursive DAGs.
 *
 * The method used in this algorithm is a tile QR algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SMB) to
 * improve the performance of the factorization.
 * A high SMB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SMB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgeqrf_id_param() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgeqrf_id_param() parameterized with systolic tree if computation
 *     load per node is very low.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 * @param[in] hnb
 *          The tile size to use for the smaller recursive call.
 *          If hnb <= 0 or hnb > A.nb, the classic algorithm without recursive
 *          calls is applied.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_id_New
 * @sa dplasma_zgeqrf_id_Destruct
 * @sa dplasma_cgeqrf_id
 * @sa dplasma_dgeqrf_id
 * @sa dplasma_sgeqrf_id
 *
 ******************************************************************************/
int
dplasma_zgeqrf_id_rec( dague_context_t *dague,
                       int optid,
                       tiled_matrix_desc_t *A1,
                       tiled_matrix_desc_t *A2,
                       tiled_matrix_desc_t *T2, int hnb )
{
    dague_handle_t *dague_zgeqrf = NULL;

    if ( (A2->mt != T2->mt) || (A2->nt != T2->nt) ) {
        dplasma_error("dplasma_zgeqrf_id", "T2 doesn't have the same number of tiles as A2");
        return -101;
    }

    dague_zgeqrf = dplasma_zgeqrf_id_New(optid, A1, A2, T2);

    if ( dague_zgeqrf != NULL ) {
        dague_enqueue( dague, (dague_handle_t*)dague_zgeqrf );
        dplasma_zgeqrf_id_setrecursive( (dague_handle_t*)dague_zgeqrf, hnb );
        dague_context_wait( dague );
        dplasma_zgeqrf_id_Destruct( dague_zgeqrf );
        dague_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */
    }

    return 0;
}
